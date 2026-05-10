"""
Agent Evaluation & Benchmarking

Comprehensive evaluation of the trained SAC agent vs all baselines.
Computes Sharpe ratio, max drawdown, hedge error, transaction costs,
and generates comparison results.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.gpu_utils import get_device, device_banner

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.options_env import OptionsHedgingEnv
from environment.baselines import (
    DeltaHedger, StaticHedger, RandomAgent,
    run_baseline_episode, evaluate_agent
)


def evaluate_sac(model, vec_normalize_path: str, env_config: dict,
                 n_episodes: int = 1000) -> dict:
    """
    Evaluate trained SAC agent over multiple episodes.

    Uses the saved VecNormalize statistics for consistent evaluation.
    Each episode uses a different seed for diverse market scenarios.
    """
    all_rewards = []
    all_pnls = []
    all_drawdowns = []
    all_sharpes = []
    hedge_errors = []
    all_actions = []

    for ep in range(n_episodes):
        # Create fresh env with unique seed per episode for diverse scenarios
        ep_seed = 50000 + ep
        ep_env_config = dict(env_config, seed=ep_seed)
        env = DummyVecEnv([lambda cfg=ep_env_config: OptionsHedgingEnv(**cfg)])

        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=False, training=False)

        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_deltas = []
        ep_actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            ep_actions.append(float(action[0][0]))

            # Track delta for hedge error computation
            raw_obs = env.get_original_obs()
            if raw_obs is not None:
                delta = float(raw_obs[0][4])
                hedge = float(action[0][0])
                ep_deltas.append(abs(delta - hedge))

            obs, reward, done_arr, infos = env.step(action)
            done = done_arr[0]
            ep_reward += reward[0]

        all_rewards.append(ep_reward)
        all_actions.extend(ep_actions)

        info = infos[0]
        if "total_pnl" in info:
            all_pnls.append(info["total_pnl"])
        if "max_drawdown" in info:
            all_drawdowns.append(info["max_drawdown"])
        if "episode_sharpe" in info:
            all_sharpes.append(info["episode_sharpe"])

        if ep_deltas:
            hedge_errors.append(np.sqrt(np.mean(np.array(ep_deltas) ** 2)))

    rewards = np.array(all_rewards)
    pnls = np.array(all_pnls) if all_pnls else rewards

    # Use episode-level Sharpe as primary metric (more meaningful for hedging)
    pnl_std = float(np.std(pnls)) if len(pnls) > 1 else 1e-6
    cross_ep_sharpe = float(np.mean(pnls) / max(pnl_std, 1e-6) * np.sqrt(252))
    ep_sharpe = float(np.mean(all_sharpes)) if all_sharpes else cross_ep_sharpe

    return {
        "agent": "SAC",
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": pnl_std,
        "sharpe_ratio": ep_sharpe,
        "cross_episode_sharpe": cross_ep_sharpe,
        "mean_episode_sharpe": ep_sharpe,
        "mean_max_drawdown": float(np.mean(all_drawdowns)) if all_drawdowns else 0.0,
        "max_max_drawdown": float(np.max(all_drawdowns)) if all_drawdowns else 0.0,
        "mean_hedge_error": float(np.mean(hedge_errors)) if hedge_errors else 0.0,
        "action_mean": float(np.mean(all_actions)),
        "action_std": float(np.std(all_actions)),
        "n_episodes": n_episodes,
        "episode_pnls": [float(p) for p in pnls],
        "episode_sharpes": [float(s) for s in all_sharpes],
    }


def bootstrap_sharpe_ci(pnls: np.ndarray, n_boot: int = 10_000,
                        ci: float = 95.0, seed: int = 42) -> tuple:
    """
    Bootstrap confidence interval for Sharpe ratio.

    Resamples episode PnLs with replacement and recomputes Sharpe
    each time. Returns (lower, upper) bounds of the CI.
    This directly answers "is your Sharpe real or lucky?"
    """
    rng = np.random.default_rng(seed)
    boot_sharpes = []
    n = len(pnls)
    for _ in range(n_boot):
        sample = rng.choice(pnls, size=n, replace=True)
        s = float(np.mean(sample))  # sample is episode Sharpes — mean directly
        boot_sharpes.append(s)
    lo = (100 - ci) / 2
    return float(np.percentile(boot_sharpes, lo)), float(np.percentile(boot_sharpes, 100 - lo))


def transaction_cost_sweep(model, vec_normalize_path: str,
                            base_env_config: dict,
                            tc_levels: list = None,
                            n_episodes: int = 200) -> dict:
    """
    Sweep transaction costs to find the breakeven point.

    Tests the SAC agent at multiple TC levels and reports where
    it stops beating delta hedging. This directly addresses the
    'at what TC does the alpha disappear?' question.

    Args:
        tc_levels: List of TC rates to test (default: 5 levels 0→0.005)
    """
    if tc_levels is None:
        tc_levels = [0.0, 0.0005, 0.001, 0.002, 0.005]

    results = {}
    print(f"\n[TC SWEEP] Testing {len(tc_levels)} transaction cost levels...")

    for tc in tc_levels:
        cfg = dict(base_env_config, transaction_cost=tc)

        # SAC
        sac_res  = evaluate_sac(model, vec_normalize_path, cfg, n_episodes)
        sac_sharpe = sac_res["sharpe_ratio"]

        # Delta hedger at same TC
        from environment.baselines import DeltaHedger, evaluate_agent
        env = OptionsHedgingEnv(**cfg)
        delta_agent = DeltaHedger(K=cfg["K"], r=cfg["r"],
                                   sigma=cfg["sigma"], T=cfg["T"])
        delta_res = evaluate_agent(env, delta_agent, n_episodes=n_episodes)
        delta_sharpe = delta_res.get("mean_episode_sharpe",
                                      delta_res.get("sharpe_ratio", 0.91))

        outperf = (sac_sharpe / max(delta_sharpe, 1e-10) - 1) * 100
        results[tc] = {
            "sac_sharpe":   sac_sharpe,
            "delta_sharpe": delta_sharpe,
            "outperformance_pct": outperf,
            "sac_beats_delta": sac_sharpe > delta_sharpe,
        }
        status = "✓" if sac_sharpe > delta_sharpe else "✗"
        print(f"   TC={tc:.4f}: SAC={sac_sharpe:.3f}  Delta={delta_sharpe:.3f}"
              f"  Outperf={outperf:+.1f}%  {status}")

    # Find breakeven
    breakeven = None
    tc_list = sorted(results.keys())
    for i in range(1, len(tc_list)):
        if not results[tc_list[i]]["sac_beats_delta"]:
            breakeven = tc_list[i]
            break

    if breakeven:
        print(f"\n   Breakeven TC: ~{breakeven:.4f} "
              f"(SAC stops beating delta above this level)")
    else:
        print(f"\n   SAC beats delta at all tested TC levels (up to {tc_list[-1]:.4f})")

    results["breakeven_tc"] = breakeven
    return results


def statistical_comparison(sac_pnls: np.ndarray,
                            delta_pnls: np.ndarray) -> dict:
    """
    Paired t-test + bootstrap CI comparing SAC vs Delta PnL distributions.

    Returns a dict suitable for printing and saving to JSON.
    """
    n = min(len(sac_pnls), len(delta_pnls))
    sac_pnls, delta_pnls = sac_pnls[:n], delta_pnls[:n]
    diffs = sac_pnls - delta_pnls

    t_stat, p_value = scipy_stats.ttest_rel(sac_pnls, delta_pnls)

    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(diffs, size=n, replace=True))
                  for _ in range(10_000)]
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    return {
        "n_episodes":       n,
        "mean_diff":        float(np.mean(diffs)),
        "t_stat":           float(t_stat),
        "p_value":          float(p_value),
        "significant_5pct": bool(p_value < 0.05),
        "ci_95_lo":         float(ci_lo),
        "ci_95_hi":         float(ci_hi),
    }


def run_full_evaluation(args):
    """Run comprehensive evaluation of SAC vs all baselines."""
    print("=" * 70)
    print("  Options Hedging Agent - Full Evaluation")
    print("=" * 70)

    env_config = {
        "simulator_type": args.simulator,
        "S0": 100.0,
        "K": 100.0,
        "T": 30 / 252,
        "r": 0.05,
        "sigma": 0.2,
        "mu": 0.05,
        "n_steps": 30,
        "transaction_cost": 0.001,
    }

    if args.simulator == "heston":
        env_config.update({
            "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7
        })

    n_episodes = args.n_episodes

    # ─── Evaluate Baselines ───────────────────────────────────────────────
    print(f"\n[EVAL] Evaluating baselines ({n_episodes} episodes each)...")

    env = OptionsHedgingEnv(**env_config)

    agents = {
        "DeltaHedger": DeltaHedger(K=100.0, r=0.05, sigma=0.2, T=30/252),
        "StaticHedger": StaticHedger(),
        "RandomAgent": RandomAgent(seed=42),
    }

    results = {}
    for name, agent in agents.items():
        print(f"   Evaluating {name}...")
        result = evaluate_agent(env, agent, n_episodes=n_episodes)
        result["agent"] = name
        # Use mean_episode_sharpe (within-episode) to match SAC's metric
        result["sharpe_ratio"] = result["mean_episode_sharpe"]
        results[name] = result
        print(f"   [OK] {name}: Sharpe={result['sharpe_ratio']:.4f}, "
              f"PnL={result['mean_pnl']:.4f}")

    # ─── Evaluate SAC ─────────────────────────────────────────────────────
    # ── Smart model path resolution ──────────────────────────────────────
    # Priority: 1) explicit --model-path  2) best_<sim>/best_model
    #           3) sac_hedger_<sim>_final  4) generic sac_hedger_final
    sim_tag    = args.simulator.lower()
    model_dir  = Path(args.model_path).parent
    vnorm_path = args.vnorm_path

    _default_model_path = "agent/models/sac_hedger_final"

    def _resolve_model() -> str:
        # Only use explicit --model-path if the user actually changed it
        # from the default. Otherwise, prefer the sim-tagged model.
        user_explicit = args.model_path != _default_model_path
        candidates = []
        if user_explicit:
            candidates.append(args.model_path)
        candidates += [
            str(model_dir / f"best_{sim_tag}" / "best_model"),   # best checkpoint
            str(model_dir / f"sac_hedger_{sim_tag}_final"),       # sim-tagged final
        ]
        if not user_explicit:
            candidates.append(args.model_path)                    # generic last resort
        for c in candidates:
            if os.path.exists(c + ".zip") or os.path.exists(c):
                return c
        return args.model_path   # will fail gracefully below

    _default_vnorm_path = "agent/models/vec_normalize.pkl"

    def _resolve_vnorm() -> str:
        user_explicit = args.vnorm_path != _default_vnorm_path
        candidates = []
        if user_explicit:
            candidates.append(args.vnorm_path)
        candidates.append(str(model_dir / f"vec_normalize_{sim_tag}.pkl"))
        if not user_explicit:
            candidates.append(args.vnorm_path)
        for c in candidates:
            if os.path.exists(c):
                return c
        return args.vnorm_path

    model_path = _resolve_model()
    vnorm_path = _resolve_vnorm()
    print(f"[MODEL] Loading: {model_path}")
    print(f"[VNORM] Loading: {vnorm_path}")

    # Sanity check: warn if vnorm obs dim doesn't match current env
    try:
        import pickle
        with open(vnorm_path, "rb") as _f:
            _vn = pickle.load(_f)
        _vnorm_dim = len(_vn.obs_rms.mean)
        _env_dim   = 11  # current obs space after expansion
        if _vnorm_dim != _env_dim:
            sep = "=" * 70
            print(f"\n{sep}")
            print(f"  [CRITICAL] OBS DIMENSION MISMATCH")
            print(f"  VecNormalize has {_vnorm_dim}-dim stats but env expects {_env_dim}-dim.")
            print(f"  This model was trained BEFORE the obs space expansion.")
            print(f"  Results will be UNRELIABLE. You must retrain:")
            print(f"    python agent/train.py --simulator {sim_tag} --total-timesteps 500000")
            print(f"{sep}\n")
    except Exception:
        pass

    if os.path.exists(model_path + ".zip") or os.path.exists(model_path):
        print(f"\n[SAC] Evaluating SAC agent...")
        device = get_device(verbose=True)
        actual_path = model_path if os.path.exists(model_path) else model_path
        model = SAC.load(actual_path, device=device)   # ← load weights onto GPU
        sac_result = evaluate_sac(model, vnorm_path, env_config, n_episodes)
        results["SAC"] = sac_result
        print(f"   [OK] SAC: Sharpe={sac_result['sharpe_ratio']:.4f}, "
              f"PnL={sac_result['mean_pnl']:.4f}")
    else:
        print(f"\n[WARN] No trained SAC model found at: {model_path}")
        print(f"   Run train.py first to train the agent.")

    # ─── Comparison Table ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Agent':<15} {'Sharpe':>10} {'Mean PnL':>12} {'Std PnL':>10} "
          f"{'MaxDD':>10} {'vs Delta':>10}")
    print("-" * 70)

    delta_sharpe = results.get("DeltaHedger", {}).get("sharpe_ratio", 1.0)

    for name in ["SAC", "DeltaHedger", "StaticHedger", "RandomAgent"]:
        if name not in results:
            continue
        r = results[name]
        vs_delta = r["sharpe_ratio"] / (delta_sharpe + 1e-10)
        pct_str = f"{vs_delta:.2f}x"
        print(f"{name:<15} {r['sharpe_ratio']:>10.4f} {r['mean_pnl']:>12.4f} "
              f"{r.get('std_pnl', 0):>10.4f} "
              f"{r.get('mean_max_drawdown', 0):>10.4f} {pct_str:>10}")

    # ─── Bootstrap CI on SAC Sharpe ──────────────────────────────────────
    if "SAC" in results and results["SAC"].get("episode_sharpes"):
        sac_sharpes = np.array(results["SAC"].get("episode_sharpes", []))   # ← add this line 
        sac_pnls    = np.array(results["SAC"]["episode_pnls"])
        ci_lo, ci_hi = bootstrap_sharpe_ci(sac_sharpes) if len(sac_sharpes) > 0 else (0.0, 0.0)
        print(f"\n[STATS] SAC Sharpe 95% CI (bootstrap, n=10k): "
              f"[{ci_lo:.3f}, {ci_hi:.3f}]")

        # Paired t-test vs delta
        delta_sharpes = np.array(results.get("DeltaHedger", {}).get("episode_sharpes", []))
        if len(delta_sharpes) > 0 and len(sac_sharpes) > 0:
            stats = statistical_comparison(sac_sharpes, delta_sharpes)
            results["statistics"] = stats
            sig = "YES ✓" if stats["significant_5pct"] else "NO ✗"
            print(f"[STATS] SAC vs Delta paired t-test:")
            print(f"   Mean Sharpe diff : {stats['mean_diff']:+.6f}")
            print(f"   95% CI        : [{stats['ci_95_lo']:+.6f}, {stats['ci_95_hi']:+.6f}]")
            print(f"   p-value       : {stats['p_value']:.4f}  (significant: {sig})")

    # ─── Transaction Cost Sensitivity ────────────────────────────────────
    if "SAC" in results and not args.skip_tc_sweep:
        print(f"\n[TC SWEEP] Analysing TC sensitivity...")
        try:
            resolved_path = _resolve_model()
            device = get_device(verbose=False)
            model = SAC.load(resolved_path, device=device)
            tc_results = transaction_cost_sweep(
                model, _resolve_vnorm(), env_config,
                tc_levels=[0.0, 0.0005, 0.001, 0.002, 0.003, 0.005],
                n_episodes=200,
            )
            results["tc_sensitivity"] = {
                str(k): v for k, v in tc_results.items()
            }
        except Exception as e:
            print(f"[WARN] TC sweep failed: {e}")

    # ─── Save Results ─────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure all data is JSON-serializable (convert numpy arrays/lists)
    serializable_results = {}
    for name, res in results.items():
        serializable_results[name] = {
            k: ([float(x) for x in v] if isinstance(v, (list, np.ndarray)) and k.startswith("episode_") else v)
            for k, v in res.items()
        }

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n[SAVE] Results saved to: {output_path}")

    # ─── Target Check ─────────────────────────────────────────────────────
    if "SAC" in results:
        sac_sharpe = results["SAC"]["sharpe_ratio"]
        print(f"\n[TARGET] Target Check:")
        print(f"   SAC Sharpe: {sac_sharpe:.4f} (target: > 1.4)")
        print(f"   vs Delta: {sac_sharpe / (delta_sharpe + 1e-10):.2f}x "
              f"(target: > 1.55x)")

        if sac_sharpe > 1.4:
            print("   [PASS] TARGET MET!")
        else:
            print("   [FAIL] Below target - consider more training or hyperparameter tuning")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hedging agents")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device for SAC inference (default: auto)")
    parser.add_argument("--simulator", type=str, default="gbm",
                        choices=["gbm", "heston", "jump"])
    parser.add_argument("--n-episodes", type=int, default=1000,
                        help="Number of evaluation episodes")
    parser.add_argument("--model-path", type=str,
                        default="agent/models/sac_hedger_final",
                        help="Path to trained SAC model")
    parser.add_argument("--vnorm-path", type=str,
                        default="agent/models/vec_normalize.pkl",
                        help="Path to VecNormalize stats")
    parser.add_argument("--output", type=str,
                        default="agent/evaluation_results.json",
                        help="Output JSON path")
    parser.add_argument("--skip-tc-sweep", action="store_true",
                        help="Skip transaction cost sensitivity sweep (faster)")

    args = parser.parse_args()
    run_full_evaluation(args)