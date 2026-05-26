"""
Agent Evaluation & Benchmarking — v2

Evaluates SAC agent vs all five baselines (Delta, Static,
Leland, WhalleyWilmott, Random).

METRIC CHANGE FROM v1:
    Primary  = cross-episode Sharpe = mean(PnL) / std(PnL) * sqrt(252)
    Secondary = mean of per-episode Sharpes (kept for reference only)

    Rationale: per-episode Sharpes over 30-step windows have enormous
    variance (observed range: -6 to +29 for the same agent). Their
    mean is a noisy estimator and produced the misleading result that
    StaticHedger > DeltaHedger in v1. Cross-episode Sharpe is the
    correct metric — it measures whether the agent consistently
    generates positive risk-adjusted PnL across independent scenarios.

Comparison order (most to least theoretically sophisticated):
    SAC > WhalleyWilmott > Leland > Delta > Static > Random
"""

import os, sys, json, argparse
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.gpu_utils import get_device, device_banner
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.options_env import OptionsHedgingEnv
from environment.baselines import (
    DeltaHedger, StaticHedger, RandomAgent,
    LelandHedger, WhalleyWilmottHedger,
    evaluate_agent,
)
from gymnasium import Env
from typing import Any




# ─────────────────────────────────────────────────────────────────────────────
# SAC evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sac(model, vec_normalize_path: str, env_config: dict,
                 n_episodes: int = 500) -> dict:
    """
    Evaluate trained SAC agent.

    Returns cross_episode_sharpe as the primary metric, plus
    mean_episode_sharpe as secondary.
    """
    all_pnls, all_sharpes, all_drawdowns, hedge_errors, all_actions = (
        [], [], [], [], []
    )

    for ep in range(n_episodes):
        ep_env_config: dict[str, Any] = dict(env_config, seed=50000 + ep)
        def make_env()->"Env":
            return OptionsHedgingEnv(**ep_env_config)
        env = DummyVecEnv([make_env])

        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            env.training    = False
            env.norm_reward = False
        else:
            env = VecNormalize(env, norm_obs=True, norm_reward=False,
                               training=False)

        obs, done = env.reset(), False
        ep_deltas = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(float(action[0][0]))

            raw_obs = env.get_original_obs()
            if raw_obs is not None:
                obs_arr = np.array(raw_obs)
                ep_deltas.append(abs(float(obs_arr[0][4]) - float(action[0][0])))

            obs, _, done_arr, infos = env.step(action)
            done = done_arr[0]

        info = infos[0]
        if "total_pnl"     in info: all_pnls.append(info["total_pnl"])
        if "max_drawdown"  in info: all_drawdowns.append(info["max_drawdown"])
        if "episode_sharpe" in info: all_sharpes.append(info["episode_sharpe"])
        if ep_deltas: hedge_errors.append(float(np.sqrt(np.mean(
            np.array(ep_deltas) ** 2))))

    pnls    = np.array(all_pnls)
    pnl_std = float(np.std(pnls)) if len(pnls) > 1 else 1e-6

    # ── PRIMARY metric ────────────────────────────────────────────────────
    cross_ep_sharpe = float(np.mean(pnls) / max(pnl_std, 1e-6) * np.sqrt(252))

    return {
        "agent":                "SAC",
        "n_episodes":           n_episodes,
        "cross_episode_sharpe": cross_ep_sharpe,        # ← PRIMARY
        "mean_episode_sharpe":  float(np.mean(all_sharpes)) if all_sharpes else 0.0,
        "mean_pnl":             float(np.mean(pnls)),
        "std_pnl":              pnl_std,
        "mean_max_drawdown":    float(np.mean(all_drawdowns)) if all_drawdowns else 0.,
        "max_max_drawdown":     float(np.max(all_drawdowns))  if all_drawdowns else 0.,
        "mean_hedge_error":     float(np.mean(hedge_errors))  if hedge_errors  else 0.,
        "action_mean":          float(np.mean(all_actions)),
        "action_std":           float(np.std(all_actions)),
        "episode_pnls":         [float(p) for p in pnls],
        "episode_sharpes":      [float(s) for s in all_sharpes],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(pnls: np.ndarray, n_boot: int = 10_000,
                 ci: float = 95.0, seed: int = 42) -> tuple:
    """Bootstrap CI on cross-episode Sharpe."""
    rng  = np.random.default_rng(seed)
    n    = len(pnls)
    boot = []
    for _ in range(n_boot):
        s = rng.choice(pnls, size=n, replace=True)
        std = float(np.std(s))
        boot.append(float(np.mean(s)) / max(std, 1e-6) * np.sqrt(252))
    lo = (100 - ci) / 2
    return float(np.percentile(boot, lo)), float(np.percentile(boot, 100 - lo))


def paired_test(pnls_a: np.ndarray, pnls_b: np.ndarray) -> dict:
    """Paired t-test on episode PnLs (not Sharpes — PnL is the raw quantity)."""
    n      = min(len(pnls_a), len(pnls_b))
    diff   = pnls_a[:n] - pnls_b[:n]
    t, p   = scipy_stats.ttest_rel(pnls_a[:n], pnls_b[:n])
    ci     = scipy_stats.t.interval(0.95, df=n-1,
                                     loc=np.mean(diff),
                                     scale=scipy_stats.sem(diff))
    return {
        "n":               n,
        "mean_diff":       float(np.mean(diff)),
        "t_stat":          float(t),
        "p_value":         float(p),
        "significant_5pct": bool(p < 0.05),
        "ci_95_lo":        float(ci[0]),
        "ci_95_hi":        float(ci[1]),
    }


def tc_sensitivity_sweep(model, vnorm_path: str, base_cfg: dict,
                          tc_levels=None, n_episodes: int = 200) -> dict:
    """
    Test SAC and Leland hedger across TC levels.
    Answers: 'at what TC does the RL alpha vanish?'
    """
    if tc_levels is None:
        tc_levels = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005]

    results = {}
    for tc in tc_levels:
        cfg : dict[str, Any] = dict(base_cfg, transaction_cost=tc)
        sac  = evaluate_sac(model, vnorm_path, cfg, n_episodes)
        env  = OptionsHedgingEnv(**cfg)

        leland = LelandHedger(sigma=cfg["sigma"], tc_rate=tc,
                              K=cfg["K"], r=cfg["r"], T=cfg["T"])
        delta  = DeltaHedger(K=cfg["K"], r=cfg["r"],
                              sigma=cfg["sigma"], T=cfg["T"])

        lel_res   = evaluate_agent(env, leland, n_episodes)
        delta_res = evaluate_agent(env, delta,  n_episodes)

        sac_sharpe   = sac["cross_episode_sharpe"]
        lel_sharpe   = lel_res["cross_episode_sharpe"]
        delta_sharpe = delta_res["cross_episode_sharpe"]

        results[tc] = {
            "sac_sharpe":   sac_sharpe,
            "leland_sharpe": lel_sharpe,
            "delta_sharpe": delta_sharpe,
            "sac_vs_leland_pct":
                (sac_sharpe / max(lel_sharpe, 1e-10) - 1) * 100,
            "sac_vs_delta_pct":
                (sac_sharpe / max(delta_sharpe, 1e-10) - 1) * 100,
        }
        print(f"  TC={tc:.4f}: SAC={sac_sharpe:.3f}  "
              f"Leland={lel_sharpe:.3f}  Delta={delta_sharpe:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_model(args):
    paths = [
        args.model_path,
        "agent/models/best_heston/best_model",
        "agent/models/sac_hedger_heston_final",
        "agent/models/sac_hedger_final",
    ]
    for p in paths:
        if os.path.exists(p) or os.path.exists(p + ".zip"):
            return p
    return paths[0]


def _resolve_vnorm(args):
    paths = [
        args.vnorm_path,
        "agent/models/vec_normalize_heston.pkl",
        "agent/models/vec_normalize.pkl",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


def run_full_evaluation(args) -> dict:
    n_ep = args.n_episodes
    sim  = args.simulator

    env_config: dict[str, Any] = dict(
        simulator_type   = sim,
        S0=100., K=100., T=30/252, r=0.05, sigma=0.2,
        n_steps=30, transaction_cost=0.003,
        execution_delay=1, variable_tc=True,
    )
    if sim == "heston":
        env_config.update(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)

    device_banner()
    results = {}

    # ── Baseline evaluations ──────────────────────────────────────────────
    baseline_specs = [
        ("DeltaHedger",        DeltaHedger(sigma=0.2, K=100., r=0.05, T=30/252)),
        ("LelandHedger",       LelandHedger(sigma=0.2, tc_rate=0.003,
                                            K=100., r=0.05, T=30/252)),
        ("WhalleyWilmottHedger", WhalleyWilmottHedger(sigma=0.2, tc_rate=0.003,
                                                       K=100., r=0.05, T=30/252)),
        ("StaticHedger",       StaticHedger()),
        ("RandomAgent",        RandomAgent(42)),
    ]

    for name, agent in baseline_specs:
        print(f"\n[BASELINE] Evaluating {name}...")
        env = OptionsHedgingEnv(**env_config)
        res = evaluate_agent(env, agent, n_episodes=n_ep)
        res["agent"] = name
        results[name] = res
        print(f"   Cross-ep Sharpe: {res['cross_episode_sharpe']:.4f}  "
              f"Mean PnL: {res['mean_pnl']:.4f}")

    # ── SAC evaluation ────────────────────────────────────────────────────
    model_path = _resolve_model(args)
    vnorm_path = _resolve_vnorm(args)
    print(f"\n[MODEL] {model_path}")
    print(f"[VNORM] {vnorm_path}")

    if os.path.exists(model_path) or os.path.exists(model_path + ".zip"):
        device = get_device(verbose=True)
        model  = SAC.load(model_path, device=device)
        sac    = evaluate_sac(model, vnorm_path, env_config, n_ep)
        results["SAC"] = sac
        print(f"\n[SAC] Cross-ep Sharpe: {sac['cross_episode_sharpe']:.4f}  "
              f"Mean PnL: {sac['mean_pnl']:.4f}")
    else:
        print(f"\n[WARN] No model at {model_path} — skipping SAC eval.")

    # ── Comparison table ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS  (PRIMARY = cross-episode Sharpe)")
    print("=" * 72)
    print(f"\n{'Agent':<22} {'XEP Sharpe':>12} {'Mean PnL':>12} "
          f"{'Std PnL':>10} {'MaxDD':>10}")
    print("-" * 72)

    rank_order = ["SAC", "WhalleyWilmottHedger", "LelandHedger",
                  "DeltaHedger", "StaticHedger", "RandomAgent"]
    for name in rank_order:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<22} "
              f"{r['cross_episode_sharpe']:>12.4f} "
              f"{r['mean_pnl']:>12.4f} "
              f"{r['std_pnl']:>10.4f} "
              f"{r.get('mean_max_drawdown', 0):>10.4f}")

    # ── Statistical comparison: SAC vs each classical baseline ───────────
    stats_block = {}
    if "SAC" in results:
        sac_pnls = np.array(results["SAC"]["episode_pnls"])
        for name in ["DeltaHedger", "LelandHedger", "WhalleyWilmottHedger"]:
            if name not in results:
                continue
            other_pnls = np.array(results[name]["episode_pnls"])
            st = paired_test(sac_pnls, other_pnls)
            stats_block[f"SAC_vs_{name}"] = st
            sig = "✓" if st["significant_5pct"] else "✗"
            print(f"\n[STATS] SAC vs {name}: "
                  f"mean_diff={st['mean_diff']:+.4f}  "
                  f"p={st['p_value']:.4f}  {sig}")

        # Bootstrap CI on SAC cross-episode Sharpe
        ci_lo, ci_hi = bootstrap_ci(sac_pnls)
        print(f"\n[CI] SAC cross-ep Sharpe 95% bootstrap CI: "
              f"[{ci_lo:.3f}, {ci_hi:.3f}]")
        results["statistics"] = stats_block

    # ── TC sensitivity (optional) ─────────────────────────────────────────
    if "SAC" in results and not args.skip_tc_sweep:
        print("\n[TC SWEEP]")
        model_path = _resolve_model(args)
        device     = get_device(verbose=False)
        model      = SAC.load(model_path, device=device)
        tc_res     = tc_sensitivity_sweep(model, vnorm_path, env_config,
                                          n_episodes=200)
        results["tc_sensitivity"] = {str(k): v for k, v in tc_res.items()}

    # ── Save ─────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else x)
    print(f"\n[SAVE] {out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",        default="auto",
                        choices=["auto","cuda","mps","cpu"])
    parser.add_argument("--simulator",     default="heston",
                        choices=["gbm","heston","jump","regime"])
    parser.add_argument("--n-episodes",    type=int, default=500)
    parser.add_argument("--model-path",    default="agent/models/best_heston/best_model")
    parser.add_argument("--vnorm-path",    default="agent/models/vec_normalize_heston.pkl")
    parser.add_argument("--output",        default="agent/evaluation_results.json")
    parser.add_argument("--skip-tc-sweep", action="store_true")
    args = parser.parse_args()
    run_full_evaluation(args)