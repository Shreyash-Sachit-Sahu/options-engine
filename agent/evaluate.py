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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
        results[name] = result
        print(f"   [OK] {name}: Sharpe={result['sharpe_ratio']:.4f}, "
              f"PnL={result['mean_pnl']:.4f}")

    # ─── Evaluate SAC ─────────────────────────────────────────────────────
    model_path = args.model_path
    vnorm_path = args.vnorm_path

    if os.path.exists(model_path + ".zip") or os.path.exists(model_path):
        print(f"\n[SAC] Evaluating SAC agent...")
        actual_path = model_path if os.path.exists(model_path) else model_path
        model = SAC.load(actual_path)
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
    parser.add_argument("--simulator", type=str, default="gbm",
                        choices=["gbm", "heston"])
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

    args = parser.parse_args()
    run_full_evaluation(args)
