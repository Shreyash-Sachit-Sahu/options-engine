"""
Baseline Hedging Agents

Three baseline agents for rigorous comparison against the RL agent:
1. DeltaHedger:  Rebalances daily to maintain delta-neutral portfolio
2. StaticHedger: Buys initial delta hedge, never rebalances
3. RandomAgent:  Takes random actions as a lower bound

All agents use the same pricer functions for fair comparison.
"""

import numpy as np
import sys
import os
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pricer.pricer_py import greeks_call


class DeltaHedger:
    """
    Classical delta-hedging baseline.

    At each time step, computes the BS delta of the short call position
    and adjusts the hedge position to be delta-neutral. This is the
    industry standard benchmark for hedging performance.

    The agent sets hedge_ratio = delta (from Black-Scholes).
    """

    def __init__(self, K: float = 100.0, r: float = 0.05,
                 sigma: float = 0.2, T: float = 30/252, n_steps: int = 30):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.step_count = 0

    def reset(self):
        """Reset step counter."""
        self.step_count = 0

    def predict(self, obs: np.ndarray, **kwargs) -> tuple:
        """
        Predict hedge action from observation.

        Matches stable-baselines3 model.predict() interface.

        Args:
            obs: [S/S0, K/S0, T_rem/T, sigma, delta, gamma, pnl]

        Returns:
            (action, None) — action is a numpy array with shape (1,)
        """
        # Extract delta from observation (index 4)
        delta = float(obs[4]) if obs.ndim == 1 else float(obs[0, 4])

        # Clip to [-1, 1] (our action space)
        action = np.clip(delta, -1.0, 1.0)

        self.step_count += 1
        return np.array([action], dtype=np.float32), None

    def predict_from_state(self, S: float, K: float, T: float,
                           r: float, sigma: float) -> float:
        """Direct prediction from market state (bypasses observation)."""
        if T <= 0:
            return 1.0 if S > K else 0.0

        g = greeks_call(S, K, T, r, sigma)
        return float(np.clip(g.delta, -1.0, 1.0))


class StaticHedger:
    """
    Static (buy-and-hold) delta hedge.

    Computes the initial delta at inception and holds that position
    throughout the option's life. Never rebalances.

    This represents the upper bound on transaction cost savings
    but poor hedge accuracy — useful as a baseline.
    """

    def __init__(self, initial_delta: Optional[float] = None):
        self.initial_delta = initial_delta
        self._initialized = False

    def reset(self):
        """Reset for new episode."""
        self._initialized = False

    def predict(self, obs: np.ndarray, **kwargs) -> tuple:
        """
        Predict hedge action.

        On the first call, locks in the delta from observations.
        All subsequent calls return the same action.
        """
        if not self._initialized:
            delta = float(obs[4]) if obs.ndim == 1 else float(obs[0, 4])
            self.initial_delta = np.clip(delta, -1.0, 1.0)
            self._initialized = True

        action = np.array([self.initial_delta], dtype=np.float32)
        return action, None


class RandomAgent:
    """
    Random hedging agent.

    Takes uniformly random hedge actions at each step.
    Lower bound on performance — any reasonable strategy should beat this.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, **kwargs) -> tuple:
        """Random action in [-1, 1]."""
        action = self.rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        return action, None


def run_baseline_episode(env, agent, seed: Optional[int] = None) -> dict:
    """
    Run a single episode with a baseline agent.

    Args:
        env: OptionsHedgingEnv instance
        agent: Any agent with .predict(obs) -> (action, state)
        seed: Random seed

    Returns:
        Episode results dict with PnL, Sharpe, etc.
    """
    obs, info = env.reset(seed=seed)
    if hasattr(agent, 'reset'):
        agent.reset()

    total_reward = 0.0
    done = False
    step_pnls = []
    hedge_positions = []

    while not done:
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        if 'total_pnl' in info:
            pass  # will get at end
        hedge_positions.append(float(action[0]))

    # Collect final metrics
    result = {
        "total_reward": total_reward,
        "portfolio_value": info.get("portfolio_value", 0.0),
        "episode_sharpe": info.get("episode_sharpe", 0.0),
        "total_pnl": info.get("total_pnl", 0.0),
        "max_drawdown": info.get("max_drawdown", 0.0),
        "mean_hedge": float(np.mean(hedge_positions)),
        "hedge_std": float(np.std(hedge_positions)),
        "n_trades": len(hedge_positions),
    }

    return result


def evaluate_agent(env, agent, n_episodes: int = 1000,
                   seed_start: int = 0) -> dict:
    """
    Evaluate an agent over multiple episodes.

    Returns aggregated metrics and raw per-episode PnL data
    for dashboard visualization.
    """
    results = []
    for i in range(n_episodes):
        result = run_baseline_episode(env, agent, seed=seed_start + i)
        results.append(result)

    rewards = [r["total_reward"] for r in results]
    pnls = [r["total_pnl"] for r in results]
    sharpes = [r["episode_sharpe"] for r in results]
    drawdowns = [r["max_drawdown"] for r in results]

    pnl_std = float(np.std(pnls)) if len(pnls) > 1 else 1e-6

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": pnl_std,
        "sharpe_ratio": float(np.mean(pnls) / max(pnl_std, 1e-6) * np.sqrt(252)),
        "mean_episode_sharpe": float(np.mean(sharpes)),
        "mean_max_drawdown": float(np.mean(drawdowns)),
        "max_max_drawdown": float(np.max(drawdowns)),
        "n_episodes": n_episodes,
        "episode_pnls": pnls,       # raw episode PnLs for dashboard
        "episode_sharpes": sharpes,  # raw episode Sharpes for dashboard
    }
