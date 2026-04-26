"""
Options Hedging Gymnasium Environment

A custom Gymnasium environment for training an RL agent to dynamically
hedge an options portfolio. The agent observes market state + Greeks and
decides the hedge ratio for the underlying position.

Compatible with stable-baselines3 SAC (continuous action space).

Reference architecture:
- Buehler et al. (2019) "Deep Hedging"
- Cao et al. (2021) "Deep Hedging of Derivatives Using RL"
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
from typing import Optional, Dict, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pricer.pricer_py import bs_call, greeks_call
from environment.market_sim import GBMSimulator, HestonSimulator


class OptionsHedgingEnv(gym.Env):
    """
    Options Hedging Environment for Reinforcement Learning.

    The agent sells an at-the-money call option and must hedge it by
    trading the underlying asset. The goal is to minimise PnL variance
    (i.e., hedge as perfectly as possible) while managing transaction costs.

    Observation Space (7-dim):
        [0] S/S0           — normalised spot price
        [1] K/S0           — normalised strike
        [2] T_rem/T_total  — remaining time fraction
        [3] sigma          — current implied/realised volatility
        [4] delta          — Black-Scholes delta
        [5] gamma          — Black-Scholes gamma
        [6] portfolio_pnl  — normalised portfolio PnL

    Action Space:
        Box([-1], [1]) — delta adjustment
          The agent outputs a correction to the BS delta:
            target_hedge = clip(delta + 0.3 * action, -1, 1)
          action =  0 -> pure delta hedge
          action =  1 -> over-hedge by 0.3
          action = -1 -> under-hedge by 0.3

    Reward (5 components):
        -0.1 * pnl^2          variance penalty
        +0.02 * pnl           small PnL incentive
        -tc                   transaction cost
        -0.5 * hedge_error    core: distance from BS delta
        -0.01 * |dposition|   smooth trading bonus

    Episode:
        30 trading days (option lifetime). Terminates early if
        portfolio value drops below -3 x initial premium.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        simulator_type: str = "gbm",
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 30 / 252,
        r: float = 0.05,
        sigma: float = 0.2,
        mu: float = 0.05,
        n_steps: int = 30,
        transaction_cost: float = 0.001,
        seed: Optional[int] = None,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.tc_rate = transaction_cost

        self.simulator_type = simulator_type
        if simulator_type == "heston":
            self.simulator = HestonSimulator(
                S0=S0, mu=mu, v0=sigma**2, kappa=kappa,
                theta=theta, xi=xi, rho=rho, dt=self.dt, seed=seed
            )
        else:
            self.simulator = GBMSimulator(
                S0=S0, mu=mu, sigma=sigma, dt=self.dt, seed=seed
            )

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -np.inf], dtype=np.float32),
            high=np.array([5.0, 5.0, 1.0, 5.0,  1.0, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.current_step = 0
        self.price = S0
        self.current_vol = sigma
        self.hedge_position = 0.0
        self.portfolio_value = 0.0
        self.option_premium = 0.0
        self.pnl_history = []
        self.prev_portfolio_value = 0.0

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        if seed is not None:
            self.price = self.simulator.reset(seed=seed)
        else:
            self.price = self.simulator.reset()

        self.current_step = 0
        self.current_vol = self.sigma
        self.hedge_position = 0.0
        self.pnl_history = []

        self.option_premium = bs_call(self.price, self.K, self.T, self.r, self.sigma)
        self.portfolio_value = self.option_premium
        self.prev_portfolio_value = self.portfolio_value

        return self._get_obs(), self._get_info(episode_done=False)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.

        1. Compute BS delta at current state
        2. Apply hedge as delta + 0.3 * rl_adjustment
        3. Deduct transaction cost for position change
        4. Simulate next market state
        5. Mark-to-market portfolio
        6. Compute shaped reward
        """
        self.current_step += 1
        old_price = self.price
        T_remaining = max(self.T - self.current_step * self.dt, 1e-10)

        # ── 1. BS delta at current state ──────────────────────────────────
        try:
            g = greeks_call(old_price, self.K, T_remaining + self.dt,
                            self.r, self.current_vol)
            delta = g.delta
        except Exception:
            delta = 0.5

        # ── 2. Target hedge: delta + small RL correction ──────────────────
        rl_adjustment = float(action[0])
        target_hedge = float(np.clip(delta + 0.3 * rl_adjustment, -1.0, 1.0))

        # ── 3. Transaction cost ───────────────────────────────────────────
        position_change = abs(target_hedge - self.hedge_position)
        tc = self.tc_rate * position_change * old_price
        self.portfolio_value -= tc
        old_hedge = self.hedge_position
        self.hedge_position = target_hedge

        # ── 4. Simulate next price ────────────────────────────────────────
        self.price, self.current_vol = self.simulator.step()

        # ── 5. Mark-to-market ─────────────────────────────────────────────
        price_change = self.price - old_price

        # PnL from hedge position (long underlying)
        hedge_pnl = self.hedge_position * price_change

        # PnL from short call (mark-to-market)
        old_option_val = bs_call(old_price, self.K,
                                 T_remaining + self.dt, self.r, self.current_vol)
        new_option_val = bs_call(self.price, self.K,
                                 T_remaining, self.r, self.current_vol)
        option_pnl = -(new_option_val - old_option_val)

        # FIX: store actual dollar PnL per step — no standardisation
        total_pnl = hedge_pnl + option_pnl
        self.portfolio_value += total_pnl
        self.pnl_history.append(total_pnl)

        # ── 6. Reward ──────────────────────────────────────────────────────
        # Re-compute delta at new state for hedge error signal
        try:
            g_new = greeks_call(self.price, self.K, T_remaining,
                                self.r, self.current_vol)
            delta_new = g_new.delta
        except Exception:
            delta_new = 0.5

        hedge_error = abs(delta_new - self.hedge_position)

        reward  = -0.1 * (total_pnl ** 2)                          # variance penalty
        reward += 0.02 * total_pnl                                  # PnL incentive
        reward -= tc                                                 # transaction cost
        reward -= 0.5 * hedge_error                                 # hedge accuracy
        reward -= 0.01 * abs(self.hedge_position - old_hedge)       # smooth trading

        # ── 7. Termination ────────────────────────────────────────────────
        terminated = False
        truncated = False

        if self.current_step >= self.n_steps:
            intrinsic = max(self.price - self.K, 0.0)
            self.portfolio_value -= intrinsic
            terminated = True

        if self.portfolio_value < -3.0 * self.option_premium:
            terminated = True
            reward -= 10.0

        self.prev_portfolio_value = self.portfolio_value

        return self._get_obs(), float(reward), terminated, truncated, self._get_info(episode_done=terminated)

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        T_remaining = max(self.T - self.current_step * self.dt, 1e-10)
        T_fraction = T_remaining / self.T

        try:
            g = greeks_call(self.price, self.K, T_remaining,
                            self.r, self.current_vol)
            delta = g.delta
            gamma = g.gamma
        except Exception:
            delta = 0.5
            gamma = 0.01

        pnl_normalised = (
            (self.portfolio_value - self.option_premium)
            / max(self.option_premium, 1e-6)
        )

        return np.array([
            self.price / self.S0,
            self.K / self.S0,
            T_fraction,
            self.current_vol,
            delta,
            gamma,
            np.clip(pnl_normalised, -10, 10),
        ], dtype=np.float32)

    def _get_info(self, episode_done: bool = False) -> Dict[str, Any]:
        """
        Return episode info dict.

        episode_sharpe, total_pnl, and max_drawdown are only populated
        when episode_done=True (at termination). Emitting them at every
        step caused the DHP entropy controller to read mid-episode
        2-step Sharpe values (can be 100+) as real signal, corrupting
        the training dynamics.
        """
        T_remaining = max(self.T - self.current_step * self.dt, 1e-10)

        info = {
            "price": self.price,
            "volatility": self.current_vol,
            "hedge_position": self.hedge_position,
            "portfolio_value": self.portfolio_value,
            "option_premium": self.option_premium,
            "step": self.current_step,
            "T_remaining": T_remaining,
        }

        # Only compute episode-level metrics at true episode end.
        # Mid-episode values from 2-5 steps are statistically meaningless
        # and produce enormous spurious Sharpe ratios.
        if episode_done and len(self.pnl_history) > 1:
            pnls = np.array(self.pnl_history)
            pnl_std = float(np.std(pnls))

            if pnl_std > 1e-8:
                info["episode_sharpe"] = float(
                    np.mean(pnls) / pnl_std * np.sqrt(252)
                )
            else:
                info["episode_sharpe"] = 0.0

            info["total_pnl"] = float(np.sum(pnls))
            info["max_drawdown"] = float(self._compute_max_drawdown())

        return info

    def _compute_max_drawdown(self) -> float:
        """Compute maximum drawdown from PnL history."""
        if len(self.pnl_history) < 2:
            return 0.0
        cumulative = np.cumsum(self.pnl_history)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    def render(self):
        """Render environment state (human-readable)."""
        if self.render_mode != "human":
            return
        T_rem = max(self.T - self.current_step * self.dt, 0)
        print(f"Step {self.current_step:3d} | "
              f"S={self.price:8.2f} | "
              f"sigma={self.current_vol:.4f} | "
              f"Hedge={self.hedge_position:+.4f} | "
              f"PV={self.portfolio_value:8.4f} | "
              f"T_rem={T_rem:.4f}y")