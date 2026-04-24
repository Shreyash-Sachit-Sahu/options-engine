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

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pricer.pricer_py import bs_call, greeks_call
from environment.market_sim import GBMSimulator, HestonSimulator


class OptionsHedgingEnv(gym.Env):
    """
    Options Hedging Environment for Reinforcement Learning.

    The agent sells an at-the-money call option and must hedge it by
    trading the underlying asset. The goal is to minimize PnL variance
    (i.e., hedge as perfectly as possible) while managing transaction costs.

    Observation Space (7-dim):
        [0] S/S0           — normalized spot price
        [1] K/S0           — normalized strike
        [2] T_rem/T_total  — remaining time fraction
        [3] sigma          — current implied/realized volatility
        [4] delta           — Black-Scholes delta
        [5] gamma           — Black-Scholes gamma
        [6] portfolio_pnl   — normalized portfolio PnL

    Action Space:
        Box([-1], [1]) — target hedge ratio
          +1 = fully long underlying (fully hedged short call)
          -1 = fully short underlying (double short exposure)
           0 = no hedge

    Reward:
        -0.5 * portfolio_variance_this_step - transaction_cost
        Encourages stable, efficient hedging.

    Episode:
        30 trading days (option lifetime). Terminates early if
        portfolio value drops below -3 × initial premium.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        simulator_type: str = "gbm",
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 30 / 252,       # 30 trading days
        r: float = 0.05,
        sigma: float = 0.2,
        mu: float = 0.05,
        n_steps: int = 30,
        transaction_cost: float = 0.001,
        seed: Optional[int] = None,
        # Heston params
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Option parameters
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.tc_rate = transaction_cost

        # Create market simulator
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

        # Spaces — Gymnasium API
        # Observation: [S/S0, K/S0, T_rem/T, sigma, delta, gamma, pnl]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -np.inf], dtype=np.float32),
            high=np.array([5.0, 5.0, 1.0, 5.0, 1.0, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        # Action: hedge ratio in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Episode state (initialized in reset)
        self.current_step = 0
        self.price = S0
        self.current_vol = sigma
        self.hedge_position = 0.0
        self.portfolio_value = 0.0
        self.option_premium = 0.0
        self.pnl_history = []
        self.prev_portfolio_value = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Reset market simulator
        if seed is not None:
            self.price = self.simulator.reset(seed=seed)
        else:
            self.price = self.simulator.reset()

        self.current_step = 0
        self.current_vol = self.sigma
        self.hedge_position = 0.0
        self.pnl_history = []

        # Sell an ATM call option — collect premium
        self.option_premium = bs_call(self.price, self.K, self.T, self.r, self.sigma)
        self.portfolio_value = self.option_premium
        self.prev_portfolio_value = self.portfolio_value

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step.

        1. Apply hedge action (trade underlying)
        2. Advance market (simulate next price)
        3. Mark-to-market portfolio
        4. Compute reward
        """
        self.current_step += 1
        old_price = self.price
        target_hedge = float(np.clip(action[0], -1.0, 1.0))

        # ── 1. Transaction cost for changing position ──────────────────────
        position_change = abs(target_hedge - self.hedge_position)
        tc = self.tc_rate * position_change * old_price
        self.portfolio_value -= tc

        # ── 2. Update hedge position ──────────────────────────────────────
        old_hedge = self.hedge_position
        self.hedge_position = target_hedge

        # ── 3. Simulate next market state ─────────────────────────────────
        self.price, self.current_vol = self.simulator.step()

        # ── 4. Mark-to-market ─────────────────────────────────────────────
        price_change = self.price - old_price
        T_remaining = max(self.T - self.current_step * self.dt, 1e-10)

        # PnL from hedge: long hedge_position units of underlying
        hedge_pnl = self.hedge_position * price_change

        # PnL from option: we are SHORT the call
        old_option_val = bs_call(old_price, self.K,
                                 T_remaining + self.dt, self.r, self.current_vol)
        new_option_val = bs_call(self.price, self.K,
                                 T_remaining, self.r, self.current_vol)
        option_pnl = -(new_option_val - old_option_val)  # short position

        # Total portfolio change
        total_pnl = hedge_pnl + option_pnl
        self.portfolio_value += total_pnl
        self.pnl_history.append(total_pnl)

        # ── 5. Reward: minimize variance + transaction costs ──────────────
        # Use step PnL variance approximation
        pnl_variance = total_pnl ** 2  # instantaneous variance proxy
        reward = -0.5 * pnl_variance - tc

        # ── 6. Termination conditions ─────────────────────────────────────
        terminated = False
        truncated = False

        # Episode ends at option expiry
        if self.current_step >= self.n_steps:
            # Final settlement
            intrinsic = max(self.price - self.K, 0.0)
            self.portfolio_value -= intrinsic  # pay out option if ITM
            terminated = True

        # Early termination: catastrophic loss
        if self.portfolio_value < -3.0 * self.option_premium:
            terminated = True
            reward -= 10.0  # large penalty

        self.prev_portfolio_value = self.portfolio_value

        obs = self._get_obs()
        info = self._get_info()

        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        T_remaining = max(self.T - self.current_step * self.dt, 1e-10)
        T_fraction = T_remaining / self.T

        # Compute Greeks
        try:
            g = greeks_call(self.price, self.K, T_remaining,
                           self.r, self.current_vol)
            delta = g.delta
            gamma = g.gamma
        except Exception:
            delta = 0.5
            gamma = 0.01

        # Normalized PnL
        pnl_normalized = (self.portfolio_value - self.option_premium) / max(self.option_premium, 1e-6)

        obs = np.array([
            self.price / self.S0,       # Normalized spot
            self.K / self.S0,           # Normalized strike
            T_fraction,                  # Time remaining fraction
            self.current_vol,            # Current volatility
            delta,                       # BS delta
            gamma,                       # BS gamma
            np.clip(pnl_normalized, -10, 10)  # Normalized PnL (clipped)
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Return episode info dict."""
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

        # Add Sharpe ratio at episode end
        if len(self.pnl_history) > 1:
            returns = np.array(self.pnl_history)
            info["episode_sharpe"] = (
                float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252))
            )
            info["total_pnl"] = float(np.sum(returns))
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
              f"σ={self.current_vol:.4f} | "
              f"Hedge={self.hedge_position:+.4f} | "
              f"PV={self.portfolio_value:8.4f} | "
              f"T_rem={T_rem:.4f}y")
