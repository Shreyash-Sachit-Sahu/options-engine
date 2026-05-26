"""
Options Hedging Gymnasium Environment — v2

Upgrades over v1:
  - Execution delay    : action decided at t executes at t+1 (realistic)
  - Variable TC        : spread widens proportionally with realised vol
  - Vol regime obs     : 12-dim observation adds regime signal [0,0.5,1]
  - Clean PnL tracking : reward and PnL are decoupled — evaluation uses
                         raw PnL only, never the shaped reward

References:
  Buehler et al. (2019) "Deep Hedging"
  Cao et al. (2021) "Deep Hedging of Derivatives Using RL"
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys, os
from typing import Optional, Dict, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pricer.pricer_py import bs_call, greeks_call
from environment.market_sim import (
    GBMSimulator, HestonSimulator,
    MertonJumpDiffusionSimulator, VolRegimeSimulator,
)


# ── Regime thresholds ────────────────────────────────────────────────────────
_REGIME_LOW_THRESH  = 0.85   # current_vol / initial_vol < this → low
_REGIME_HIGH_THRESH = 1.25   # current_vol / initial_vol > this → high


class OptionsHedgingEnv(gym.Env):
    """
    Options Hedging Environment — v2.

    Observation Space (12-dim):
        [0]  S/S0              — normalised spot price
        [1]  K/S0              — normalised strike
        [2]  T_rem/T_total     — remaining time fraction
        [3]  sigma             — current realised volatility
        [4]  delta             — Black-Scholes delta
        [5]  gamma × 100       — scaled gamma
        [6]  portfolio_pnl     — normalised cumulative PnL
        [7]  vega              — vega
        [8]  theta             — theta (short call, clipped ≥ 0)
        [9]  vol_carry         — realised / implied vol ratio
        [10] current_hedge     — current (executed) hedge position
        [11] vol_regime        — 0.0=low / 0.5=medium / 1.0=high

    Action Space:
        Box([-1], [1]) — residual delta correction
          target_hedge = clip(delta + 0.3 * action, -1, 1)
          action = 0  → pure delta hedge

    Key changes from v1:
        execution_delay=1  : action at step t is queued and applied at t+1.
                             The agent must predict one step ahead.
        variable_tc=True   : effective TC = base_tc × (1 + 1.5 × max(vol_ratio-1, 0))
                             Spreads widen in high-vol environments.
    """

    metadata = {"render_modes": ["human"]}

    OBS_DIM = 12

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
        transaction_cost: float = 0.003,
        execution_delay: int = 1,
        variable_tc: bool = True,
        seed: Optional[int] = None,
        # Heston params
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.3,
        rho: float = -0.7,
        # Merton jump params
        lam: float = 1.0,
        mu_j: float = -0.10,
        sigma_j: float = 0.15,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma          # initial / reference vol
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.base_tc = transaction_cost
        self.execution_delay = max(0, int(execution_delay))
        self.variable_tc = variable_tc

        # ── Simulator ────────────────────────────────────────────────────
        self.simulator_type = simulator_type
        if simulator_type == "heston":
            self.simulator = HestonSimulator(
                S0=S0, mu=mu, v0=sigma**2,
                kappa=kappa, theta=theta, xi=xi, rho=rho,
                dt=self.dt, seed=seed,
            )
        elif simulator_type == "jump":
            self.simulator = MertonJumpDiffusionSimulator(
                S0=S0, mu=mu, sigma=sigma,
                lam=lam, mu_j=mu_j, sigma_j=sigma_j,
                dt=self.dt, seed=seed,
            )
        elif simulator_type == "regime":
            self.simulator = VolRegimeSimulator(
                S0=S0, mu=mu, dt=self.dt, seed=seed,
            )
        else:  # gbm
            self.simulator = GBMSimulator(
                S0=S0, mu=mu, sigma=sigma, dt=self.dt, seed=seed,
            )

        # ── Spaces ───────────────────────────────────────────────────────
        low  = np.array([0., 0., 0., 0., -1., 0., -np.inf,
                         0., -np.inf, 0., -1., 0.], dtype=np.float32)
        high = np.array([5., 5., 1., 5.,  1., np.inf, np.inf,
                         np.inf, np.inf, 5.,  1., 1.], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.], dtype=np.float32),
            high=np.array([ 1.], dtype=np.float32),
            dtype=np.float32,
        )

        # ── Episode state (initialised in reset) ─────────────────────────
        self.current_step    = 0
        self.price           = S0
        self.current_vol     = sigma
        self.hedge_position  = 0.0     # the *executed* position
        self.pending_action  = 0.0     # queued, not yet executed
        self.portfolio_value = 0.0
        self.option_premium  = 0.0
        self.pnl_history: list = []

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self.price = (self.simulator.reset(seed=seed)
                      if seed is not None else self.simulator.reset())

        self.current_step    = 0
        self.current_vol     = self.sigma
        self.hedge_position  = 0.0
        self.pending_action  = 0.0
        self.pnl_history     = []

        self.option_premium  = bs_call(self.price, self.K, self.T,
                                       self.r, self.sigma)
        self.portfolio_value = self.option_premium

        return self._get_obs(), self._get_info(episode_done=False)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        One trading step.

        Execution delay:
          action received now → becomes the *next* step's hedge.
          The hedge applied *this* step is the one queued from the
          previous step. This forces the agent to think ahead.

        Variable TC:
          effective_tc = base_tc × (1 + 1.5 × max(vol_ratio - 1, 0))
          where vol_ratio = current_vol / sigma_0
        """
        self.current_step += 1
        old_price = self.price
        T_remaining = max(self.T - self.current_step * self.dt, 1e-10)

        # ── 1. Determine this step's target hedge ─────────────────────────
        if self.execution_delay > 0:
            # Execute the *previous* action; queue the new one.
            execute_rl = float(self.pending_action)
            self.pending_action = float(action[0])
        else:
            execute_rl = float(action[0])

        # BS delta at OLD price (used to convert RL output → hedge fraction)
        try:
            g = greeks_call(old_price, self.K,
                            T_remaining + self.dt, self.r, self.current_vol)
            delta = g.delta
        except Exception:
            delta = 0.5

        target_hedge = float(np.clip(delta + 0.3 * execute_rl, -1.0, 1.0))

        # ── 2. Effective TC (widens with vol) ─────────────────────────────
        if self.variable_tc:
            vol_ratio = self.current_vol / max(self.sigma, 1e-8)
            tc_mult   = 1.0 + 1.5 * max(vol_ratio - 1.0, 0.0)
            eff_tc    = self.base_tc * tc_mult
        else:
            eff_tc = self.base_tc

        # ── 3. Transaction cost ───────────────────────────────────────────
        position_change  = abs(target_hedge - self.hedge_position)
        tc               = eff_tc * position_change * old_price
        self.portfolio_value -= tc
        self.hedge_position  = target_hedge

        # ── 4. Simulate next price ────────────────────────────────────────
        self.price, self.current_vol = self.simulator.step()

        # ── 5. Mark-to-market ─────────────────────────────────────────────
        price_change  = self.price - old_price
        hedge_pnl     = self.hedge_position * price_change

        old_option_val = bs_call(old_price, self.K,
                                 T_remaining + self.dt, self.r, self.current_vol)
        new_option_val = bs_call(self.price, self.K,
                                 T_remaining, self.r, self.current_vol)
        option_pnl = -(new_option_val - old_option_val)

        raw_pnl = hedge_pnl + option_pnl          # raw dollar PnL
        self.portfolio_value += raw_pnl
        self.pnl_history.append(raw_pnl)

        # ── 6. Shaped reward (training signal only) ───────────────────────
        # ── 6. Shaped reward (training signal only) ───────────────────────
        # Shaped reward — variance minimization (aligned with Sharpe)
        reward  = -0.5 * (raw_pnl ** 2)       # variance penalty
        reward -= tc                            # actual cost
        reward -= 0.005 * position_change      # turnover nudge
        reward += 0.3                          # constant offset → makes Q-values positive
                                       # fixes actor loss without changing objective
        # NOTE: turnover penalty kept very small (0.005) so it doesn't
        # distort baseline comparisons. Evaluation uses raw PnL only.

        # ── 7. Termination ────────────────────────────────────────────────
        terminated = truncated = False

        if self.current_step >= self.n_steps:
            intrinsic = max(self.price - self.K, 0.0)
            self.portfolio_value -= intrinsic
            terminated = True

        if self.portfolio_value < -3.0 * self.option_premium:
            reward -= 10.0
            terminated = True

        return (self._get_obs(), float(reward),
                terminated, truncated,
                self._get_info(episode_done=terminated))

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        T_remaining = max(self.T - self.current_step * self.dt, 1e-10)
        T_fraction  = T_remaining / self.T

        try:
            g     = greeks_call(self.price, self.K, T_remaining,
                                self.r, self.current_vol)
            delta = float(g.delta)
            gamma = float(g.gamma)
            vega  = float(g.vega)
            theta = float(g.theta)
        except Exception:
            delta, gamma, vega, theta = 0.5, 0.01, 0.0, 0.0

        pnl_norm  = ((self.portfolio_value - self.option_premium)
                     / max(self.option_premium, 1e-6))
        vol_carry = float(np.clip(
            self.current_vol / max(self.sigma, 1e-6), 0.0, 5.0))

        # ── Vol regime flag ───────────────────────────────────────────────
        # 0.0 = low vol  (carry < 0.85)
        # 0.5 = normal   (0.85 ≤ carry < 1.25)
        # 1.0 = high vol (carry ≥ 1.25)
        if vol_carry < _REGIME_LOW_THRESH:
            regime_flag = 0.0
        elif vol_carry < _REGIME_HIGH_THRESH:
            regime_flag = 0.5
        else:
            regime_flag = 1.0

        return np.array([
            self.price / self.S0,
            self.K     / self.S0,
            T_fraction,
            self.current_vol,
            delta,
            gamma * 100.0,
            float(np.clip(pnl_norm, -10, 10)),
            vega,
            float(np.clip(theta, 0.0, 10.0)),
            vol_carry,
            float(np.clip(self.hedge_position, -1.0, 1.0)),
            regime_flag,                               # ← new in v2
        ], dtype=np.float32)

    # ── Info ──────────────────────────────────────────────────────────────────

    def _get_info(self, episode_done: bool = False) -> Dict[str, Any]:
        T_remaining = max(self.T - self.current_step * self.dt, 1e-10)

        info = {
            "price":           self.price,
            "volatility":      self.current_vol,
            "hedge_position":  self.hedge_position,
            "portfolio_value": self.portfolio_value,
            "option_premium":  self.option_premium,
            "step":            self.current_step,
            "T_remaining":     T_remaining,
        }

        if episode_done and len(self.pnl_history) > 1:
            pnls    = np.array(self.pnl_history)
            pnl_std = float(np.std(pnls))
            info["episode_sharpe"] = (
                float(np.mean(pnls) / pnl_std * np.sqrt(252))
                if pnl_std > 1e-8 else 0.0
            )
            info["total_pnl"]    = float(np.sum(pnls))
            info["max_drawdown"] = float(self._compute_max_drawdown())

        return info

    def _compute_max_drawdown(self) -> float:
        if len(self.pnl_history) < 2:
            return 0.0
        cumulative  = np.cumsum(self.pnl_history)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns   = running_max - cumulative
        return float(np.max(drawdowns))

    def render(self):
        if self.render_mode != "human":
            return
        T_rem = max(self.T - self.current_step * self.dt, 0)
        print(f"Step {self.current_step:3d} | "
              f"S={self.price:8.2f} | sigma={self.current_vol:.4f} | "
              f"Hedge={self.hedge_position:+.4f} | "
              f"PV={self.portfolio_value:8.4f} | T_rem={T_rem:.4f}y")