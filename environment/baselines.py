"""
Baseline Hedging Agents — v2

Five baselines for rigorous comparison:

  DeltaHedger          — classical BS delta hedge (industry benchmark)
  StaticHedger         — buy-and-hold initial delta (TC lower bound)
  LelandHedger         — modified vol to account for TC (Leland 1985)
  WhalleyWilmottHedger — no-trade band around delta (Whalley & Wilmott 1997)
  RandomAgent          — random actions (sanity lower bound)

The two new baselines matter because they represent the *theoretically
optimal* solutions under their respective frameworks:

  Leland:         Adjust replication cost for discrete hedging with TC.
  Whalley-Wilmott: Asymptotically optimal CARA-utility hedging with TC.

If SAC does not outperform both, the RL adds nothing over classical theory.

Evaluation metric:
  Primary  — cross-episode Sharpe = mean(PnL) / std(PnL) * sqrt(252)
  Secondary — mean of per-episode Sharpes (noisier, shown for comparison)

Both metrics computed from *raw PnL* — shaped reward is never used here.

References:
  Leland, H.E. (1985) "Option Pricing and Replication with Transaction Costs"
      J. Finance 40(5):1283–1301.
  Whalley, A.E. & Wilmott, P. (1997) "An Asymptotic Analysis of an Optimal
      Hedging Model for Option Pricing with Transaction Costs"
      Math. Finance 7(3):307–324.
"""

import numpy as np
import sys, os
from typing import Optional
from scipy.stats import norm as scipy_norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pricer.pricer_py import greeks_call, bs_call


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# _ROOT_2_PI = sqrt(2 / pi)  used in the Leland formula
# _INV_SQRT_2PI for norm_pdf
# ─────────────────────────────────────────────────────────────────────────────
_ROOT_2_PI    = np.sqrt(2.0 / np.pi)   # ≈ 0.7979
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_cdf(x: float) -> float:
    return float(scipy_norm.cdf(x))


def _leland_sigma(sigma: float, tc_rate: float, dt: float) -> float:
    """
    Leland (1985) modified volatility for a short-call writer.

    The writer of a call rebalances at each dt, paying proportional
    transaction costs k = tc_rate per unit of underlying traded.
    Leland shows the correct replication cost uses:

        sigma_L = sigma * sqrt(1 + sqrt(2/pi) * k / (sigma * sqrt(dt)))

    where k = tc_rate (one-way) for the writer's perspective.

    Args:
        sigma    : Black-Scholes implied vol
        tc_rate  : one-way proportional TC rate (e.g. 0.003)
        dt       : hedging interval in years (e.g. 1/252)

    Returns:
        Modified vol sigma_L ≥ sigma
    """
    if sigma <= 0 or dt <= 0:
        return sigma
    adjustment = _ROOT_2_PI * tc_rate / (sigma * np.sqrt(dt))
    return sigma * np.sqrt(max(1.0 + adjustment, 0.0))


def _ww_halfband(gamma: float, S: float, tc_rate: float, dt: float,
                 risk_aversion: float = 1.0) -> float:
    """
    Whalley-Wilmott (1997) no-trade band half-width.

    For a CARA investor with risk aversion A facing proportional TC rate k,
    the asymptotically optimal strategy is to NOT rebalance until the
    hedge deviates from delta by more than H, where:

        H = [ 3 * k * |Gamma| * S^2 * sigma^2 * dt / (2 * A) ]^(1/3)

    Equivalently, in position (fraction-of-underlying) units:
        H_frac ≈ [ 3 * k * |Gamma_dollar| * dt / (2 * A) ]^(1/3) / S

    where Gamma_dollar = Gamma * S^2 is the dollar curvature.

    For a short-maturity ATM option (gamma ~ 0.04, S=100, tc=0.003, A=1):
        H ≈ 0.13–0.22  (13–22% position deviation triggers rebalance)

    Args:
        gamma        : BS gamma (∂²C/∂S²)
        S            : current spot
        tc_rate      : one-way proportional TC
        dt           : time step (years)
        risk_aversion: CARA coefficient A (default 1.0)

    Returns:
        H: half-bandwidth in hedge-fraction units
    """
    dollar_gamma = abs(gamma) * (S ** 2)
    numerator    = 3.0 * tc_rate * dollar_gamma * dt
    denominator  = max(2.0 * risk_aversion, 1e-8)
    return float(np.cbrt(numerator / denominator))


# ─────────────────────────────────────────────────────────────────────────────
# Baseline agents
# ─────────────────────────────────────────────────────────────────────────────

class DeltaHedger:
    """
    Classical delta-hedging baseline.
    Rebalances to BS delta at every step — the industry benchmark.
    """

    def __init__(self, K: float = 100.0, r: float = 0.05,
                 sigma: float = 0.2, T: float = 30/252, n_steps: int = 30):
        self.K = K; self.r = r; self.sigma = sigma
        self.T = T; self.n_steps = n_steps
        self.dt = T / n_steps
        self.step_count = 0

    def reset(self):
        self.step_count = 0

    def predict(self, obs: np.ndarray, **kwargs):
        delta = float(obs[4]) if obs.ndim == 1 else float(obs[0, 4])
        self.step_count += 1
        return np.array([np.clip(delta, -1.0, 1.0)], dtype=np.float32), None


class StaticHedger:
    """
    Buy-and-hold initial delta hedge.
    Locks the inception delta and never rebalances.
    Represents the extreme low-TC / high-error trade-off.
    """

    def __init__(self, initial_delta: Optional[float] = None):
        self.initial_delta = initial_delta
        self._initialized  = False

    def reset(self):
        self._initialized = False

    def predict(self, obs: np.ndarray, **kwargs):
        if not self._initialized:
            delta              = float(obs[4]) if obs.ndim == 1 else float(obs[0, 4])
            self.initial_delta = float(np.clip(delta, -1.0, 1.0))
            self._initialized  = True
        return np.array([self.initial_delta], dtype=np.float32), None


class LelandHedger:
    """
    Leland (1985) transaction-cost-adjusted delta hedge.

    Modifies the implied volatility used to compute delta, increasing it
    to compensate for the cost of discrete rebalancing. For an option
    WRITER, the effective hedging vol is strictly greater than the
    Black-Scholes vol, producing a larger delta and a tighter hedge.

    This is the theoretically correct replication strategy when TC is
    the only departure from the BSM assumptions (constant vol, no jumps).
    It should beat pure delta hedging in any environment with non-zero TC.

    Usage:
        agent = LelandHedger(sigma=0.2, tc_rate=0.003, dt=1/252)
    """

    def __init__(self, sigma: float = 0.2, tc_rate: float = 0.003,
                 K: float = 100.0, r: float = 0.05,
                 T: float = 30/252, n_steps: int = 30):
        self.sigma    = sigma
        self.tc_rate  = tc_rate
        self.K        = K
        self.r        = r
        self.T        = T
        self.n_steps  = n_steps
        self.dt       = T / n_steps
        self.step_count = 0
        # Pre-compute the modified volatility (constant for fixed tc and dt)
        self.sigma_L  = _leland_sigma(sigma, tc_rate, self.dt)

    def reset(self):
        self.step_count = 0

    def predict(self, obs: np.ndarray, **kwargs):
        """
        Compute Leland delta using modified vol sigma_L.

        Reads S, remaining T from the observation vector:
          obs[0] = S/S0, obs[2] = T_rem/T_total
        Uses S0=100 internally (observation is normalised by S0).
        """
        self.step_count += 1
        if obs.ndim > 1:
            obs = obs[0]

        # Decode observation
        S_norm   = float(obs[0])   # S / S0
        T_frac   = float(obs[2])   # T_remaining / T_total
        S        = S_norm * 100.0  # assume S0=100 (normalised env)
        T_rem    = max(T_frac * self.T, 1e-10)

        try:
            # Use Leland-modified vol to compute delta
            g     = greeks_call(S, self.K, T_rem, self.r, self.sigma_L)
            delta = float(np.clip(g.delta, -1.0, 1.0))
        except Exception:
            # Fallback to standard delta from obs
            delta = float(np.clip(obs[4], -1.0, 1.0))

        return np.array([delta], dtype=np.float32), None


class WhalleyWilmottHedger:
    """
    Whalley-Wilmott (1997) asymptotically-optimal CARA hedger.

    Maintains a NO-TRADE BAND of half-width H around the BS delta.
    Only rebalances when the current position deviates beyond the band.
    Within the band, transaction costs outweigh the hedging benefit.

    The band half-width H scales with:
        H ∝ (TC rate × |Gamma| × S² × dt)^(1/3)

    This is optimal for a CARA investor in the limit of small TC and
    continuous time. It provably outperforms both delta hedging (which
    over-trades) and static hedging (which under-trades) for any
    non-degenerate TC regime.

    Implementation note:
        We track the *executed* hedge position internally to correctly
        compute deviations and apply the band logic. This is reset at
        the start of each episode.

    Usage:
        agent = WhalleyWilmottHedger(sigma=0.2, tc_rate=0.003,
                                     dt=1/252, risk_aversion=1.0)
    """

    def __init__(self, sigma: float = 0.2, tc_rate: float = 0.003,
                 K: float = 100.0, r: float = 0.05,
                 T: float = 30/252, n_steps: int = 30,
                 risk_aversion: float = 1.0):
        self.sigma         = sigma
        self.tc_rate       = tc_rate
        self.K             = K
        self.r             = r
        self.T             = T
        self.n_steps       = n_steps
        self.dt            = T / n_steps
        self.risk_aversion = risk_aversion
        self._hedge        = 0.0    # current executed position

    def reset(self):
        self._hedge = 0.0

    def predict(self, obs: np.ndarray, **kwargs):
        """
        Apply WW no-trade band.

        If current hedge is within [delta - H, delta + H]: hold.
        Otherwise: rebalance to the near edge of the band.
        """
        if obs.ndim > 1:
            obs = obs[0]

        S_norm = float(obs[0])
        T_frac = float(obs[2])
        S      = S_norm * 100.0
        T_rem  = max(T_frac * self.T, 1e-10)

        # Use current_vol from obs (index 3) for gamma computation
        sigma_now = float(obs[3]) if float(obs[3]) > 0.01 else self.sigma

        # Read delta and gamma directly from the observation vector.
        # obs[4] = BS delta, obs[5] = gamma×100 (consistent with all agents).
        # This avoids recomputing greeks from S/K/T which can produce the
        # same values when normalised observations are identical.
        delta = float(np.clip(obs[4], -1.0, 1.0))
        gamma = float(obs[5]) / 100.0

        # Band half-width
        H = _ww_halfband(gamma, S, self.tc_rate, self.dt, self.risk_aversion)

        # Band boundaries
        lower = delta - H
        upper = delta + H

        if self._hedge < lower:
            # Under-hedged — rebalance up to lower boundary
            new_hedge = lower
        elif self._hedge > upper:
            # Over-hedged — rebalance down to upper boundary
            new_hedge = upper
        else:
            # Within band — hold
            new_hedge = self._hedge

        new_hedge    = float(np.clip(new_hedge, -1.0, 1.0))
        self._hedge  = new_hedge
        return np.array([new_hedge], dtype=np.float32), None


class RandomAgent:
    """
    Random hedging (sanity lower bound).
    Any reasonable strategy should beat this.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, **kwargs):
        return self.rng.uniform(-1., 1., size=(1,)).astype(np.float32), None


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner and batch evaluator
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_episode(env, agent, seed: Optional[int] = None) -> dict:
    """Run one episode; return raw metrics (no shaped reward)."""
    obs, _ = env.reset(seed=seed)
    if hasattr(agent, 'reset'):
        agent.reset()

    done     = False
    ep_pnl   = 0.0
    hedges   = []

    while not done:
        action, _ = agent.predict(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        hedges.append(float(action[0]))

    return {
        "total_pnl":      info.get("total_pnl", ep_pnl),
        "portfolio_value": info.get("portfolio_value", 0.0),
        "episode_sharpe": info.get("episode_sharpe", 0.0),
        "max_drawdown":   info.get("max_drawdown", 0.0),
        "mean_hedge":     float(np.mean(hedges)),
        "hedge_std":      float(np.std(hedges)),
        "n_trades":       len(hedges),
    }


def evaluate_agent(env, agent, n_episodes: int = 500,
                   seed_start: int = 0) -> dict:
    """
    Evaluate an agent over n_episodes.

    PRIMARY metric: cross-episode Sharpe
        = mean(episode_PnLs) / std(episode_PnLs) * sqrt(252)

    This is computed from raw PnL — the shaped reward is never used.

    SECONDARY metric: mean of per-episode Sharpes
        Kept for comparison and plotting, but NOT used for ranking.
        Per-episode Sharpes over 30-step windows are very noisy.

    Statistical note:
        The cross-episode Sharpe is the only metric that correctly
        measures whether the agent generates consistent PnL relative
        to its variance across independent market scenarios.
    """
    results = []
    for i in range(n_episodes):
        r = run_baseline_episode(env, agent, seed=seed_start + i)
        results.append(r)

    pnls      = [r["total_pnl"]      for r in results]
    sharpes   = [r["episode_sharpe"] for r in results]
    drawdowns = [r["max_drawdown"]   for r in results]

    pnl_arr = np.array(pnls)
    pnl_std = float(np.std(pnl_arr))

    # ── Cross-episode Sharpe (PRIMARY) ────────────────────────────────────
    cross_ep_sharpe = float(
        np.mean(pnl_arr) / max(pnl_std, 1e-6) * np.sqrt(252)
    )

    return {
        "agent":                 agent.__class__.__name__,
        "n_episodes":            n_episodes,
        # ── Primary metric ────────────────────────────────────────────────
        "cross_episode_sharpe":  cross_ep_sharpe,
        # ── Secondary / diagnostic ────────────────────────────────────────
        "mean_episode_sharpe":   float(np.mean(sharpes)),
        "mean_pnl":              float(np.mean(pnl_arr)),
        "std_pnl":               pnl_std,
        "mean_max_drawdown":     float(np.mean(drawdowns)),
        "max_max_drawdown":      float(np.max(drawdowns)) if drawdowns else 0.,
        "episode_pnls":          pnls,
        "episode_sharpes":       sharpes,
    }