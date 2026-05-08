"""
Market Simulators: GBM + Heston Stochastic Volatility

GBMSimulator: Geometric Brownian Motion for standard market simulation.
HestonSimulator: Stochastic volatility model with mean-reverting variance.

Both produce daily price paths for the options hedging environment.

References:
- Heston, S.L. (1993) "A Closed-Form Solution for Options with
  Stochastic Volatility"
- Full truncation scheme: Lord, Koekkoek, van Dijk (2010)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MarketState:
    """Snapshot of market state at a single time step."""
    price: float
    volatility: float
    realized_vol: float
    step: int
    time: float  # in years


class GBMSimulator:
    """
    Geometric Brownian Motion market simulator.

    dS = mu * S * dt + sigma * S * dW

    Simulates log-normal price dynamics with constant drift and volatility.
    Uses exact simulation (not Euler) for accuracy.
    """

    def __init__(self, S0: float = 100.0, mu: float = 0.05,
                 sigma: float = 0.2, dt: float = 1/252,
                 seed: Optional[int] = 42):
        """
        Args:
            S0: Initial spot price
            mu: Annualized drift (expected return)
            sigma: Annualized volatility
            dt: Time step size (default: 1 trading day = 1/252)
            seed: Random seed for reproducibility
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # State
        self.price = S0
        self.step_count = 0
        self._prices = [S0]

    def reset(self, seed: Optional[int] = None) -> float:
        """Reset simulator to initial state."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.price = self.S0
        self.step_count = 0
        self._prices = [self.S0]
        return self.price

    def step(self) -> Tuple[float, float]:
        """
        Advance one time step.

        Returns:
            (new_price, realized_volatility)
        """
        Z = self.rng.standard_normal()

        # Exact GBM simulation (log-normal)
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        self.price *= np.exp(drift + diffusion)

        self.step_count += 1
        self._prices.append(self.price)

        # Compute realized volatility from return history
        realized_vol = self._compute_realized_vol()

        return self.price, realized_vol

    def _compute_realized_vol(self, window: int = 20) -> float:
        """Compute annualized realized volatility from recent returns."""
        if len(self._prices) < 3:
            return self.sigma

        prices = np.array(self._prices[-min(window + 1, len(self._prices)):])
        log_returns = np.diff(np.log(prices))

        if len(log_returns) < 2:
            return self.sigma

        return float(np.std(log_returns) * np.sqrt(252))

    def generate_path(self, n_steps: int) -> np.ndarray:
        """Generate a complete price path of n_steps."""
        self.reset()
        prices = np.zeros(n_steps + 1)
        prices[0] = self.S0

        Z = self.rng.standard_normal(n_steps)
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        vol_sqrt_dt = self.sigma * np.sqrt(self.dt)

        for i in range(n_steps):
            prices[i + 1] = prices[i] * np.exp(drift + vol_sqrt_dt * Z[i])

        return prices

    @property
    def time(self) -> float:
        """Current time in years."""
        return self.step_count * self.dt


class HestonSimulator:
    """
    Heston Stochastic Volatility model simulator.

    dS = mu * S * dt + sqrt(v) * S * dW1
    dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW2
    corr(dW1, dW2) = rho

    Uses Full Truncation scheme for variance positivity (Lord et al. 2010).
    Simulates log(S) for numerical stability.

    Default parameters calibrated to typical US equity dynamics:
      kappa=2.0 (mean reversion speed)
      theta=0.04 (long-run variance = 20% vol)
      xi=0.3 (vol-of-vol)
      rho=-0.7 (leverage effect: vol rises when price falls)
    """

    def __init__(self, S0: float = 100.0, mu: float = 0.05,
                 v0: float = 0.04, kappa: float = 2.0,
                 theta: float = 0.04, xi: float = 0.3,
                 rho: float = -0.7, dt: float = 1/252,
                 seed: Optional[int] = 42):
        """
        Args:
            S0: Initial spot price
            mu: Annualized drift
            v0: Initial variance (sigma^2, not sigma)
            kappa: Mean reversion speed (higher = faster revert)
            theta: Long-run variance level
            xi: Volatility of volatility
            rho: Correlation between price and vol processes
            dt: Time step (1/252 = daily)
            seed: Random seed
        """
        self.S0 = S0
        self.mu = mu
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Feller condition: 2*kappa*theta > xi^2
        # If violated, variance can hit zero
        self.feller_satisfied = (2 * kappa * theta > xi ** 2)

        # State
        self.price = S0
        self.variance = v0
        self.step_count = 0
        self._prices = [S0]
        self._variances = [v0]

    def reset(self, seed: Optional[int] = None) -> float:
        """Reset simulator to initial conditions."""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.price = self.S0
        self.variance = self.v0
        self.step_count = 0
        self._prices = [self.S0]
        self._variances = [self.v0]
        return self.price

    def step(self) -> Tuple[float, float]:
        """
        Advance one time step under Heston dynamics.

        Uses Full Truncation scheme:
          - Replace v with max(v, 0) inside the sqrt
          - But keep v itself unrestricted in the drift

        Returns:
            (new_price, current_volatility = sqrt(max(v, 0)))
        """
        # Generate correlated Brownian increments
        Z1 = self.rng.standard_normal()
        Z2 = self.rng.standard_normal()
        W1 = Z1
        W2 = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2

        sqrt_dt = np.sqrt(self.dt)
        v_plus = max(self.variance, 0.0)  # Full truncation
        sqrt_v = np.sqrt(v_plus)

        # Variance process (CIR / Cox-Ingersoll-Ross) — Milstein scheme
        self.variance = (self.variance
                         + self.kappa * (self.theta - v_plus) * self.dt
                         + self.xi * sqrt_v * sqrt_dt * W2
                         + 0.25 * self.xi ** 2 * self.dt * (W2 ** 2 - 1))

        # Price process (log transform for stability)
        v_plus_new = max(self.variance, 0.0)
        log_return = ((self.mu - 0.5 * v_plus) * self.dt
                      + sqrt_v * sqrt_dt * W1)
        self.price *= np.exp(log_return)

        self.step_count += 1
        self._prices.append(self.price)
        self._variances.append(max(self.variance, 0.0))

        current_vol = np.sqrt(max(self.variance, 0.0))
        return self.price, float(current_vol)

    def generate_path(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete price and volatility paths.

        Returns:
            (prices array, volatilities array)
        """
        self.reset()
        prices = np.zeros(n_steps + 1)
        vols = np.zeros(n_steps + 1)
        prices[0] = self.S0
        vols[0] = np.sqrt(self.v0)

        for i in range(n_steps):
            price, vol = self.step()
            prices[i + 1] = price
            vols[i + 1] = vol

        return prices, vols

    @property
    def volatility(self) -> float:
        """Current instantaneous volatility."""
        return float(np.sqrt(max(self.variance, 0.0)))

    @property
    def time(self) -> float:
        """Current time in years."""
        return self.step_count * self.dt




class MertonJumpDiffusionSimulator:
    """
    Merton (1976) Jump-Diffusion model.

    dS/S = (mu - lambda*k_bar) dt + sigma dW + J dN

    where:
        dN  ~ Poisson(lambda * dt)   — jump arrival
        J   ~ LogNormal(mu_j, sigma_j) — jump size
        k_bar = exp(mu_j + 0.5*sigma_j^2) - 1  — mean jump size

    This is the most important model for stress-testing the RL agent:
    delta hedging fails badly at jump events because the hedge cannot
    be rebalanced instantaneously. A robust agent learns to carry a
    larger inventory buffer before expiry to absorb gap risk.

    Typical US equity parameters:
        lambda=1.0  (1 jump per year on average)
        mu_j=-0.10  (average -10% jump — left tail)
        sigma_j=0.15 (jump size uncertainty)
    """

    def __init__(self, S0: float = 100.0, mu: float = 0.05,
                 sigma: float = 0.15, lam: float = 1.0,
                 mu_j: float = -0.10, sigma_j: float = 0.15,
                 dt: float = 1/252, seed: Optional[int] = 42):
        """
        Args:
            S0:      Initial spot price
            mu:      Continuous drift
            sigma:   Diffusive (non-jump) volatility
            lam:     Jump intensity (jumps per year)
            mu_j:    Mean log jump size (negative = left-skewed)
            sigma_j: Std of log jump size
            dt:      Time step (1/252 = daily)
            seed:    Random seed
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Mean jump size for drift correction
        self.k_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1

        # State
        self.price = S0
        self.step_count = 0
        self._prices = [S0]
        self._had_jump = [False]

    def reset(self, seed: Optional[int] = None) -> float:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.price = self.S0
        self.step_count = 0
        self._prices = [self.S0]
        self._had_jump = [False]
        return self.price

    def step(self) -> Tuple[float, float]:
        """
        Advance one time step under Merton jump-diffusion.

        Returns:
            (new_price, realized_vol_including_jump_variance)
        """
        # Diffusive component
        Z = self.rng.standard_normal()
        drift = (self.mu - self.lam * self.k_bar - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * Z
        log_return = drift + diffusion

        # Jump component — Poisson number of jumps this step
        n_jumps = self.rng.poisson(self.lam * self.dt)
        had_jump = n_jumps > 0

        if n_jumps > 0:
            # Each jump is log-normal
            jump_sizes = self.rng.normal(self.mu_j, self.sigma_j, n_jumps)
            log_return += np.sum(jump_sizes)

        self.price *= np.exp(log_return)
        self.step_count += 1
        self._prices.append(self.price)
        self._had_jump.append(had_jump)

        # Realized vol includes both diffusive and jump variance
        realized_vol = self._compute_realized_vol()
        return self.price, realized_vol

    def _compute_realized_vol(self, window: int = 20) -> float:
        if len(self._prices) < 3:
            return self.sigma
        prices = np.array(self._prices[-min(window + 1, len(self._prices)):])
        log_returns = np.diff(np.log(prices))
        if len(log_returns) < 2:
            return self.sigma
        return float(np.std(log_returns) * np.sqrt(252))

    def generate_path(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete price path with jump indicators.

        Returns:
            (prices, realized_vols, jump_flags)
        """
        self.reset()
        prices = np.zeros(n_steps + 1)
        vols = np.zeros(n_steps + 1)
        jumps = np.zeros(n_steps + 1, dtype=bool)
        prices[0] = self.S0
        vols[0] = self.sigma

        for i in range(n_steps):
            price, vol = self.step()
            prices[i + 1] = price
            vols[i + 1] = vol
            jumps[i + 1] = self._had_jump[-1]

        return prices, vols, jumps

    @property
    def time(self) -> float:
        return self.step_count * self.dt

class VolRegimeSimulator:
    """
    Market simulator with regime-switching volatility.

    Alternates between low-vol and high-vol regimes using
    a Markov chain transition matrix. Useful for stress-testing
    the RL agent's adaptability.
    """

    def __init__(self, S0: float = 100.0, mu: float = 0.05,
                 sigma_low: float = 0.15, sigma_high: float = 0.40,
                 p_low_to_high: float = 0.02, p_high_to_low: float = 0.05,
                 dt: float = 1/252, seed: Optional[int] = 42):
        self.S0 = S0
        self.mu = mu
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.p_low_to_high = p_low_to_high
        self.p_high_to_low = p_high_to_low
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # State
        self.price = S0
        self.regime = 0  # 0 = low vol, 1 = high vol
        self.step_count = 0
        self._prices = [S0]

    def reset(self, seed: Optional[int] = None) -> float:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.price = self.S0
        self.regime = 0
        self.step_count = 0
        self._prices = [self.S0]
        return self.price

    def step(self) -> Tuple[float, float]:
        """Step with regime switching."""
        # Regime transition
        u = self.rng.uniform()
        if self.regime == 0 and u < self.p_low_to_high:
            self.regime = 1
        elif self.regime == 1 and u < self.p_high_to_low:
            self.regime = 0

        sigma = self.sigma_high if self.regime == 1 else self.sigma_low

        Z = self.rng.standard_normal()
        drift = (self.mu - 0.5 * sigma ** 2) * self.dt
        self.price *= np.exp(drift + sigma * np.sqrt(self.dt) * Z)

        self.step_count += 1
        self._prices.append(self.price)

        return self.price, sigma

    @property
    def time(self) -> float:
        return self.step_count * self.dt