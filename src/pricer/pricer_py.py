"""
Pure-Python pricing implementation — API-compatible with the C++ module.

This serves as both a fallback when the C++ module hasn't been compiled
and a reference implementation for testing. Uses NumPy/SciPy for
vectorized operations where possible.
"""

import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass


# ─── Result Structs ───────────────────────────────────────────────────────────

@dataclass
class Greeks:
    """Container for option Greeks."""
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0

    def __repr__(self):
        return (f"Greeks(delta={self.delta:.6f}, gamma={self.gamma:.6f}, "
                f"vega={self.vega:.6f}, theta={self.theta:.6f}, rho={self.rho:.6f})")


@dataclass
class MCResult:
    """Monte Carlo pricing result."""
    price: float = 0.0
    std_error: float = 0.0
    n_paths: int = 0

    def __repr__(self):
        return (f"MCResult(price={self.price:.6f}, std_error={self.std_error:.6f}, "
                f"n_paths={self.n_paths})")


# ─── Normal Distribution ─────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return float(norm.cdf(x))


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return float(norm.pdf(x))


# ─── Black-Scholes Pricing ───────────────────────────────────────────────────

def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        return max(S - K * math.exp(-r * T), 0.0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    return float(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2))


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0.0:
        return max(K - S, 0.0)
    if sigma <= 0.0:
        return max(K * math.exp(-r * T) - S, 0.0)

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    return float(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ─── Greeks ───────────────────────────────────────────────────────────────────

def greeks_call(S: float, K: float, T: float, r: float, sigma: float) -> Greeks:
    """Compute all Greeks for a European call."""
    if T <= 0.0 or sigma <= 0.0:
        return Greeks(delta=1.0 if S > K else 0.0)

    sqrtT = math.sqrt(T)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    nd1 = float(norm.pdf(d1))
    Nd1 = float(norm.cdf(d1))
    Nd2 = float(norm.cdf(d2))
    Ke_rT = K * math.exp(-r * T)

    return Greeks(
        delta=Nd1,
        gamma=nd1 / (S * sigma * sqrtT),
        vega=S * nd1 * sqrtT,
        theta=-(S * nd1 * sigma) / (2.0 * sqrtT) - r * Ke_rT * Nd2,
        rho=Ke_rT * T * Nd2
    )


def greeks_put(S: float, K: float, T: float, r: float, sigma: float) -> Greeks:
    """Compute all Greeks for a European put."""
    if T <= 0.0 or sigma <= 0.0:
        return Greeks(delta=-1.0 if S < K else 0.0)

    sqrtT = math.sqrt(T)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(d1, sigma, T)
    nd1 = float(norm.pdf(d1))
    Nd1 = float(norm.cdf(d1))
    Nd2 = float(norm.cdf(d2))
    Ke_rT = K * math.exp(-r * T)

    return Greeks(
        delta=Nd1 - 1.0,
        gamma=nd1 / (S * sigma * sqrtT),
        vega=S * nd1 * sqrtT,
        theta=-(S * nd1 * sigma) / (2.0 * sqrtT) + r * Ke_rT * (1.0 - Nd2),
        rho=-Ke_rT * T * (1.0 - Nd2)
    )


def greeks(S: float, K: float, T: float, r: float, sigma: float,
           flag: str = "call") -> Greeks:
    """Compute all Greeks for a European option."""
    if flag in ("call", "c"):
        return greeks_call(S, K, T, r, sigma)
    elif flag in ("put", "p"):
        return greeks_put(S, K, T, r, sigma)
    else:
        raise ValueError(f"flag must be 'call' or 'put', got: {flag}")


# ─── Implied Volatility ──────────────────────────────────────────────────────

def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                flag: str = "call", tol: float = 1e-12,
                max_iter: int = 200) -> float:
    """Compute implied volatility using Brent's method (scipy)."""
    price_fn = bs_call if flag in ("call", "c") else bs_put

    def objective(sigma):
        return price_fn(S, K, T, r, sigma) - market_price

    try:
        return float(brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter))
    except ValueError:
        raise RuntimeError("Cannot bracket implied vol: market price out of range")


# ─── Monte Carlo ──────────────────────────────────────────────────────────────

def mc_price(S: float, K: float, T: float, r: float, sigma: float,
             n_paths: int = 50000, seed: int = 42,
             flag: str = "call") -> MCResult:
    """Monte Carlo European option price with antithetic variates."""
    if T <= 0.0:
        if flag in ("call", "c"):
            payoff = max(S - K, 0.0)
        else:
            payoff = max(K - S, 0.0)
        return MCResult(payoff, 0.0, 0)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    drift = (r - 0.5 * sigma ** 2) * T
    vol_sqrt_T = sigma * math.sqrt(T)

    ST_plus = S * np.exp(drift + vol_sqrt_T * Z)
    ST_minus = S * np.exp(drift - vol_sqrt_T * Z)

    if flag in ("call", "c"):
        payoff_plus = np.maximum(ST_plus - K, 0.0)
        payoff_minus = np.maximum(ST_minus - K, 0.0)
    else:
        payoff_plus = np.maximum(K - ST_plus, 0.0)
        payoff_minus = np.maximum(K - ST_minus, 0.0)

    payoff_avg = 0.5 * (payoff_plus + payoff_minus)
    discount = math.exp(-r * T)

    mean_payoff = float(np.mean(payoff_avg))
    std_payoff = float(np.std(payoff_avg))
    std_error = std_payoff / math.sqrt(n_paths)

    return MCResult(discount * mean_payoff, discount * std_error, n_paths)


def mc_price_multistep(S: float, K: float, T: float, r: float, sigma: float,
                       n_paths: int = 50000, n_steps: int = 252,
                       seed: int = 42, flag: str = "call") -> MCResult:
    """Monte Carlo with multi-step simulation for path-dependent options."""
    if T <= 0.0:
        if flag in ("call", "c"):
            payoff = max(S - K, 0.0)
        else:
            payoff = max(K - S, 0.0)
        return MCResult(payoff, 0.0, 0)

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol_sqrt_dt = sigma * math.sqrt(dt)

    Z = rng.standard_normal((n_paths, n_steps))

    # Log-price simulation for numerical stability
    log_S_plus = math.log(S) + np.cumsum(drift + vol_sqrt_dt * Z, axis=1)
    log_S_minus = math.log(S) + np.cumsum(drift - vol_sqrt_dt * Z, axis=1)

    ST_plus = np.exp(log_S_plus[:, -1])
    ST_minus = np.exp(log_S_minus[:, -1])

    if flag in ("call", "c"):
        payoff_plus = np.maximum(ST_plus - K, 0.0)
        payoff_minus = np.maximum(ST_minus - K, 0.0)
    else:
        payoff_plus = np.maximum(K - ST_plus, 0.0)
        payoff_minus = np.maximum(K - ST_minus, 0.0)

    payoff_avg = 0.5 * (payoff_plus + payoff_minus)
    discount = math.exp(-r * T)

    mean_payoff = float(np.mean(payoff_avg))
    std_payoff = float(np.std(payoff_avg))
    std_error = std_payoff / math.sqrt(n_paths)

    return MCResult(discount * mean_payoff, discount * std_error, n_paths)
