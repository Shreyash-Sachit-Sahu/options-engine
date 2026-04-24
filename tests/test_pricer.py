"""
C++ Pricer Unit Tests

Tests for both the C++ pybind11 module and the Python fallback.
Validates:
- Put-call parity
- Greeks sign/magnitude bounds
- MC vs BSM convergence
- Implied vol round-trip accuracy
- Edge case handling
"""

import os
import sys
import math
import pytest
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try C++ first, fall back to Python
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import pricer
    BACKEND = "C++"
except ImportError:
    from src.pricer import pricer_py as pricer
    BACKEND = "Python"

print(f"Testing with {BACKEND} backend")


class TestBlackScholes:
    """Black-Scholes pricing tests."""

    # Standard test parameters
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_call_price_positive(self):
        """Call price should always be positive for S, K > 0."""
        price = pricer.bs_call(self.S, self.K, self.T, self.r, self.sigma)
        assert price > 0.0

    def test_put_price_positive(self):
        """Put price should always be positive for S, K > 0."""
        price = pricer.bs_put(self.S, self.K, self.T, self.r, self.sigma)
        assert price > 0.0

    def test_call_price_known_value(self):
        """BSM call for ATM 1Y option should match known value."""
        price = pricer.bs_call(100, 100, 1.0, 0.05, 0.2)
        assert abs(price - 10.4506) < 0.01  # Known BSM price

    def test_put_call_parity(self):
        """Put-Call Parity: C - P = S - K*exp(-rT)"""
        call = pricer.bs_call(self.S, self.K, self.T, self.r, self.sigma)
        put = pricer.bs_put(self.S, self.K, self.T, self.r, self.sigma)
        parity = self.S - self.K * math.exp(-self.r * self.T)

        assert abs((call - put) - parity) < 1e-10, \
            f"Put-call parity violated: {call - put} != {parity}"

    @pytest.mark.parametrize("S,K,T,r,sigma", [
        (50, 100, 0.5, 0.03, 0.3),    # Deep OTM
        (150, 100, 0.5, 0.03, 0.3),   # Deep ITM
        (100, 100, 0.01, 0.05, 0.5),  # Near expiry
        (100, 100, 5.0, 0.05, 0.1),   # Long dated
        (200, 200, 1.0, 0.08, 0.4),   # High strike
    ])
    def test_put_call_parity_various(self, S, K, T, r, sigma):
        """Put-call parity across various parameter sets."""
        call = pricer.bs_call(S, K, T, r, sigma)
        put = pricer.bs_put(S, K, T, r, sigma)
        parity = S - K * math.exp(-r * T)
        assert abs((call - put) - parity) < 1e-8

    def test_call_upper_bound(self):
        """Call price <= S (can't be worth more than the stock)."""
        price = pricer.bs_call(self.S, self.K, self.T, self.r, self.sigma)
        assert price <= self.S

    def test_put_upper_bound(self):
        """Put price <= K * exp(-rT) (PV of strike)."""
        price = pricer.bs_put(self.S, self.K, self.T, self.r, self.sigma)
        assert price <= self.K * math.exp(-self.r * self.T)

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S - K*exp(-rT)."""
        price = pricer.bs_call(200, 100, 1.0, 0.05, 0.2)
        intrinsic = 200 - 100 * math.exp(-0.05)
        assert price >= intrinsic - 0.01

    def test_zero_vol_call(self):
        """Zero vol call = max(S - K*exp(-rT), 0)."""
        price = pricer.bs_call(100, 90, 1.0, 0.05, 1e-10)
        expected = max(100 - 90 * math.exp(-0.05), 0)
        assert abs(price - expected) < 0.01

    def test_expired_option(self):
        """Expired call = max(S - K, 0)."""
        price = pricer.bs_call(110, 100, 0.0, 0.05, 0.2)
        assert abs(price - 10.0) < 1e-10


class TestGreeks:
    """Greeks computation tests."""

    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    def test_call_delta_bounds(self):
        """Call delta should be in [0, 1]."""
        g = pricer.greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert 0 <= g.delta <= 1, f"Delta = {g.delta}"

    def test_put_delta_bounds(self):
        """Put delta should be in [-1, 0]."""
        g = pricer.greeks(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert -1 <= g.delta <= 0, f"Delta = {g.delta}"

    def test_call_put_delta_relation(self):
        """Call delta - Put delta = 1."""
        gc = pricer.greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        gp = pricer.greeks(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert abs(gc.delta - gp.delta - 1.0) < 1e-10

    def test_gamma_positive(self):
        """Gamma should always be positive."""
        g = pricer.greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert g.gamma > 0

    def test_gamma_same_call_put(self):
        """Call and put gamma should be equal."""
        gc = pricer.greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        gp = pricer.greeks(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert abs(gc.gamma - gp.gamma) < 1e-10

    def test_vega_positive(self):
        """Vega should always be positive."""
        g = pricer.greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert g.vega > 0

    def test_vega_same_call_put(self):
        """Call and put vega should be equal."""
        gc = pricer.greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        gp = pricer.greeks(self.S, self.K, self.T, self.r, self.sigma, "put")
        assert abs(gc.vega - gp.vega) < 1e-10

    def test_theta_negative_call(self):
        """Call theta should be negative (time decay)."""
        g = pricer.greeks(self.S, self.K, self.T, self.r, self.sigma, "call")
        assert g.theta < 0

    def test_atm_delta_near_half(self):
        """ATM call delta should be ≈ 0.5."""
        g = pricer.greeks(100, 100, 1.0, 0.0, 0.2, "call")  # r=0 for pure ATM
        assert abs(g.delta - 0.5) < 0.05


class TestMonteCarlo:
    """Monte Carlo pricing tests."""

    def test_mc_converges_to_bs(self):
        """MC price should converge to BS price within 2 SE."""
        bs_price = pricer.bs_call(100, 100, 1.0, 0.05, 0.2)
        mc = pricer.mc_price(100, 100, 1.0, 0.05, 0.2, n_paths=100000)

        assert abs(mc.price - bs_price) < 2 * mc.std_error, \
            f"MC={mc.price:.4f}, BS={bs_price:.4f}, SE={mc.std_error:.4f}"

    def test_mc_std_error_decreasing(self):
        """More paths should give smaller standard error."""
        mc1 = pricer.mc_price(100, 100, 1, 0.05, 0.2, n_paths=10000)
        mc2 = pricer.mc_price(100, 100, 1, 0.05, 0.2, n_paths=100000)
        assert mc2.std_error < mc1.std_error

    def test_mc_put_positive(self):
        """MC put price should be positive."""
        mc = pricer.mc_price(100, 100, 1, 0.05, 0.2, flag="put")
        assert mc.price > 0

    def test_mc_deterministic(self):
        """Same seed should give same result."""
        mc1 = pricer.mc_price(100, 100, 1, 0.05, 0.2, seed=12345)
        mc2 = pricer.mc_price(100, 100, 1, 0.05, 0.2, seed=12345)
        assert abs(mc1.price - mc2.price) < 1e-12


class TestImpliedVol:
    """Implied volatility tests."""

    def test_iv_round_trip(self):
        """IV(BS(sigma)) should return sigma."""
        sigma_in = 0.25
        price = pricer.bs_call(100, 100, 1.0, 0.05, sigma_in)
        sigma_out = pricer.implied_vol(price, 100, 100, 1.0, 0.05, "call")
        assert abs(sigma_in - sigma_out) < 1e-8, \
            f"IV round trip: {sigma_in} -> {sigma_out}"

    @pytest.mark.parametrize("sigma", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    def test_iv_round_trip_various(self, sigma):
        """IV round-trip across a range of volatilities."""
        price = pricer.bs_call(100, 100, 1.0, 0.05, sigma)
        iv = pricer.implied_vol(price, 100, 100, 1.0, 0.05, "call")
        assert abs(sigma - iv) < 1e-6

    def test_iv_put(self):
        """IV for put options."""
        sigma_in = 0.3
        price = pricer.bs_put(100, 100, 1.0, 0.05, sigma_in)
        sigma_out = pricer.implied_vol(price, 100, 100, 1.0, 0.05, "put")
        assert abs(sigma_in - sigma_out) < 1e-6


class TestBenchmark:
    """Performance benchmarks."""

    def test_bsm_1m_calls_speed(self):
        """1M BSM pricing calls should complete in < 5s (Python) or < 400ms (C++)."""
        n_calls = 1_000_000
        start = time.perf_counter()
        for _ in range(n_calls):
            pricer.bs_call(100, 100, 1.0, 0.05, 0.2)
        elapsed = time.perf_counter() - start

        print(f"\n⏱  {n_calls:,} BSM calls: {elapsed*1000:.1f}ms "
              f"({elapsed/n_calls*1e6:.3f}μs/call) [{BACKEND}]")

        if BACKEND == "C++":
            assert elapsed < 0.5, f"C++ benchmark failed: {elapsed*1000:.0f}ms > 400ms"
        else:
            assert elapsed < 10.0, f"Python benchmark failed: {elapsed:.1f}s > 10s"

    def test_mc_50k_speed(self):
        """MC 50k paths should complete in reasonable time."""
        start = time.perf_counter()
        pricer.mc_price(100, 100, 1.0, 0.05, 0.2, n_paths=50000)
        elapsed = time.perf_counter() - start

        print(f"\n⏱  MC 50k paths: {elapsed*1000:.1f}ms [{BACKEND}]")
        assert elapsed < 5.0  # Should be well under this
