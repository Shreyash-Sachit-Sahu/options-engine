"""
Implied Volatility Surface Construction

Builds a 3D implied volatility surface from live or cached SPY
options chain data. Maps strike x expiry -> implied volatility
using the C++ Brent's method IV solver.
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pricer.pricer_py import implied_vol, bs_call


def build_vol_surface(spot: float, r: float = 0.05,
                      max_expiries: int = 8,
                      min_volume: int = 10) -> pd.DataFrame:
    """
    Build an implied volatility surface from live SPY options data.

    For each available expiry and strike, computes the implied
    volatility by inverting the BS formula using Brent's method.

    Args:
        spot: Current spot price of SPY
        r: Risk-free rate
        max_expiries: Maximum number of expiry dates to include
        min_volume: Minimum volume filter for reliable quotes

    Returns:
        DataFrame with columns: [strike, expiry, T, iv, moneyness, mid_price]
    """
    import yfinance as yf

    spy = yf.Ticker("SPY")
    expiries = spy.options

    if not expiries:
        print("[DATA] No live options data available. Using synthetic surface.")
        return build_synthetic_surface(spot, r)

    expiries = expiries[:max_expiries]
    today = datetime.now()

    surface_data = []

    for exp_str in expiries:
        try:
            chain = spy.option_chain(exp_str)
            calls = chain.calls

            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T = max((exp_date - today).days / 365.0, 1 / 365.0)

            if T <= 0:
                continue

            # Filter for liquid options
            calls_filtered = calls[
                (calls["volume"] >= min_volume) &
                (calls["lastPrice"] > 0.01) &
                (calls["strike"] > spot * 0.7) &
                (calls["strike"] < spot * 1.3)
            ]

            for _, row in calls_filtered.iterrows():
                try:
                    K = float(row["strike"])
                    market_price = float(row["lastPrice"])

                    # Compute implied vol via Brent's method
                    iv = implied_vol(market_price, spot, K, T, r, "call")

                    if 0.01 < iv < 2.0:  # Sanity filter
                        surface_data.append({
                            "strike": K,
                            "expiry": exp_str,
                            "T": T,
                            "iv": iv,
                            "moneyness": K / spot,
                            "mid_price": market_price,
                            "volume": int(row.get("volume", 0)),
                            "open_interest": int(row.get("openInterest", 0)),
                        })
                except Exception:
                    continue  # Skip if IV can't be computed

        except Exception as e:
            print(f"   [WARN] Skipping expiry {exp_str}: {e}")
            continue

    if not surface_data:
        print("[WARN] No valid IV points computed. Using synthetic surface.")
        return build_synthetic_surface(spot, r)

    df = pd.DataFrame(surface_data)
    print(f"[OK] Built IV surface: {len(df)} points across "
          f"{df['expiry'].nunique()} expiries, "
          f"{df['strike'].nunique()} strikes")

    return df


def build_synthetic_surface(spot: float = 500.0, r: float = 0.05) -> pd.DataFrame:
    """
    Build a synthetic (parametric) IV surface for demo/testing.

    Uses a simple SSVI-like parameterization:
      IV(K, T) = base_vol + skew * log(K/S) / sqrt(T) + smile * (log(K/S))^2 / T

    This produces a realistic-looking volatility smile/skew.
    """
    base_vol = 0.20
    skew = -0.10    # Negative skew (typical equity)
    smile = 0.05    # Smile curvature
    term_slope = -0.02  # Vol decreases slightly with maturity

    # Grid
    T_values = [7, 14, 30, 45, 60, 90, 120, 180]  # days
    moneyness_range = np.linspace(0.85, 1.15, 40)

    surface_data = []

    for T_days in T_values:
        T = T_days / 365.0
        exp_date = datetime.now() + pd.Timedelta(days=T_days)
        exp_str = exp_date.strftime("%Y-%m-%d")

        for m in moneyness_range:
            K = spot * m
            log_m = np.log(m)

            # SSVI-inspired parameterization
            iv = (base_vol
                  + skew * log_m / np.sqrt(T)
                  + smile * log_m ** 2 / T
                  + term_slope * np.sqrt(T))

            iv = max(iv, 0.05)  # Floor

            # Compute mid price from IV
            price = bs_call(spot, K, T, r, iv)

            surface_data.append({
                "strike": round(K, 2),
                "expiry": exp_str,
                "T": T,
                "iv": iv,
                "moneyness": m,
                "mid_price": price,
                "volume": int(np.random.randint(100, 10000)),
                "open_interest": int(np.random.randint(500, 50000)),
            })

    df = pd.DataFrame(surface_data)
    print(f"[OK] Built synthetic IV surface: {len(df)} points")
    return df


def surface_to_grid(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert surface DataFrame to regular grids for 3D plotting.

    Returns:
        (strikes_2d, T_2d, iv_2d) — meshgrid arrays
    """
    pivot = df.pivot_table(
        values="iv",
        index="strike",
        columns="T",
        aggfunc="mean"
    )

    # Interpolate missing values
    pivot = pivot.interpolate(method="linear", axis=0).interpolate(
        method="linear", axis=1
    )
    pivot = pivot.ffill().bfill()

    strikes = pivot.index.values
    T_values = pivot.columns.values
    iv_matrix = pivot.values

    T_2d, strikes_2d = np.meshgrid(T_values, strikes)

    return strikes_2d, T_2d, iv_matrix


if __name__ == "__main__":
    print("Building IV surface from live SPY data...")

    # Try live data first
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1d")
        spot = float(hist["Close"].iloc[-1])
        print(f"SPY Spot: ${spot:.2f}")
    except Exception:
        spot = 500.0
        print(f"Using default spot: ${spot:.2f}")

    df = build_vol_surface(spot)
    print(f"[OPTIONS] Available expiries: {df['expiry'].unique()[:8]}...")
    print(f"\nIV range: [{df['iv'].min():.4f}, {df['iv'].max():.4f}]")
    print(f"Strike range: [{df['strike'].min():.2f}, {df['strike'].max():.2f}]")
    print(f"Expiries: {sorted(df['expiry'].unique())}")
