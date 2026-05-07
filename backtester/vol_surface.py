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


def _market_is_open() -> bool:
    """Return True if NYSE is currently in regular trading hours (ET)."""
    try:
        from datetime import timezone, timedelta
        ET = timezone(timedelta(hours=-4))   # EDT (UTC-4); EST is UTC-5
        now = datetime.now(ET)
        if now.weekday() >= 5:               # Saturday / Sunday
            return False
        open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
        return open_t <= now <= close_t
    except Exception:
        return False


def build_vol_surface(spot: float, r: float = 0.05,
                      max_expiries: int = 8,
                      min_volume: int = 0) -> pd.DataFrame:
    """
    Build an implied volatility surface from live SPY options data.

    For each available expiry and strike, computes the implied
    volatility by inverting the BS formula using Brent's method.

    Mid-price logic (in order of preference):
        1. (bid + ask) / 2  — most reliable when market is open
        2. lastPrice        — fallback when bid/ask are zero/NaN (closed market)

    Volume filter:
        NaN volume is treated as unknown (kept). Only rows with a known
        volume strictly below min_volume are dropped. Set min_volume=0
        to disable the filter entirely (recommended outside market hours).

    Args:
        spot:         Current spot price of SPY
        r:            Risk-free rate
        max_expiries: Maximum number of expiry dates to include
        min_volume:   Minimum known volume to keep a row (0 = no filter)

    Returns:
        DataFrame with columns: [strike, expiry, T, iv, moneyness,
                                  mid_price, volume, open_interest, spread_pct]
    """
    import yfinance as yf

    if not _market_is_open():
        print("[INFO] Market is currently closed — bid/ask quotes will be 0.")
        print("       Using lastPrice as fallback. IV quality may be lower.")
        print("       For best results run during NYSE hours (9:30–16:00 ET).")

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

            # ── Compute mid-price (bid/ask preferred, lastPrice fallback) ───
            if "bid" in calls.columns and "ask" in calls.columns:
                calls["_mid"] = (
                    calls["bid"].fillna(0) + calls["ask"].fillna(0)
                ) / 2
                # Fall back to lastPrice where bid/ask are both zero/NaN
                mask = calls["_mid"] <= 0.01
                calls.loc[mask, "_mid"] = calls.loc[mask, "lastPrice"].fillna(0)
            else:
                calls["_mid"] = calls["lastPrice"].fillna(0)

            # ── Volume filter: NaN volume is common — treat as unknown, keep ─
            vol_ok = calls["volume"].isna() | (calls["volume"] >= min_volume)

            calls_filtered = calls[
                vol_ok &
                (calls["_mid"] > 0.05) &
                (calls["strike"] > spot * 0.70) &
                (calls["strike"] < spot * 1.30)
            ].copy()

            if calls_filtered.empty:
                continue

            for _, row in calls_filtered.iterrows():
                try:
                    K           = float(row["strike"])
                    market_price = float(row["_mid"])

                    # Use bid/ask spread as a liquidity proxy
                    spread = 0.0
                    if "bid" in row and "ask" in row:
                        b = row["bid"] if pd.notna(row["bid"]) else 0
                        a = row["ask"] if pd.notna(row["ask"]) else 0
                        if a > 0:
                            spread = (a - b) / a  # relative spread

                    # Skip if spread is extremely wide (stale quote)
                    if spread > 0.8:
                        continue

                    # Compute implied vol via Brent's method
                    iv = implied_vol(market_price, spot, K, T, r, "call")

                    if 0.02 < iv < 3.0:  # Sanity filter
                        surface_data.append({
                            "strike":         K,
                            "expiry":         exp_str,
                            "T":              T,
                            "iv":             iv,
                            "moneyness":      K / spot,
                            "mid_price":      market_price,
                            "volume":         int(row["volume"]) if pd.notna(row.get("volume")) else 0,
                            "open_interest":  int(row["openInterest"]) if pd.notna(row.get("openInterest")) else 0,
                            "spread_pct":     round(spread, 4),
                        })
                except Exception:
                    continue  # Skip if IV can't be computed

        except Exception as e:
            print(f"   [WARN] Skipping expiry {exp_str}: {e}")
            continue

    if not surface_data:
        print("[WARN] No valid IV points computed. Using synthetic surface.")
        print("       Common causes:")
        print("         1. volume NaN + min_volume filter dropped all rows  (now fixed)")
        print("         2. Market closed — lastPrice is stale, bid/ask = 0")
        print("         3. IV solver failed for all strikes (check pricer)")
        print("       Tip: run with min_volume=0 to bypass volume filter entirely.")
        return build_synthetic_surface(spot, r)

    df = pd.DataFrame(surface_data)
    live_pct = len(df) / max(sum(
        len(spy.option_chain(e).calls) for e in expiries
        if e in df["expiry"].values
    ), 1) * 100

    print(f"[OK] Built live IV surface: {len(df)} points across "
          f"{df['expiry'].nunique()} expiries, "
          f"{df['strike'].nunique()} strikes  "
          f"(IV range: {df['iv'].min():.2f}–{df['iv'].max():.2f})")

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