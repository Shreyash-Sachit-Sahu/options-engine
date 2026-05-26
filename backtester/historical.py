"""
Historical Backtester — yfinance Replay Environment  v2

Key upgrades over v1:
  - VIX as IV proxy       : uses ^VIX daily close as 30-day implied vol
                            for SPY, replacing the inaccurate realized-vol
                            proxy. vol_carry = realized_vol / VIX is now
                            a meaningful signal.
  - Per-episode IV ref    : each episode uses the VIX value at its start
                            date, not a single fixed sigma for all episodes.
  - More baselines        : LelandHedger and WhalleyWilmottHedger added
                            alongside DeltaHedger for richer comparison.
  - Stride=1 default      : maximises episode count for robust statistics.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pricer.pricer_py import bs_call, greeks_call, implied_vol


# ─────────────────────────────────────────────────────────────────────────────
# Data fetchers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_spy_data(period: str = "1y") -> pd.DataFrame:
    """Fetch historical SPY price data via yfinance."""
    import yfinance as yf
    spy  = yf.Ticker("SPY")
    hist = spy.history(period=period)
    if hist.empty:
        raise ValueError("Failed to fetch SPY data from yfinance")
    hist = hist.reset_index()
    hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
    hist = hist.sort_values('Date').reset_index(drop=True)
    print(f"📈 Fetched {len(hist)} days of SPY data "
          f"({hist['Date'].iloc[0].date()} to {hist['Date'].iloc[-1].date()})")
    return hist


def fetch_vix_data(period: str = "1y") -> pd.DataFrame:
    """
    Fetch VIX daily close as 30-day implied vol proxy for SPY.

    VIX represents the market's expectation of 30-day SPY volatility
    (annualised, in percentage points). Dividing by 100 gives the
    implied vol in the same units as sigma in the BSM formula.

    This is far more accurate than using realised vol as an IV proxy —
    realised vol lags by 20 days, while VIX is forward-looking.
    """
    import yfinance as yf
    vix  = yf.Ticker("^VIX")
    hist = vix.history(period=period)
    if hist.empty:
        raise ValueError("Failed to fetch VIX data from yfinance")
    hist = hist.reset_index()
    hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
    hist = hist.sort_values('Date').reset_index(drop=True)
    hist['iv'] = hist['Close'] / 100.0   # convert % → decimal
    print(f"📊 Fetched {len(hist)} days of VIX data  "
          f"(mean IV={hist['iv'].mean():.1%}, "
          f"range [{hist['iv'].min():.1%}, {hist['iv'].max():.1%}])")
    return hist[['Date', 'iv']]


def fetch_options_chain(expiry: Optional[str] = None
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch current SPY options chain (for reference / live use)."""
    import yfinance as yf
    spy     = yf.Ticker("SPY")
    expiries = spy.options
    if not expiries:
        raise ValueError("No options data available for SPY")
    if expiry is None:
        expiry = expiries[min(2, len(expiries) - 1)]
    chain = spy.option_chain(expiry)
    return chain.calls, chain.puts


# ─────────────────────────────────────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalBacktester:
    """
    Replays historical SPY price paths to evaluate hedging agents.

    v2 improvements:
      - iv_series: per-day implied vol from VIX (replaces fixed sigma)
      - vol_carry = realized_vol / iv_at_episode_start (now meaningful)
      - Each episode uses the VIX value on its start date as the
        reference implied vol for premium pricing and vol_carry.
    """

    def __init__(self, prices: np.ndarray, iv_series: np.ndarray,
                 K: float, r: float = 0.05,
                 T_days: int = 30, transaction_cost: float = 0.003):
        """
        Args:
            prices:           Historical daily closing prices
            iv_series:        Daily implied vol (VIX/100), aligned with prices
            K:                Strike price (used as fallback; each episode
                              sets ATM strike = S0)
            r:                Risk-free rate
            T_days:           Option lifetime in trading days
            transaction_cost: One-way proportional TC rate
        """
        assert len(prices) == len(iv_series), \
            "prices and iv_series must have the same length"
        self.prices    = prices
        self.iv_series = iv_series
        self.K         = K
        self.r         = r
        self.T_days    = T_days
        self.tc_rate   = transaction_cost

    def run_episode(self, agent, start_idx: int = 0,
                    residual_action: bool = False) -> dict:
        """
        Run one 30-day hedging episode.

        The implied vol at episode start (VIX[start_idx]) is used as:
          - The reference sigma for pricing the initial option premium
          - The denominator in vol_carry = realized_vol / iv_start
            (so vol_carry > 1 means realised vol exceeded implied vol)
        """
        if start_idx + self.T_days >= len(self.prices):
            raise ValueError("Not enough data for a full episode")

        S0      = self.prices[start_idx]
        K       = S0                          # ATM strike per episode
        T       = self.T_days / 252
        iv_start = float(self.iv_series[start_idx])  # VIX at episode start
        iv_start = max(iv_start, 0.05)        # floor at 5%

        # Normalise to training-env scale (trained on S0=100)
        scale        = 100.0 / S0
        premium      = bs_call(S0, K, T, self.r, iv_start) * scale
        premium      = max(premium, 0.5)
        portfolio_value = premium
        hedge_position  = 0.0

        pnl_history, hedge_history, price_history = [], [], [S0]

        if hasattr(agent, 'reset'):
            agent.reset()

        for step in range(self.T_days):
            price       = self.prices[start_idx + step]
            T_remaining = max((self.T_days - step) / 252, 1e-10)

            # ── Realised vol from recent returns ──────────────────────────
            lookback = min(step + 1, 20)
            if lookback >= 2:
                recent = self.prices[
                    start_idx + max(0, step - lookback):start_idx + step + 1
                ]
                log_ret      = np.diff(np.log(recent))
                realized_vol = float(np.std(log_ret) * np.sqrt(252))
                realized_vol = max(realized_vol, 0.05)
            else:
                realized_vol = iv_start

            # ── Greeks ────────────────────────────────────────────────────
            try:
                g     = greeks_call(price, K, T_remaining, self.r, realized_vol)
                delta = float(g.delta)
                gamma = float(g.gamma)
                vega  = float(g.vega)
                theta = float(g.theta)
            except Exception:
                delta, gamma, vega, theta = 0.5, 0.01, 0.0, 0.0

            # ── Observation ───────────────────────────────────────────────
            pnl_norm    = (portfolio_value - premium) / max(premium, 1e-6)
            # vol_carry: realized / implied — now uses VIX as denominator
            vol_carry   = float(np.clip(realized_vol / max(iv_start, 1e-6),
                                        0.0, 5.0))
            norm_factor = S0 / 100.0

            if vol_carry < 0.85:
                vol_regime = 0.0
            elif vol_carry < 1.25:
                vol_regime = 0.5
            else:
                vol_regime = 1.0

            obs = np.array([
                price / S0,
                K     / S0,
                T_remaining / T,
                realized_vol,
                delta,
                (gamma * 100.0) / norm_factor,
                float(np.clip(pnl_norm, -10, 10)),
                vega  / norm_factor,
                float(np.clip(-theta / norm_factor, 0.0, 10.0)),
                vol_carry,
                float(np.clip(hedge_position, -1.0, 1.0)),
                vol_regime,
            ], dtype=np.float32)

            # ── Action ────────────────────────────────────────────────────
            action, _ = agent.predict(obs)
            if residual_action:
                target_hedge = float(np.clip(
                    delta + 0.3 * float(action[0]), -1.0, 1.0))
            else:
                target_hedge = float(np.clip(action[0], -1.0, 1.0))

            # ── TC ────────────────────────────────────────────────────────
            tc = self.tc_rate * abs(target_hedge - hedge_position) * (price * scale)
            portfolio_value -= tc
            hedge_position   = target_hedge

            # ── PnL ───────────────────────────────────────────────────────
            next_price = (self.prices[start_idx + step + 1]
                          if step + 1 < self.T_days else price)
            price_change_norm = (next_price - price) * scale
            hedge_pnl         = hedge_position * price_change_norm

            T_next   = max((self.T_days - step - 1) / 252, 1e-10)
            old_opt  = bs_call(price,      K, T_remaining, self.r, realized_vol) * scale
            new_opt  = bs_call(next_price, K, T_next,      self.r, realized_vol) * scale
            option_pnl = -(new_opt - old_opt)

            total_pnl = hedge_pnl + option_pnl
            portfolio_value += total_pnl
            pnl_history.append(total_pnl)
            hedge_history.append(target_hedge)
            price_history.append(next_price)

        # ── Final settlement ──────────────────────────────────────────────
        final_price = self.prices[start_idx + self.T_days]
        intrinsic   = max(final_price - K, 0.0) * scale
        portfolio_value -= intrinsic

        pnls      = np.array(pnl_history)
        pnls_norm = pnls / max(premium, 1e-6)

        sharpe  = float(np.mean(pnls_norm) / (np.std(pnls_norm) + 1e-10))
        cum_pnl = np.cumsum(pnls_norm)
        max_dd  = float(np.max(np.maximum.accumulate(cum_pnl) - cum_pnl))

        return {
            "portfolio_value": float(portfolio_value),
            "total_pnl":       float(np.sum(pnls_norm)),
            "total_pnl_raw":   float(np.sum(pnls)),
            "premium":         float(premium),
            "iv_start":        float(iv_start),
            "pnl_history":     pnls_norm.tolist(),
            "hedge_history":   hedge_history,
            "price_history":   price_history,
            "sharpe":          sharpe,
            "max_drawdown":    max_dd,
        }

    def run_backtest(self, agent, stride: int = 1,
                     residual_action: bool = False) -> dict:
        """Rolling backtest across full price history."""
        n_possible = len(self.prices) - self.T_days - 1
        if n_possible <= 0:
            raise ValueError("Not enough data")

        episodes = []
        for start in range(0, n_possible, stride):
            try:
                episodes.append(self.run_episode(
                    agent, start_idx=start,
                    residual_action=residual_action))
            except Exception:
                continue

        if not episodes:
            raise ValueError("No successful episodes")

        sharpes   = [e["sharpe"]     for e in episodes]
        pnls      = [e["total_pnl"]  for e in episodes]
        drawdowns = [e["max_drawdown"] for e in episodes]

        return {
            "n_episodes":          len(episodes),
            "sharpe_ratio":        float(np.mean(sharpes)),
            "cross_ep_sharpe":     float(np.mean(pnls) /
                                         (np.std(pnls) + 1e-10)),
            "mean_pnl":            float(np.mean(pnls)),
            "std_pnl":             float(np.std(pnls)),
            "mean_sharpe":         float(np.mean(sharpes)),
            "std_sharpe":          float(np.std(sharpes)),
            "max_max_drawdown":    float(np.max(drawdowns)),
            "episodes":            episodes,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    from scipy import stats as scipy_stats

    parser = argparse.ArgumentParser(description="Historical SPY backtest v2")
    parser.add_argument("--period",     default="2y")
    parser.add_argument("--stride",     type=int,   default=1)
    parser.add_argument("--tc",         type=float, default=0.003)
    parser.add_argument("--model-path", default="agent/models/sac_hedger_final")
    parser.add_argument("--vnorm-path", default=None)
    parser.add_argument("--output",     default="agent/historical_results.json")
    args = parser.parse_args()

    print("=" * 65)
    print("  Historical SPY Backtest v2 — VIX as IV proxy")
    print("=" * 65)

    # ── Fetch SPY + VIX ──────────────────────────────────────────────
    spy_df  = fetch_spy_data(args.period)
    vix_df  = fetch_vix_data(args.period)

    # Align on common dates
    merged  = spy_df.merge(vix_df, on='Date', how='inner')
    prices  = merged['Close'].values.astype(float)
    iv_ser  = merged['iv'].values.astype(float)

    print(f"\n  Aligned {len(prices)} trading days")
    print(f"  SPY range : ${prices.min():.2f} – ${prices.max():.2f}")
    print(f"  VIX range : {iv_ser.min():.1%} – {iv_ser.max():.1%}")

    K     = float(np.median(prices))
    sigma = float(np.mean(iv_ser))   # mean VIX as reference sigma
    print(f"  Mean IV (VIX): {sigma:.1%}  |  ATM strike: ${K:.2f}")

    bt = HistoricalBacktester(prices, iv_ser, K=K, r=0.05,
                               transaction_cost=args.tc)

    # ── VecNormalize path ─────────────────────────────────────────────
    if args.vnorm_path is None:
        model_dir = Path(args.model_path).parent
        candidates = [
            str(model_dir / "vec_normalize_heston.pkl"),
            str(model_dir / "vec_normalize.pkl"),
        ]
        args.vnorm_path = next(
            (c for c in candidates if Path(c).exists()),
            candidates[-1])
    print(f"  VecNormalize : {args.vnorm_path}")

    all_results = {}

    # ── Baselines ─────────────────────────────────────────────────────
    from environment.baselines import (
        DeltaHedger, LelandHedger, WhalleyWilmottHedger
    )

    baseline_specs = [
        ("DeltaHedger",          DeltaHedger(K=K, sigma=sigma)),
        ("LelandHedger",         LelandHedger(sigma=sigma, tc_rate=args.tc,
                                              K=K, r=0.05, T=30/252)),
        ("WhalleyWilmottHedger", WhalleyWilmottHedger(sigma=sigma,
                                                       tc_rate=args.tc,
                                                       K=K, r=0.05,
                                                       T=30/252)),
    ]

    for name, agent in baseline_specs:
        print(f"\n[{name}] Running backtest...")
        res = bt.run_backtest(agent, stride=args.stride)
        all_results[name] = res
        print(f"  Episodes       : {res['n_episodes']}")
        print(f"  Mean Sharpe    : {res['sharpe_ratio']:.4f}")
        print(f"  Cross-ep Sharpe: {res['cross_ep_sharpe']:.4f}")
        print(f"  Mean PnL       : {res['mean_pnl']:.6f}")

    # ── SAC agent ─────────────────────────────────────────────────────
    model_zip = args.model_path + ".zip"
    if Path(args.model_path).exists() or Path(model_zip).exists():
        print(f"\n[SAC] Loading from {args.model_path}...")
        from stable_baselines3 import SAC as SB3SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from agent.gpu_utils import get_device
        from environment.options_env import OptionsHedgingEnv

        device    = get_device(verbose=True)
        sac_model = SB3SAC.load(args.model_path, device=device)

        class SACAgent:
            def __init__(self, model, vnorm_path):
                self.model = model
                self._vn   = None
                if os.path.exists(vnorm_path):
                    try:
                        dummy = DummyVecEnv([lambda: OptionsHedgingEnv()])
                        vn    = VecNormalize.load(vnorm_path, dummy)
                        vn.training    = False
                        vn.norm_reward = False
                        self._vn = vn
                        print(f"[SAC] VecNormalize loaded "
                              f"(obs_mean={vn.obs_rms.mean[:3].round(3)}...)")
                    except Exception as e:
                        print(f"[WARN] VecNormalize load failed: {e}")
                else:
                    print(f"[WARN] VecNormalize not found at {vnorm_path}")

            def predict(self, obs):
                obs_in = obs.reshape(1, -1).astype(np.float32)
                if self._vn is not None:
                    obs_in = self._vn.normalize_obs(obs_in)
                action, state = self.model.predict(obs_in, deterministic=True)
                return action[0], state

        sac_agent = SACAgent(sac_model, args.vnorm_path)
        print("[SAC] Running backtest...")
        sac_res = bt.run_backtest(sac_agent, stride=args.stride,
                                   residual_action=True)
        all_results["SAC"] = sac_res
        print(f"  Episodes       : {sac_res['n_episodes']}")
        print(f"  Mean Sharpe    : {sac_res['sharpe_ratio']:.4f}")
        print(f"  Cross-ep Sharpe: {sac_res['cross_ep_sharpe']:.4f}")
        print(f"  Mean PnL       : {sac_res['mean_pnl']:.6f}")
    else:
        print(f"\n[WARN] No SAC model at {args.model_path}")

    # ── Statistical comparison ────────────────────────────────────────
    if "SAC" in all_results:
        sac_pnls = np.array([e["total_pnl"]
                              for e in all_results["SAC"]["episodes"]])

        print(f"\n{'=' * 65}")
        print(f"  STATISTICAL RESULTS  (n={len(sac_pnls)} episodes)")
        print(f"{'=' * 65}")

        for name in ["DeltaHedger", "LelandHedger", "WhalleyWilmottHedger"]:
            if name not in all_results:
                continue
            other_pnls = np.array([e["total_pnl"]
                                    for e in all_results[name]["episodes"]])
            n = min(len(sac_pnls), len(other_pnls))
            t_stat, p_val = scipy_stats.ttest_rel(
                sac_pnls[:n], other_pnls[:n])
            diff = float(np.mean(sac_pnls[:n] - other_pnls[:n]))

            rng  = np.random.default_rng(42)
            boot = [np.mean(sac_pnls[:n][rng.integers(0, n, n)] -
                            other_pnls[:n][rng.integers(0, n, n)])
                    for _ in range(10_000)]
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

            sac_sh   = all_results["SAC"]["sharpe_ratio"]
            other_sh = all_results[name]["sharpe_ratio"]
            outperf  = (sac_sh / max(other_sh, 1e-10) - 1) * 100
            sig      = "✓" if p_val < 0.05 else "✗"

            print(f"\n  SAC vs {name}:")
            print(f"    Sharpe  : {sac_sh:.4f} vs {other_sh:.4f}  "
                  f"({outperf:+.1f}%)")
            print(f"    PnL diff: {diff:+.6f}  CI [{ci_lo:+.6f}, "
                  f"{ci_hi:+.6f}]")
            print(f"    p-value : {p_val:.4f}  {sig}")

            all_results[f"stats_SAC_vs_{name}"] = {
                "t_stat":    float(t_stat),
                "p_value":   float(p_val),
                "mean_diff": diff,
                "ci_lo":     float(ci_lo),
                "ci_hi":     float(ci_hi),
                "sig":       bool(p_val < 0.05),
                "outperf_pct": float(outperf),
            }

    # ── Save ─────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            save[k] = {kk: vv for kk, vv in v.items() if kk != "episodes"}
        else:
            save[k] = v
    with open(out, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n[SAVE] {out}")