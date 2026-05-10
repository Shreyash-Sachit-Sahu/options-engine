"""
Historical Backtester — yfinance Replay Environment

Downloads real SPY options chain data via yfinance and replays
historical price paths to test the trained SAC agent on real
market conditions.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pricer.pricer_py import bs_call, greeks_call, implied_vol


def fetch_spy_data(period: str = "6mo") -> pd.DataFrame:
    """
    Fetch historical SPY price data via yfinance.

    Args:
        period: Data period ('1mo', '3mo', '6mo', '1y')

    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume
    """
    import yfinance as yf

    spy = yf.Ticker("SPY")
    hist = spy.history(period=period)

    if hist.empty:
        raise ValueError("Failed to fetch SPY data from yfinance")

    hist = hist.reset_index()
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist = hist.sort_values('Date').reset_index(drop=True)

    print(f"📈 Fetched {len(hist)} days of SPY data "
          f"({hist['Date'].iloc[0].date()} to {hist['Date'].iloc[-1].date()})")

    return hist


def fetch_options_chain(expiry: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch current SPY options chain from yfinance.

    Args:
        expiry: Specific expiry date string (YYYY-MM-DD). If None, uses nearest.

    Returns:
        (calls_df, puts_df) DataFrames with strike, lastPrice, impliedVolatility, etc.
    """
    import yfinance as yf

    spy = yf.Ticker("SPY")
    expiries = spy.options  # list of available expiry dates

    if not expiries:
        raise ValueError("No options data available for SPY")

    if expiry is None:
        expiry = expiries[min(2, len(expiries) - 1)]  # pick ~1 month out

    print(f"📋 Available expiries: {expiries[:8]}...")
    print(f"   Selected: {expiry}")

    chain = spy.option_chain(expiry)
    return chain.calls, chain.puts


class HistoricalBacktester:
    """
    Replays historical price data to evaluate hedging agents.

    Instead of simulating prices with GBM/Heston, this uses actual
    SPY closing prices to step through time. This tests whether
    the RL agent generalizes to real market dynamics.
    """

    def __init__(self, prices: np.ndarray, K: float, r: float = 0.05,
                 sigma: float = 0.2, T_days: int = 30,
                 transaction_cost: float = 0.001):
        """
        Args:
            prices: Array of historical daily closing prices
            K: Strike price for the option
            r: Risk-free rate
            sigma: Initial implied volatility estimate
            T_days: Option lifetime in trading days
            transaction_cost: Cost rate per unit traded
        """
        self.prices = prices
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T_days = T_days
        self.tc_rate = transaction_cost

    def run_episode(self, agent, start_idx: int = 0,
                    residual_action: bool = False) -> dict:
        """
        Run a single backtest episode starting from start_idx.

        Args:
            agent: Agent with .predict(obs) -> (action, state)
            start_idx: Starting index in price array

        Returns:
            Episode results dict
        """
        if start_idx + self.T_days >= len(self.prices):
            raise ValueError("Not enough data for a full episode")

        S0 = self.prices[start_idx]
        K = S0  # ATM strike per episode — avoids deep OTM/ITM premium explosion
        T = self.T_days / 252

        # Initial option premium — normalised to training-env scale (S0=100)
        scale = 100.0 / S0
        premium = bs_call(S0, K, T, self.r, self.sigma) * scale
        premium = max(premium, 0.5)  # floor: prevents division explosion on near-zero OTM premiums
        portfolio_value = premium
        hedge_position = 0.0

        pnl_history = []
        hedge_history = []
        price_history = [S0]

        if hasattr(agent, 'reset'):
            agent.reset()

        for step in range(self.T_days):
            price = self.prices[start_idx + step]
            T_remaining = max((self.T_days - step) / 252, 1e-10)

            # Compute realized vol from recent returns
            lookback = min(step + 1, 20)
            if lookback >= 2:
                recent_prices = self.prices[
                    start_idx + max(0, step - lookback):start_idx + step + 1
                ]
                log_returns = np.diff(np.log(recent_prices))
                realized_vol = float(np.std(log_returns) * np.sqrt(252))
                realized_vol = max(realized_vol, 0.05)  # floor
            else:
                realized_vol = self.sigma

            # Compute Greeks
            try:
                g = greeks_call(price, K, T_remaining, self.r, realized_vol)
                delta = g.delta
                gamma = g.gamma
            except:
                delta = 0.5
                gamma = 0.01

            # Build 11-dim observation (matches options_env._get_obs)
            pnl_norm = (portfolio_value - premium) / max(premium, 1e-6)

            # Vol carry: realized / implied ratio
            vol_carry = float(np.clip(realized_vol / max(self.sigma, 1e-6), 0.0, 5.0))

            try:
                vega  = float(greeks_call(price, K, T_remaining, self.r, realized_vol).vega)
                theta = float(greeks_call(price, K, T_remaining, self.r, realized_vol).theta)
            except Exception:
                vega, theta = 0.0, 0.0

            # Normalise Greeks to training-env scale (trained on S0=100).
            # vega, theta, gamma all scale with spot price, so we
            # divide by (S0/100) to match the magnitude the policy saw
            # during training. Without this, obs[6-8] are 6x out of range
            # and the policy produces garbage actions.
            norm_factor = S0 / 100.0
            obs = np.array([
                price / S0,
                K / S0,
                T_remaining / T,
                realized_vol,
                delta,                                      # dimensionless [0,1]
                (gamma * 100.0) / norm_factor,              # gamma scales with 1/S
                np.clip(pnl_norm, -10, 10),
                vega       / norm_factor,                   # vega scales with S
                np.clip(-theta / norm_factor, 0.0, 10.0),  # negate: short call earns theta decay (matches options_env._get_obs)
                vol_carry,                                  # dimensionless ratio
                float(np.clip(hedge_position, -1.0, 1.0)), # dimensionless [-1,1]
            ], dtype=np.float32)

            # Get action
            action, _ = agent.predict(obs)
            action, _ = agent.predict(obs)
            if residual_action:
                # SAC trained with residual architecture:
                # output is a correction to delta, NOT an absolute hedge.
                # Without this, action=-0.2 means "short 20%" not "reduce by 6%"
                target_hedge = float(np.clip(delta + 0.3 * float(action[0]), -1.0, 1.0))
            else:
                target_hedge = float(np.clip(action[0], -1.0, 1.0))

            # Transaction cost
            tc = self.tc_rate * abs(target_hedge - hedge_position) * (price * 100.0 / S0)
            portfolio_value -= tc

            # Update position
            old_hedge = hedge_position
            hedge_position = target_hedge

            # Next price
            if step + 1 < self.T_days:
                next_price = self.prices[start_idx + step + 1]
            else:
                next_price = price

            # PnL — normalise by S0 so units match the training env.
            # During training, hedge_position is a fraction of 1 unit where
            # the stock is normalised to S0=100. In the real backtest S0
            # can be ~$731, so without normalisation a position of 0.5
            # earns 0.5 * $5 = $2.50/step instead of 0.5 * $0.68 = $0.34.
            # Dividing price_change by S0 restores the correct scaling.
            # Normalise price change to training-env scale (S0=100)
            # hedge_position is in [-1,1], trained on S0=100 prices
            price_change_norm = (next_price - price) * (100.0 / S0)
            hedge_pnl = hedge_position * price_change_norm

            # Option PnL: also normalise option values by S0 for consistency
            T_next = max((self.T_days - step - 1) / 252, 1e-10)
            # Normalise option prices to training-env scale (S0=100)
            # so option_pnl and hedge_pnl are in the same units
            scale = 100.0 / S0
            old_opt = bs_call(price,      K, T_remaining, self.r, realized_vol) * scale
            new_opt = bs_call(next_price, K, T_next,      self.r, realized_vol) * scale
            option_pnl = -(new_opt - old_opt)

            total_pnl = hedge_pnl + option_pnl
            portfolio_value += total_pnl
            pnl_history.append(total_pnl)
            hedge_history.append(target_hedge)
            price_history.append(next_price)

        # Final settlement
        final_price = self.prices[start_idx + self.T_days]
        intrinsic = max(final_price - K, 0.0) * scale
        portfolio_value -= intrinsic

        pnls = np.array(pnl_history)

        # Normalise PnL by premium so results are comparable across
        # different price levels and option maturities.
        # This matches how the training env computes rewards.
        pnls_norm = pnls / max(premium, 1e-6)
        total_pnl_norm = float(np.sum(pnls_norm))

        # Sharpe on normalised daily PnL, annualised
        # Information ratio across 30 steps (not annualised)
        sharpe = float(
            np.mean(pnls_norm) / (np.std(pnls_norm) + 1e-10)
        )
        # Max drawdown on normalised cumulative PnL
        cum_pnl = np.cumsum(pnls_norm)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns_arr = running_max - cum_pnl
        max_dd = float(np.max(drawdowns_arr)) if len(drawdowns_arr) > 0 else 0.0

        return {
            "portfolio_value": portfolio_value,
            "total_pnl":       total_pnl_norm,        # normalised by premium
            "total_pnl_raw":   float(np.sum(pnls)),   # raw dollars (for reference)
            "premium":         premium,
            "pnl_history":     pnls_norm.tolist(),
            "pnl_history_raw": pnl_history,
            "hedge_history":   hedge_history,
            "price_history":   price_history,
            "sharpe":          sharpe,
            "max_drawdown":    max_dd,
            "transaction_costs": float(
                sum(self.tc_rate * abs(hedge_history[i] - (
                    hedge_history[i-1] if i > 0 else 0
                )) * price_history[i] for i in range(len(hedge_history)))
            ) / max(premium, 1e-6),   # also normalised
        }

    def run_backtest(self, agent, stride: int = 5,
                     residual_action: bool = False) -> dict:
        """
        Run rolling backtests across the full price history.

        Args:
            agent: Hedging agent
            stride: Days between episode starts

        Returns:
            Aggregated backtest results
        """
        n_possible = len(self.prices) - self.T_days - 1
        if n_possible <= 0:
            raise ValueError("Not enough data for backtesting")

        episodes = []
        for start in range(0, n_possible, stride):
            try:
                result = self.run_episode(agent, start_idx=start,
                                          residual_action=residual_action)
                episodes.append(result)
            except Exception as e:
                continue

        if not episodes:
            raise ValueError("No successful episodes")

        sharpes = [e["sharpe"] for e in episodes]
        pnls = [e["total_pnl"] for e in episodes]
        drawdowns = [e["max_drawdown"] for e in episodes]

        return {
            "n_episodes": len(episodes),
            "mean_sharpe": float(np.mean(sharpes)),
            "std_sharpe": float(np.std(sharpes)),
            "mean_pnl": float(np.mean(pnls)),
            "std_pnl": float(np.std(pnls)),
            "sharpe_ratio": float(np.mean(sharpes)),  # mean of per-episode Sharpes (correct)
            "sharpe_ratio_cross_ep": float(
                np.mean(pnls) / (np.std(pnls) + 1e-10)
            ),  # cross-episode Sharpe (biased by S0 variation — for reference only)
            "max_max_drawdown": float(np.max(drawdowns)),
            "episodes": episodes,
        }

    @staticmethod
    def _max_drawdown(pnls: np.ndarray) -> float:
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0


if __name__ == "__main__":
    import argparse, json
    from pathlib import Path
    from scipy import stats as scipy_stats

    parser = argparse.ArgumentParser(description="Historical SPY backtest")
    parser.add_argument("--period",     default="1y",
                        help="yfinance period: 6mo, 1y, 2y (default: 1y)")
    parser.add_argument("--stride",     type=int, default=5,
                        help="Days between episode start points (default: 5)")
    parser.add_argument("--tc",         type=float, default=0.001,
                        help="Transaction cost rate (default: 0.001)")
    parser.add_argument("--model-path", default="agent/models/sac_hedger_final",
                        help="Path to trained SAC model")
    parser.add_argument("--vnorm-path", default=None,
                        help="Path to VecNormalize stats (auto-detected if omitted)")
    parser.add_argument("--output",     default="agent/historical_results.json")
    args = parser.parse_args()

    print("=" * 65)
    print("  Historical SPY Backtest — SAC vs Delta Hedger")
    print("=" * 65)

    # ── Fetch data ────────────────────────────────────────────────────
    hist  = fetch_spy_data(args.period)
    prices = hist["Close"].values
    K     = float(np.median(prices))      # median as proxy for ATM strike
    sigma = float(np.std(np.diff(np.log(prices))) * np.sqrt(252))
    print(f"\n  Estimated historical vol: {sigma:.1%}")
    print(f"  ATM strike (median spot): {K:.2f}")

    bt = HistoricalBacktester(prices, K=K, sigma=sigma,
                               transaction_cost=args.tc)

    # Auto-detect VecNormalize path: prefer sim-tagged, fall back to generic
    if args.vnorm_path is None:
        model_dir = Path(args.model_path).parent
        sim_tag = Path(args.model_path).stem.replace("sac_hedger_", "").replace("_final", "")
        candidates = [
            str(model_dir / f"vec_normalize_{sim_tag}.pkl"),
            str(model_dir / "vec_normalize.pkl"),
        ]
        args.vnorm_path = next((c for c in candidates if Path(c).exists()),
                               str(model_dir / "vec_normalize.pkl"))
    print(f"  VecNormalize path : {args.vnorm_path}")

    # ── Delta Hedger baseline ─────────────────────────────────────────
    from environment.baselines import DeltaHedger
    delta_agent = DeltaHedger(K=K, sigma=sigma)
    print(f"\n[DELTA] Running delta hedge backtest...")
    delta_results = bt.run_backtest(delta_agent, stride=args.stride)
    print(f"  Episodes : {delta_results['n_episodes']}")
    print(f"  Sharpe   : {delta_results['sharpe_ratio']:.4f}")
    print(f"  Mean PnL : {delta_results['mean_pnl']:.6f}")
    print(f"  Max DD   : {delta_results['max_max_drawdown']:.6f}")

    all_results = {"DeltaHedger": delta_results}

    # ── SAC Agent ─────────────────────────────────────────────────────
    model_zip = args.model_path + ".zip"
    if Path(args.model_path).exists() or Path(model_zip).exists():
        print(f"\n[SAC] Loading model from {args.model_path}...")

        # Warn if VecNormalize dim doesn't match 11-dim obs space
        try:
            import pickle as _pkl
            with open(args.vnorm_path, "rb") as _f:
                _vn_check = _pkl.load(_f)
            _dim = len(_vn_check.obs_rms.mean)
            if _dim != 11:
                print(f"\n{'='*65}")
                print(f"  [CRITICAL] OBS MISMATCH: VecNormalize is {_dim}-dim, env is 11-dim.")
                print(f"  This model was trained BEFORE the obs space expansion.")
                print(f"  Results will be INVALID. Retrain first:")
                print(f"    train_all.bat   (or: python agent/train.py --simulator heston)")
                print(f"{'='*65}\n")
        except Exception:
            pass

        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from agent.gpu_utils import get_device
            import os

            device = get_device(verbose=True)
            sac_model = SAC.load(args.model_path, device=device)

            # Wrap SAC into an agent interface compatible with run_backtest
            class SACAgent:
                """
                Wraps a SAC model with VecNormalize statistics.

                Uses the actual VecNormalize object (not manual obs_rms)
                to ensure observations are normalised identically to
                how they were during training.
                """
                def __init__(self, model, vnorm_path):
                    self.model = model
                    self._vn = None
                    if os.path.exists(vnorm_path):
                        try:
                            from stable_baselines3.common.vec_env import (
                                DummyVecEnv, VecNormalize
                            )
                            from environment.options_env import OptionsHedgingEnv
                            # Create a dummy env just to load normalisation stats
                            dummy = DummyVecEnv([lambda: OptionsHedgingEnv()])
                            vn = VecNormalize.load(vnorm_path, dummy)
                            vn.training = False   # freeze stats
                            vn.norm_reward = False
                            self._vn = vn
                            print(f"[SAC] VecNormalize loaded: "
                                  f"obs_mean={vn.obs_rms.mean[:3].round(3)}...")
                        except Exception as e:
                            print(f"[WARN] VecNormalize load failed: {e}. "
                                  f"Using raw observations.")
                    else:
                        print(f"[WARN] VecNormalize not found at {vnorm_path}. "
                              f"Observations will NOT be normalised — results "
                              f"may be unreliable.")

                def predict(self, obs):
                    obs_in = obs.reshape(1, -1).astype(np.float32)
                    if self._vn is not None:
                        # normalise using the frozen VecNormalize stats
                        obs_in = self._vn.normalize_obs(obs_in)
                    action, state = self.model.predict(
                        obs_in, deterministic=True
                    )
                    return action[0], state

            sac_agent = SACAgent(sac_model, args.vnorm_path)
            print(f"[SAC] Running SAC backtest on same real SPY paths...")
            sac_results = bt.run_backtest(sac_agent, stride=args.stride,
                                              residual_action=True)
            all_results["SAC"] = sac_results
            print(f"  Episodes : {sac_results['n_episodes']}")
            print(f"  Sharpe   : {sac_results['sharpe_ratio']:.4f}")
            print(f"  Mean PnL : {sac_results['mean_pnl']:.6f}")
            print(f"  Max DD   : {sac_results['max_max_drawdown']:.6f}")

        except Exception as e:
            print(f"[WARN] SAC evaluation failed: {e}")
    else:
        print(f"\n[WARN] No SAC model found at {args.model_path}.")
        print("       Run `python agent/train.py` first.")

    # ── Statistical Significance ──────────────────────────────────────
    if "SAC" in all_results:
        sac_pnls   = np.array([e["total_pnl"] for e in all_results["SAC"]["episodes"]])
        delta_pnls = np.array([e["total_pnl"] for e in all_results["DeltaHedger"]["episodes"]])
        n = min(len(sac_pnls), len(delta_pnls))
        sac_pnls, delta_pnls = sac_pnls[:n], delta_pnls[:n]

        t_stat, p_value = scipy_stats.ttest_rel(sac_pnls, delta_pnls)
        diff_mean = np.mean(sac_pnls - delta_pnls)

        # Bootstrap 95% CI on mean PnL difference
        rng = np.random.default_rng(42)
        boot_diffs = []
        for _ in range(10_000):
            idx = rng.integers(0, n, size=n)
            boot_diffs.append(np.mean(sac_pnls[idx] - delta_pnls[idx]))
        ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

        print(f"\n{'=' * 65}")
        print(f"  STATISTICAL SIGNIFICANCE (paired t-test, n={n} episodes)")
        print(f"{'=' * 65}")
        print(f"  SAC vs Delta mean PnL difference : {diff_mean:+.6f}")
        print(f"  95% Bootstrap CI                 : [{ci_lo:+.6f}, {ci_hi:+.6f}]")
        print(f"  t-statistic                      : {t_stat:.4f}")
        print(f"  p-value                          : {p_value:.4f}")
        sig = "YES ✓" if p_value < 0.05 else "NO ✗ (not significant)"
        print(f"  Significant at 5% level          : {sig}")

        sac_sharpe   = all_results["SAC"]["sharpe_ratio"]
        delta_sharpe = all_results["DeltaHedger"]["sharpe_ratio"]
        outperf      = (sac_sharpe / max(delta_sharpe, 1e-10) - 1) * 100
        print(f"\n  Sharpe outperformance on REAL DATA: {outperf:+.1f}%")

        all_results["statistics"] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_pnl_diff": float(diff_mean),
            "ci_95_lo": float(ci_lo),
            "ci_95_hi": float(ci_hi),
            "significant_5pct": bool(p_value < 0.05),
            "sharpe_outperformance_pct": float(outperf),
        }

    # ── Save ──────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # strip episode lists for cleaner JSON (keep summary stats)
    save_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            save_results[k] = {kk: vv for kk, vv in v.items() if kk != "episodes"}
        else:
            save_results[k] = v
    with open(out, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n[SAVE] Results saved to: {out}")