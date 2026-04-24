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

    def run_episode(self, agent, start_idx: int = 0) -> dict:
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
        T = self.T_days / 252

        # Initial option premium
        premium = bs_call(S0, self.K, T, self.r, self.sigma)
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
                g = greeks_call(price, self.K, T_remaining, self.r, realized_vol)
                delta = g.delta
                gamma = g.gamma
            except:
                delta = 0.5
                gamma = 0.01

            # Build observation
            pnl_norm = (portfolio_value - premium) / max(premium, 1e-6)
            obs = np.array([
                price / S0,
                self.K / S0,
                T_remaining / T,
                realized_vol,
                delta,
                gamma,
                np.clip(pnl_norm, -10, 10),
            ], dtype=np.float32)

            # Get action
            action, _ = agent.predict(obs)
            target_hedge = float(np.clip(action[0], -1.0, 1.0))

            # Transaction cost
            tc = self.tc_rate * abs(target_hedge - hedge_position) * price
            portfolio_value -= tc

            # Update position
            old_hedge = hedge_position
            hedge_position = target_hedge

            # Next price
            if step + 1 < self.T_days:
                next_price = self.prices[start_idx + step + 1]
            else:
                next_price = price

            # PnL
            price_change = next_price - price
            hedge_pnl = hedge_position * price_change

            T_next = max((self.T_days - step - 1) / 252, 1e-10)
            old_opt = bs_call(price, self.K, T_remaining, self.r, realized_vol)
            new_opt = bs_call(next_price, self.K, T_next, self.r, realized_vol)
            option_pnl = -(new_opt - old_opt)

            total_pnl = hedge_pnl + option_pnl
            portfolio_value += total_pnl
            pnl_history.append(total_pnl)
            hedge_history.append(target_hedge)
            price_history.append(next_price)

        # Final settlement
        final_price = self.prices[start_idx + self.T_days]
        intrinsic = max(final_price - self.K, 0.0)
        portfolio_value -= intrinsic

        pnls = np.array(pnl_history)

        return {
            "portfolio_value": portfolio_value,
            "total_pnl": float(np.sum(pnls)),
            "premium": premium,
            "pnl_history": pnl_history,
            "hedge_history": hedge_history,
            "price_history": price_history,
            "sharpe": float(
                np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)
            ),
            "max_drawdown": float(self._max_drawdown(pnls)),
            "transaction_costs": float(
                sum(self.tc_rate * abs(hedge_history[i] - (
                    hedge_history[i-1] if i > 0 else 0
                )) * price_history[i] for i in range(len(hedge_history)))
            ),
        }

    def run_backtest(self, agent, stride: int = 5) -> dict:
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
                result = self.run_episode(agent, start_idx=start)
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
            "sharpe_ratio": float(
                np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)
            ),
            "mean_max_drawdown": float(np.mean(drawdowns)),
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
    print("Fetching SPY data...")
    hist = fetch_spy_data("6mo")
    prices = hist["Close"].values

    print(f"\nRunning backtest with DeltaHedger...")
    from environment.baselines import DeltaHedger

    K = float(np.mean(prices))  # ATM strike
    bt = HistoricalBacktester(prices, K=K, sigma=0.2)
    agent = DeltaHedger(K=K)

    results = bt.run_backtest(agent, stride=5)
    print(f"\nResults ({results['n_episodes']} episodes):")
    print(f"  Sharpe: {results['sharpe_ratio']:.4f}")
    print(f"  Mean PnL: {results['mean_pnl']:.4f}")
    print(f"  Max Drawdown: {results['max_max_drawdown']:.4f}")
