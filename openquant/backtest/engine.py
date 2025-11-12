"""Minimal vectorized backtest engine for long/flat signals."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.Series


def backtest_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    fee_bps: float = 1.0,
    weight: float = 1.0,
) -> BacktestResult:
    """Backtest long/flat signals on Close prices with fees in basis points per trade.
    Args:
        df: OHLCV DataFrame with 'Close'.
        signals: Series of {0,1} same index as df.
        fee_bps: Fee per change in position (entry/exit) in basis points.
        weight: Fraction of capital allocated when in position (0..).
    Returns:
        BacktestResult with returns and equity curve (starting at 1.0).
    """
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column")
    # Ensure non-negative weight
    w = max(0.0, float(weight))

    px = df["Close"].astype(float)
    sig = signals.reindex(px.index).fillna(0).astype(int)

    # Compute simple returns; explicitly disable implicit padding to avoid deprecation
    ret = px.pct_change(fill_method=None).fillna(0.0)

    # Position changes incur fees; scale by weight to reflect allocated notional
    pos = sig.shift(1).fillna(0).astype(int)  # enter at next bar open/close approx
    pos_change = pos.diff().abs().fillna(pos.abs())
    fee = pos_change * w * (fee_bps / 10000.0)

    strat_ret = (pos * w) * ret - fee
    equity = (1.0 + strat_ret).cumprod()

    # Approximate trades as position changes
    trades = pos_change

    return BacktestResult(equity_curve=equity, returns=strat_ret, positions=pos, trades=trades)

