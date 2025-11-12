"""Backtest metrics: Sharpe, Sortino (optional), Max Drawdown."""
from __future__ import annotations
import numpy as np
import pandas as pd


def annualization_factor(freq: str) -> float:
    freq = freq.lower()
    if freq in {"d","1d","daily"}:
        return 252.0
    if freq in {"h","1h"}:
        return 252.0 * 6.5  # approx trading hours
    if freq in {"4h"}:
        return 252.0 * (6.5/4.0)
    if freq in {"30m","15m"}:
        return 252.0 * (6.5*60/30.0) if freq=="30m" else 252.0 * (6.5*60/15.0)
    return 252.0


def sharpe(returns: pd.Series, freq: str = "1d", eps: float = 1e-12) -> float:
    r = returns.dropna().values
    if r.size == 0:
        return 0.0
    af = annualization_factor(freq)
    mu = np.mean(r)
    sd = np.std(r, ddof=1) if r.size > 1 else 0.0
    # If variance is (near) zero, floor the denominator to eps so constant
    # positive returns yield a large positive Sharpe instead of 0.
    if sd < eps:
        sd = eps
    return (mu / sd) * np.sqrt(af)


def max_drawdown(equity_curve: pd.Series) -> float:
    ec = equity_curve.dropna().values
    if ec.size == 0:
        return 0.0
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / peak
    return float(np.min(dd))  # negative value




def cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """Conditional Value at Risk (CVaR) at level alpha on returns.
    Interpreted as average tail loss (positive number)."""
    r = returns.dropna().values
    if r.size == 0:
        return 0.0
    losses = -r
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else var
