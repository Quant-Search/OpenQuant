"""Backtest metrics: Sharpe, Sortino, Max Drawdown, CVaR, Monte Carlo."""
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


def sortino(returns: pd.Series, freq: str = "1d", eps: float = 1e-12) -> float:
    r = returns.dropna().values
    if r.size == 0:
        return 0.0
    af = annualization_factor(freq)
    downside = r[r < 0]
    sd = np.std(downside, ddof=1) if downside.size > 1 else 0.0
    if sd < eps:
        sd = eps
    mu = np.mean(r)
    return (mu / sd) * np.sqrt(af)


def win_rate(returns: pd.Series) -> float:
    r = returns.dropna().values
    if r.size == 0:
        return 0.0
    wins = (r > 0).sum()
    return float(wins) / float(r.size)


def profit_factor(returns: pd.Series) -> float:
    r = returns.dropna().values
    if r.size == 0:
        return 0.0
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains) / float(losses) if losses > 0 else float("inf")


def monte_carlo_bootstrap(returns: pd.Series, n: int = 500, block: int = 10, freq: str = "1d") -> dict:
    r = returns.dropna().values
    if r.size == 0:
        return {"sharpe": [], "max_dd": [], "sortino": []}
    out_sharpe = []
    out_dd = []
    out_sortino = []
    for _ in range(max(1, n)):
        idx = np.random.randint(0, r.size, size=r.size)
        # Optional block bootstrap
        if block > 1:
            for i in range(0, r.size, block):
                if i + block <= r.size:
                    start = np.random.randint(0, r.size - block + 1)
                    idx[i:i+block] = np.arange(start, start+block)
        rb = r[idx]
        ec = (1.0 + pd.Series(rb)).cumprod()
        out_sharpe.append(sharpe(pd.Series(rb), freq=freq))
        out_sortino.append(sortino(pd.Series(rb), freq=freq))
        out_dd.append(abs(max_drawdown(ec)))
    return {"sharpe": out_sharpe, "sortino": out_sortino, "max_dd": out_dd}
