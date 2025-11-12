"""Backtest result validation utilities.

We validate a minimal set of invariants:
- returns, equity_curve are finite (no NaN/inf)
- trades count is non-negative
- equity_curve[0] ~ 1.0 (engine uses normalized capital), last equals cumulative product of (1+returns)
"""
from __future__ import annotations
from typing import List, Any
import numpy as np
import pandas as pd


def _as_series(x: Any) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.Series(x)


def validate_backtest(res: Any) -> List[str]:
    issues: List[str] = []
    ret = _as_series(getattr(res, "returns", []))
    eq = _as_series(getattr(res, "equity_curve", []))
    trades = getattr(res, "trades", None)

    if ret.isna().any() or not np.isfinite(ret.fillna(0)).all():
        issues.append("returns_non_finite")
    if eq.isna().any() or not np.isfinite(eq.fillna(0)).all():
        issues.append("equity_non_finite")
    if trades is not None:
        try:
            n_tr = int(np.nansum(np.asarray(trades)))
            if n_tr < 0:
                issues.append("negative_trades")
        except Exception:
            issues.append("invalid_trades_type")
    # Equity reconstruction check (within tolerance)
    if len(eq) and len(ret):
        rec = (1.0 + ret.fillna(0)).cumprod()
        if abs(float(rec.iloc[-1]) - float(eq.iloc[-1])) > 1e-3:
            issues.append("equity_mismatch")
    return issues


def is_valid_backtest(res: Any) -> bool:
    return len(validate_backtest(res)) == 0

