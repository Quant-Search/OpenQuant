"""Strategy signals validation utilities.

Checks implemented:
- Signal index matches input df index
- No NaN/inf values
- Values restricted to {-1, 0, 1}
- Reasonable turnover: consecutive differences in {-2,-1,0,1,2}
"""
from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np


def validate_signals(df: pd.DataFrame, sig: pd.Series) -> List[str]:
    issues: List[str] = []
    if not sig.index.equals(df.index):
        issues.append("index_mismatch")
    v = sig.astype(float)
    if not np.isfinite(v).all():
        issues.append("non_finite_values")
    uniq = set(sig.dropna().unique())
    if not uniq.issubset({-1, 0, 1}):
        issues.append("invalid_values")
    d = sig.diff().fillna(0)
    if not set(d.unique()).issubset({-2, -1, 0, 1, 2}):
        issues.append("excessive_turnover")
    return issues


def is_valid_signals(df: pd.DataFrame, sig: pd.Series) -> bool:
    return len(validate_signals(df, sig)) == 0

