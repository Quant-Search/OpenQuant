"""OHLCV data validation utilities.

Checks implemented:
- Non-negative prices and volume
- High >= max(Open, Close), Low <= min(Open, Close)
- Monotonic, unique datetime index
- Sufficient rows (>= 10) for basic indicators
"""
from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np


def validate_ohlcv(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []
    req = {"Open", "High", "Low", "Close", "Volume"}
    if not req.issubset(df.columns):
        issues.append("missing_columns")
        return issues
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("index_not_datetime")
    if df.index.has_duplicates:
        issues.append("duplicate_index")
    if not df.index.is_monotonic_increasing:
        issues.append("non_monotonic_index")
    # Non-negative values
    vals = df[["Open", "High", "Low", "Close", "Volume"]]
    if (vals < 0).any().any():
        issues.append("negative_values")
    # OHLC relationships
    if ((df["High"] < df[["Open", "Close"]].max(axis=1))).any():
        issues.append("high_below_open_close")
    if ((df["Low"] > df[["Open", "Close"]].min(axis=1))).any():
        issues.append("low_above_open_close")
    # NaNs excessive
    if vals.isna().mean().mean() > 0.1:
        issues.append("too_many_nans")
    # Size
    if len(df) < 10:
        issues.append("too_few_rows")
    return issues


def is_valid_ohlcv(df: pd.DataFrame) -> bool:
    return len(validate_ohlcv(df)) == 0

