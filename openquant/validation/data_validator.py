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
        
    # Sanity Check: Volume
    # If average volume is very low (e.g. < $1000 or < 100 units), it's risky.
    # We don't know the price unit here easily without multiplying, but we can check raw volume.
    # Let's assume crypto/forex context where volume should be substantial.
    # Warning only, as some assets might be low volume but valid.
    if df['Volume'].mean() < 1.0:
        issues.append("low_volume_warning")
    
    # Spike Filter: Detect unrealistic price movements
    # Calculate percent change for each bar
    pct_change = df['Close'].pct_change().abs()
    # If any single bar moves > 50%, it's likely a data error (or flash crash)
    # We use 50% as a very generous threshold; most real moves are < 20%
    if (pct_change > 0.5).any():
        issues.append("price_spike_detected")
        
    return issues


def is_valid_ohlcv(df: pd.DataFrame) -> bool:
    return len(validate_ohlcv(df)) == 0

