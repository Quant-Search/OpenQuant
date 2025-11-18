"""RSI momentum strategy: long when RSI crosses above threshold; exit on cross below.
Vectorized with cumulative entry/exit counting to avoid iterative loops.
"""
from __future__ import annotations
import pandas as pd
from ..base import BaseStrategy
from ...features.ta_features import rsi as rsi_feat


class RSICrossStrategy(BaseStrategy):
    def __init__(self, length: int = 14, threshold: float = 50.0):
        if length <= 1:
            raise ValueError("RSI length must be > 1")
        if not (0 < threshold < 100):
            raise ValueError("RSI threshold must be in (0,100)")
        self.length = int(length)
        self.threshold = float(threshold)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = rsi_feat(df, self.length)
        up = (r.shift(1) <= self.threshold) & (r > self.threshold)
        dn = (r.shift(1) >= self.threshold) & (r < self.threshold)
        pos = (up.cumsum() - dn.cumsum()) > 0
        sig = pos.astype(int).reindex(df.index).fillna(0)
        return sig

