"""MACD crossover strategy: long when MACD crosses above signal; exit on cross below.
Vectorized with cumulative entry/exit counting.
"""
from __future__ import annotations
import pandas as pd
from ..base import BaseStrategy
from ...features.ta_features import macd_features


class MACDCrossoverStrategy(BaseStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        if fast <= 0 or slow <= 0 or signal <= 0 or fast >= slow:
            raise ValueError("Require 0 < fast < slow and signal > 0")
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        feats = macd_features(df, fast=self.fast, slow=self.slow, signal=self.signal)
        macd = feats["macd"]
        sigl = feats["signal"]
        up = (macd.shift(1) <= sigl.shift(1)) & (macd > sigl)
        dn = (macd.shift(1) >= sigl.shift(1)) & (macd < sigl)
        pos = (up.cumsum() - dn.cumsum()) > 0
        return pos.astype(int).reindex(df.index).fillna(0)

