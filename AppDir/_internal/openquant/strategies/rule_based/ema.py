"""Momentum strategy: EMA crossover long/flat.
- Long when fast EMA > slow EMA, flat otherwise.
"""
from __future__ import annotations
import pandas as pd
from ..base import BaseStrategy
from ...features.ta_features import ema


class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, fast: int = 10, slow: int = 30):
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("Require 0 < fast < slow")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        f = ema(df, self.fast)
        s = ema(df, self.slow)
        sig = (f > s).astype(int)
        return sig.reindex(df.index).fillna(0).astype(int)

