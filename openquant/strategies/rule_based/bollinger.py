"""Bollinger Bands mean-reversion: enter long on cross below lower, exit on cross above mid.
Vectorized state via cumulative entries/exits.
"""
from __future__ import annotations
import pandas as pd
from ..base import BaseStrategy
from ...features.ta_features import bollinger_bands


class BollingerMeanReversionStrategy(BaseStrategy):
    def __init__(self, length: int = 20, k: float = 2.0):
        if length <= 1:
            raise ValueError("length must be > 1")
        if k <= 0:
            raise ValueError("k must be > 0")
        self.length = int(length)
        self.k = float(k)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        bb = bollinger_bands(df, length=self.length, k=self.k)
        px = df["Close"].astype(float)
        lower, mid = bb["lower"], bb["mid"]
        # entry when price crosses below lower band; exit when crosses above mid
        entry = (px.shift(1) >= lower.shift(1)) & (px < lower)
        exit_ = (px.shift(1) <= mid.shift(1)) & (px > mid)
        pos = (entry.cumsum() - exit_.cumsum()) > 0
        return pos.astype(int).reindex(df.index).fillna(0)

