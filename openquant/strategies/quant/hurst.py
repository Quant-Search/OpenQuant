import numpy as np
import pandas as pd

from ..base import BaseStrategy


class HurstExponentStrategy(BaseStrategy):
    """Fractal Dimension / Hurst Exponent Strategy.

    Calculates the Hurst Exponent (H) to determine market regime.
    H < 0.5: Mean Reverting (Anti-persistent)
    H > 0.5: Trending (Persistent)

    Strategy:
    - If H > 0.5 (Trend): Use SMA Crossover logic.
    - If H < 0.5 (Mean Revert): Use RSI logic.
    """

    def __init__(self, lookback: int = 100, trend_threshold: float = 0.55, mr_threshold: float = 0.45) -> None:
        self.lookback: int = int(lookback)
        self.trend_threshold: float = float(trend_threshold)
        self.mr_threshold: float = float(mr_threshold)

    def _calculate_hurst(self, series: pd.Series) -> float:
        lags = range(2, 20)
        tau = [np.sqrt(np.std(series.diff(lag).dropna())) for lag in lags]
        try:
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = m[0] * 2.0
            return float(hurst)
        except:
            return 0.5

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=int)

        close = df['Close']
        change = close.diff(self.lookback).abs()
        volatility = close.diff().abs().rolling(window=self.lookback).sum()

        er = change / volatility

        sma = close.rolling(window=20).mean()

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        signals = pd.Series(0, index=df.index, dtype=int)

        trend_cond = er > 0.3
        signals[trend_cond & (close > sma)] = 1
        signals[trend_cond & (close < sma)] = -1

        mr_cond = er < 0.2
        signals[mr_cond & (rsi < 30)] = 1
        signals[mr_cond & (rsi > 70)] = -1

        return signals
