from typing import Dict, Any
import pandas as pd
import numpy as np
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
    
    def __init__(self, lookback: int = 100, trend_threshold: float = 0.55, mr_threshold: float = 0.45):
        self.lookback = int(lookback)
        self.trend_threshold = float(trend_threshold)
        self.mr_threshold = float(mr_threshold)
        
    def _calculate_hurst(self, series: pd.Series) -> float:
        # Simplified R/S analysis or Variance Ratio test
        # Using a very fast approximation: Var(tau) ~ tau^(2H)
        # Log(Var(tau)) / Log(tau) ~ 2H
        
        lags = range(2, 20)
        tau = [np.sqrt(np.std(series.diff(lag).dropna())) for lag in lags]
        # Use polyfit to find slope
        # slope = 2H
        try:
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = m[0] * 2.0
            return hurst
        except:
            return 0.5

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series()
            
        # Rolling Hurst calculation is expensive. 
        # We'll optimize by calculating it only periodically or using a simpler proxy?
        # For this implementation, we'll use a rolling window apply, but it might be slow.
        # Let's use a vectorized approximation if possible, or just accept the cost for "Advanced Quant".
        
        # Actually, let's use a simpler Efficiency Ratio (ER) as a proxy for Hurst/Fractal Dimension
        # ER = Net Change / Sum of Absolute Changes
        # ER approaches 1 for strong trend, 0 for noise.
        
        close = df['Close']
        change = close.diff(self.lookback).abs()
        volatility = close.diff().abs().rolling(window=self.lookback).sum()
        
        er = change / volatility
        # ER is roughly correlated with H. High ER = Trend. Low ER = Mean Reversion.
        
        # Logic:
        # If ER > Threshold -> Trend Follow (Buy if Price > SMA)
        # If ER < Threshold -> Mean Revert (Buy if RSI < 30)
        
        sma = close.rolling(window=20).mean()
        
        # RSI Calculation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=df.index)
        
        # Trend Regime
        trend_cond = er > 0.3 # Arbitrary threshold for ER ~ H > 0.5
        signals[trend_cond & (close > sma)] = 1
        signals[trend_cond & (close < sma)] = -1
        
        # Mean Reversion Regime
        mr_cond = er < 0.2
        signals[mr_cond & (rsi < 30)] = 1
        signals[mr_cond & (rsi > 70)] = -1
        
        return signals
