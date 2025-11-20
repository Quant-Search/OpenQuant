"""
Liquidity Provision Strategy (Market Microstructure).
Adapts trading style based on Flow Toxicity (VPIN).
"""
import pandas as pd
import numpy as np
from openquant.strategies.base import BaseStrategy
from openquant.quant.microstructure import vpin, amihud_illiquidity

class LiquidityProvisionStrategy(BaseStrategy):
    """
    Liquidity Provision Strategy.
    
    Logic:
    - Calculate VPIN (Flow Toxicity).
    - If VPIN < Threshold (Low Toxicity): Market Making / Mean Reversion.
      - Assume price moves are noise/uninformed. Fade moves.
    - If VPIN > Threshold (High Toxicity): Momentum / Taking Liquidity.
      - Assume price moves are informed. Follow moves.
      
    Parameters:
    - vpin_window: Buckets for VPIN (default 50).
    - vpin_threshold: Level to switch regime (default 0.5? VPIN is usually around 0.2-0.3 for liquid, spikes to 0.8).
      - Let's use a dynamic threshold or fixed 0.3.
    - lookback: Window for local trend/mean-reversion calculation.
    """
    
    def __init__(self, vpin_window: int = 50, vpin_threshold: float = 0.25, lookback: int = 20):
        self.vpin_window = vpin_window
        self.vpin_threshold = vpin_threshold
        self.lookback = lookback
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.lookback * 2:
            return pd.Series(0, index=df.index)
            
        # 1. Calculate VPIN
        # We need to estimate bucket volume.
        # Heuristic: Total volume / (Rows / 5) -> 5 bars per bucket?
        # Or just use mean volume * 10.
        bucket_vol = df['Volume'].rolling(100).mean().iloc[-1] * 10
        if np.isnan(bucket_vol) or bucket_vol == 0:
            bucket_vol = df['Volume'].mean() * 10
            
        vpin_series = vpin(df, bucket_volume=bucket_vol, window_buckets=self.vpin_window)
        
        # 2. Calculate Signals
        signals = pd.Series(0, index=df.index)
        
        # Price change (Return)
        ret = df['Close'].pct_change()
        
        # Mean Reversion Signal (Fade recent move)
        # If price is above MA, Short. Below MA, Long.
        ma = df['Close'].rolling(self.lookback).mean()
        mr_signal = np.where(df['Close'] > ma, -1, 1)
        
        # Momentum Signal (Follow trend)
        # If Close > MA, Trend is Up -> Long (1)
        # If Close < MA, Trend is Down -> Short (-1)
        mom_signal = np.where(df['Close'] > ma, 1, -1)
        
        # 3. Combine based on VPIN
        # High VPIN -> Toxic -> Informed -> Trend likely to continue -> Momentum
        # Low VPIN -> Benign -> Uninformed -> Noise -> Mean Reversion
        
        is_toxic = vpin_series > self.vpin_threshold
        
        # Vectorized choice
        signals = np.where(is_toxic, mom_signal, mr_signal)
        
        return pd.Series(signals, index=df.index)
