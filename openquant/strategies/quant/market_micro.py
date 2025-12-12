"""
Liquidity Provision Strategy (Market Microstructure).
Adapts trading style based on Flow Toxicity (VPIN).
"""
import numpy as np
import pandas as pd

from openquant.quant.microstructure import vpin
from openquant.strategies.base import BaseStrategy
from openquant.utils.validation import (
    validate_params,
    validate_positive_int_param,
    validate_range_param
)

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
    
    @validate_params(
        vpin_window=validate_positive_int_param('vpin_window'),
        vpin_threshold=validate_range_param('vpin_threshold', min_val=0.0, max_val=1.0),
        lookback=validate_positive_int_param('lookback')
    )
    def __init__(self, vpin_window: int = 50, vpin_threshold: float = 0.25, lookback: int = 20) -> None:
        self.vpin_window: int = vpin_window
        self.vpin_threshold: float = vpin_threshold
        self.lookback: int = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if len(df) < self.lookback * 2:
            return pd.Series(0, index=df.index, dtype=int)

        bucket_vol = df['Volume'].rolling(100).mean().iloc[-1] * 10
        if np.isnan(bucket_vol) or bucket_vol == 0:
            bucket_vol = df['Volume'].mean() * 10

        vpin_series = vpin(df, bucket_volume=bucket_vol, window_buckets=self.vpin_window)

        pd.Series(0, index=df.index, dtype=int)

        df['Close'].pct_change()

        ma = df['Close'].rolling(self.lookback).mean()
        mr_signal = np.where(df['Close'] > ma, -1, 1)

        mom_signal = np.where(df['Close'] > ma, 1, -1)

        is_toxic = vpin_series > self.vpin_threshold

        signals_arr = np.where(is_toxic, mom_signal, mr_signal)

        return pd.Series(signals_arr, index=df.index, dtype=int)
