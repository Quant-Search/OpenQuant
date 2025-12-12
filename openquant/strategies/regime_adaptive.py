"""Regime Adaptive Strategy.

Uses RegimeDetector to classify market regime and dynamically routes to
appropriate sub-strategies:
- Trending: Hurst Exponent Strategy
- Mean-reverting: Statistical Arbitrage
- High Volatility: Reduce exposure or flat
"""
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from openquant.strategies.base import BaseStrategy
from openquant.quant.regime_detector import RegimeDetector, RegimeType
from openquant.strategies.quant.hurst import HurstExponentStrategy
from openquant.strategies.quant.stat_arb import StatArbStrategy
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


class RegimeAdaptiveStrategy(BaseStrategy):
    """
    Adaptive strategy that switches between sub-strategies based on market regime.
    
    Uses RegimeDetector to classify the market and routes to:
    - HurstExponentStrategy: For trending markets (H > 0.55)
    - StatArbStrategy: For mean-reverting markets (H < 0.45)
    - Reduced/No exposure: During high volatility periods
    
    Args:
        lookback: Lookback period for regime detection (default: 100)
        hurst_threshold_trend: Hurst threshold for trending regime (default: 0.55)
        hurst_threshold_mr: Hurst threshold for mean-reverting regime (default: 0.45)
        vol_reduce_factor: Factor to reduce exposure during high volatility (default: 0.5)
        enable_vol_scaling: Enable volatility-based position scaling (default: True)
        hurst_params: Parameters for HurstExponentStrategy (dict)
        stat_arb_params: Parameters for StatArbStrategy (dict)
        pair_df: DataFrame for StatArb pairs trading (optional)
    """
    
    def __init__(
        self,
        lookback: int = 100,
        hurst_threshold_trend: float = 0.55,
        hurst_threshold_mr: float = 0.45,
        vol_reduce_factor: float = 0.5,
        enable_vol_scaling: bool = True,
        hurst_params: Optional[Dict[str, Any]] = None,
        stat_arb_params: Optional[Dict[str, Any]] = None,
        pair_df: Optional[pd.DataFrame] = None
    ):
        super().__init__()
        self.lookback = int(lookback)
        self.hurst_threshold_trend = float(hurst_threshold_trend)
        self.hurst_threshold_mr = float(hurst_threshold_mr)
        self.vol_reduce_factor = float(vol_reduce_factor)
        self.enable_vol_scaling = enable_vol_scaling
        self.pair_df = pair_df
        
        self.regime_detector = RegimeDetector(lookback=lookback)
        
        if hurst_params is None:
            hurst_params = {"lookback": lookback}
        if stat_arb_params is None:
            stat_arb_params = {"lookback": lookback}
            
        self.hurst_strategy = HurstExponentStrategy(**hurst_params)
        self.stat_arb_strategy = StatArbStrategy(**stat_arb_params)
        
        self.regime_history = []
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals by detecting regime and routing to appropriate strategy.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Series of signals {-1, 0, 1}
        """
        if df.empty or len(df) < self.lookback:
            return pd.Series(0, index=df.index)
            
        signals = pd.Series(0, index=df.index)
        
        for i in range(self.lookback, len(df)):
            df_window = df.iloc[:i+1]
            
            regime_info = self.regime_detector.detect_regime(df_window)
            
            self.regime_history.append({
                'timestamp': df.index[i],
                'trend_regime': regime_info['trend_regime'].value,
                'volatility_regime': regime_info['volatility_regime'].value,
                'hurst_exponent': regime_info['hurst_exponent'],
                'volatility': regime_info['volatility']
            })
            
            trend_regime = regime_info['trend_regime']
            vol_regime = regime_info['volatility_regime']
            hurst = regime_info['hurst_exponent']
            
            sub_signal = 0
            
            if vol_regime == RegimeType.HIGH_VOLATILITY:
                if self.enable_vol_scaling:
                    if hurst > self.hurst_threshold_trend:
                        sub_signals = self.hurst_strategy.generate_signals(df_window)
                        if not sub_signals.empty:
                            sub_signal = int(sub_signals.iloc[-1] * self.vol_reduce_factor)
                    elif hurst < self.hurst_threshold_mr:
                        if self.pair_df is not None:
                            pair_window = self.pair_df.iloc[:i+1]
                            sub_signals = self.stat_arb_strategy.generate_signals(
                                df_window, pair_df=pair_window
                            )
                        else:
                            sub_signals = self.stat_arb_strategy.generate_signals(df_window)
                        if not sub_signals.empty:
                            sub_signal = int(sub_signals.iloc[-1] * self.vol_reduce_factor)
                else:
                    sub_signal = 0
                    
            elif hurst > self.hurst_threshold_trend:
                sub_signals = self.hurst_strategy.generate_signals(df_window)
                if not sub_signals.empty:
                    sub_signal = int(sub_signals.iloc[-1])
                    
            elif hurst < self.hurst_threshold_mr:
                if self.pair_df is not None:
                    pair_window = self.pair_df.iloc[:i+1]
                    sub_signals = self.stat_arb_strategy.generate_signals(
                        df_window, pair_df=pair_window
                    )
                else:
                    sub_signals = self.stat_arb_strategy.generate_signals(df_window)
                if not sub_signals.empty:
                    sub_signal = int(sub_signals.iloc[-1])
            else:
                sub_signal = 0
                
            signals.iloc[i] = sub_signal
            
        return signals
    
    def get_regime_history(self) -> pd.DataFrame:
        """
        Get the history of detected regimes.
        
        Returns:
            DataFrame with regime history
        """
        if not self.regime_history:
            return pd.DataFrame()
        return pd.DataFrame(self.regime_history).set_index('timestamp')
    
    def get_regime_stats(self) -> Dict[str, Any]:
        """
        Get statistics about regime distribution.
        
        Returns:
            Dictionary with regime statistics
        """
        if not self.regime_history:
            return {}
            
        df = pd.DataFrame(self.regime_history)
        
        trend_counts = df['trend_regime'].value_counts().to_dict()
        vol_counts = df['volatility_regime'].value_counts().to_dict()
        
        return {
            'trend_regime_distribution': trend_counts,
            'volatility_regime_distribution': vol_counts,
            'mean_hurst_exponent': float(df['hurst_exponent'].mean()),
            'std_hurst_exponent': float(df['hurst_exponent'].std()),
            'mean_volatility': float(df['volatility'].mean()),
            'std_volatility': float(df['volatility'].std())
        }
