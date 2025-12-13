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
from openquant.utils.validation import (
    validate_params,
    validate_positive_int_param,
    validate_range_param,
    validate_positive_param
)

LOGGER = get_logger(__name__)


class RegimeAdaptiveStrategy(BaseStrategy):
    """
    Adaptive strategy that switches between sub-strategies based on market regime.
    
    Uses RegimeDetector to classify the market and routes to:
    - HurstExponentStrategy: For trending markets (H > hurst_threshold_trend)
    - StatArbStrategy: For mean-reverting markets (H < hurst_threshold_mr)
    - Reduced/No exposure: During high volatility periods
    
    The strategy continuously monitors market conditions and adapts its approach
    based on the detected regime, providing regime-aware position sizing.
    
    Parameters
    ----------
    lookback : int, optional
        Lookback period for regime detection. Default: 100
    hurst_threshold_trend : float, optional
        Hurst threshold for trending regime (H > threshold triggers trend strategy).
        Default: 0.55
    hurst_threshold_mr : float, optional
        Hurst threshold for mean-reverting regime (H < threshold triggers MR strategy).
        Default: 0.45
    vol_reduce_factor : float, optional
        Factor to reduce exposure during high volatility (0.0 to 1.0).
        Default: 0.5 (50% reduction)
    enable_vol_scaling : bool, optional
        Enable volatility-based position scaling. If False, no positions during
        high volatility. Default: True
    hurst_params : dict, optional
        Parameters for HurstExponentStrategy. If None, uses default with lookback.
    stat_arb_params : dict, optional
        Parameters for StatArbStrategy. If None, uses default with lookback.
    pair_df : pd.DataFrame, optional
        DataFrame for StatArb pairs trading mode. If provided, StatArb will
        use pairs trading instead of single-asset mean reversion.
    
    Attributes
    ----------
    regime_detector : RegimeDetector
        Instance for detecting market regimes
    hurst_strategy : HurstExponentStrategy
        Sub-strategy for trending regimes
    stat_arb_strategy : StatArbStrategy
        Sub-strategy for mean-reverting regimes
    regime_history : list
        Historical record of regime classifications and metrics
    
    Examples
    --------
    >>> # Basic usage with default parameters
    >>> strategy = RegimeAdaptiveStrategy(lookback=100)
    >>> signals = strategy.generate_signals(df)
    
    >>> # Custom volatility handling
    >>> strategy = RegimeAdaptiveStrategy(
    ...     lookback=100,
    ...     vol_reduce_factor=0.3,
    ...     enable_vol_scaling=True
    ... )
    
    >>> # With custom sub-strategy parameters
    >>> strategy = RegimeAdaptiveStrategy(
    ...     lookback=100,
    ...     hurst_params={'lookback': 80, 'trend_threshold': 0.6},
    ...     stat_arb_params={'lookback': 80, 'entry_z': 2.5}
    ... )
    
    >>> # Pairs trading mode
    >>> strategy = RegimeAdaptiveStrategy(
    ...     lookback=100,
    ...     stat_arb_params={'pair_symbol': 'SPY'},
    ...     pair_df=spy_df
    ... )
    
    Notes
    -----
    The strategy performs regime classification on a rolling basis, allowing it to
    adapt to changing market conditions. Position sizing is automatically adjusted
    based on the detected volatility regime.
    
    See Also
    --------
    openquant.quant.regime_detector.RegimeDetector : Regime detection implementation
    openquant.strategies.quant.hurst.HurstExponentStrategy : Trending strategy
    openquant.strategies.quant.stat_arb.StatArbStrategy : Mean-reversion strategy
    """
    
    @validate_params(
        lookback=validate_positive_int_param('lookback'),
        hurst_threshold_trend=validate_range_param('hurst_threshold_trend', min_val=0.0, max_val=1.0),
        hurst_threshold_mr=validate_range_param('hurst_threshold_mr', min_val=0.0, max_val=1.0),
        vol_reduce_factor=validate_range_param('vol_reduce_factor', min_val=0.0, max_val=1.0)
    )
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
        
        The method performs the following steps for each time period:
        1. Detect current market regime using RegimeDetector
        2. Select appropriate sub-strategy based on regime
        3. Apply volatility-based position scaling if in high volatility regime
        4. Record regime information for analysis
        
        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with at least 'Close' column. Must have sufficient
            history (>= lookback) for regime detection.
        
        Returns
        -------
        pd.Series
            Series of signals {-1, 0, 1} with same index as df:
            - 1: Long position
            - 0: Flat (no position)
            - -1: Short position
        
        Examples
        --------
        >>> strategy = RegimeAdaptiveStrategy(lookback=50)
        >>> signals = strategy.generate_signals(df)
        >>> print(f"Long signals: {(signals == 1).sum()}")
        >>> print(f"Short signals: {(signals == -1).sum()}")
        >>> print(f"Flat periods: {(signals == 0).sum()}")
        
        Notes
        -----
        The strategy requires at least `lookback` periods of data before generating
        signals. Earlier periods will have signal = 0.
        
        If sub-strategy signal generation fails, the method logs the error and
        returns signal = 0 for that period.
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
                        try:
                            sub_signals = self.hurst_strategy.generate_signals(df_window)
                            if not sub_signals.empty:
                                raw_signal = sub_signals.iloc[-1]
                                sub_signal = int(np.sign(raw_signal) * abs(raw_signal) * self.vol_reduce_factor)
                        except Exception as e:
                            LOGGER.debug(f"Hurst strategy failed in high vol: {e}")
                            sub_signal = 0
                    elif hurst < self.hurst_threshold_mr:
                        try:
                            if self.pair_df is not None:
                                pair_window = self.pair_df.iloc[:i+1]
                                sub_signals = self.stat_arb_strategy.generate_signals(
                                    df_window, pair_df=pair_window
                                )
                            else:
                                sub_signals = self.stat_arb_strategy.generate_signals(df_window)
                            if not sub_signals.empty:
                                raw_signal = sub_signals.iloc[-1]
                                sub_signal = int(np.sign(raw_signal) * abs(raw_signal) * self.vol_reduce_factor)
                        except Exception as e:
                            LOGGER.debug(f"StatArb strategy failed in high vol: {e}")
                            sub_signal = 0
                else:
                    sub_signal = 0
                    
            elif hurst > self.hurst_threshold_trend:
                try:
                    sub_signals = self.hurst_strategy.generate_signals(df_window)
                    if not sub_signals.empty:
                        sub_signal = int(sub_signals.iloc[-1])
                except Exception as e:
                    LOGGER.debug(f"Hurst strategy failed: {e}")
                    sub_signal = 0
                    
            elif hurst < self.hurst_threshold_mr:
                try:
                    if self.pair_df is not None:
                        pair_window = self.pair_df.iloc[:i+1]
                        sub_signals = self.stat_arb_strategy.generate_signals(
                            df_window, pair_df=pair_window
                        )
                    else:
                        sub_signals = self.stat_arb_strategy.generate_signals(df_window)
                    if not sub_signals.empty:
                        sub_signal = int(sub_signals.iloc[-1])
                except Exception as e:
                    LOGGER.debug(f"StatArb strategy failed: {e}")
                    sub_signal = 0
            else:
                sub_signal = 0
                
            signals.iloc[i] = sub_signal
            
        return signals
    
    def get_regime_history(self) -> pd.DataFrame:
        """
        Get the history of detected regimes.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - trend_regime: Trend regime classification
            - volatility_regime: Volatility regime classification
            - hurst_exponent: Calculated Hurst exponent
            - volatility: Measured volatility
            Index is the timestamp of each observation.
        
        Examples
        --------
        >>> strategy = RegimeAdaptiveStrategy(lookback=50)
        >>> signals = strategy.generate_signals(df)
        >>> history = strategy.get_regime_history()
        >>> print(history.tail())
        >>> print(f"Time in trending: {(history['trend_regime'] == 'trending_up').sum()}")
        """
        if not self.regime_history:
            return pd.DataFrame()
        return pd.DataFrame(self.regime_history).set_index('timestamp')
    
    def get_regime_stats(self) -> Dict[str, Any]:
        """
        Get statistics about regime distribution.
        
        Computes summary statistics across all detected regimes, including
        distribution of regime types and average metrics.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - trend_regime_distribution: Count of each trend regime type
            - volatility_regime_distribution: Count of each volatility regime type
            - mean_hurst_exponent: Average Hurst exponent
            - std_hurst_exponent: Standard deviation of Hurst exponent
            - mean_volatility: Average volatility
            - std_volatility: Standard deviation of volatility
        
        Examples
        --------
        >>> strategy = RegimeAdaptiveStrategy(lookback=50)
        >>> signals = strategy.generate_signals(df)
        >>> stats = strategy.get_regime_stats()
        >>> print(f"Mean Hurst: {stats['mean_hurst_exponent']:.3f}")
        >>> print(f"Regime distribution: {stats['trend_regime_distribution']}")
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
    
    def reset_history(self) -> None:
        """
        Reset regime history.
        
        Clears all stored regime history. Useful when running multiple
        backtests or when you want to start fresh.
        
        Examples
        --------
        >>> strategy = RegimeAdaptiveStrategy(lookback=50)
        >>> signals1 = strategy.generate_signals(df1)
        >>> strategy.reset_history()
        >>> signals2 = strategy.generate_signals(df2)
        """
        self.regime_history = []
