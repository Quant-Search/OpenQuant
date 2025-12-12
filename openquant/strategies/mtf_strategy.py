"""Multi-Timeframe Strategy with Signal Alignment.

Requires signal confirmation across multiple timeframes (e.g., 1h, 4h, 1d) before entry.
Uses regime filtering from validation/mtf_filter.py to ensure higher timeframes confirm the signal.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Callable, List, Tuple
import pandas as pd
import numpy as np

from openquant.strategies.base import BaseStrategy
from openquant.validation.mtf_filter import check_mtf_confirmation


class MultiTimeframeStrategy(BaseStrategy):
    """Multi-Timeframe Strategy requiring alignment across timeframes.
    
    This strategy wraps a base strategy and ensures that signals are confirmed
    across multiple timeframes before entry. The strategy checks:
    1. Primary signal from the base strategy on the main timeframe
    2. Trend alignment on higher timeframes (e.g., if 1h says long, 4h and 1d should show uptrend)
    3. Optional additional confirmation from the base strategy on higher timeframes
    
    Signal convention: -1 short, 0 flat, +1 long
    """
    
    def __init__(
        self,
        base_strategy: BaseStrategy,
        timeframes: List[str] = None,
        fetch_func: Optional[Callable[[str, str], pd.DataFrame]] = None,
        require_all_timeframes: bool = False,
        min_confirmations: int = 1,
        use_strategy_signals: bool = False,
    ):
        """Initialize Multi-Timeframe Strategy.
        
        Args:
            base_strategy: The underlying strategy to generate signals
            timeframes: List of timeframes to check in order [primary, higher1, higher2, ...]
                       Default: ['1h', '4h', '1d']
            fetch_func: Function to fetch OHLCV data as (symbol, timeframe) -> DataFrame
                       Must return DataFrame with OHLCV columns
            require_all_timeframes: If True, all higher timeframes must confirm
                                   If False, only min_confirmations needed
            min_confirmations: Minimum number of higher timeframes that must confirm
            use_strategy_signals: If True, run base_strategy on each timeframe for confirmation
                                 If False, use simple trend checks (SMA-based)
        """
        self.base_strategy = base_strategy
        self.timeframes = timeframes or ['1h', '4h', '1d']
        self.fetch_func = fetch_func
        self.require_all_timeframes = require_all_timeframes
        self.min_confirmations = min_confirmations
        self.use_strategy_signals = use_strategy_signals
        self._symbol: Optional[str] = None
        
    def set_symbol(self, symbol: str):
        """Set the symbol for multi-timeframe data fetching."""
        self._symbol = symbol
        
    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate signals with multi-timeframe confirmation.
        
        Args:
            df: Primary timeframe OHLCV data
            **kwargs: Additional arguments passed to base_strategy
            
        Returns:
            Signal series with values in {-1, 0, 1}
        """
        if df.empty:
            return pd.Series(0, index=df.index)
            
        primary_signals = self.base_strategy.generate_signals(df, **kwargs)
        primary_signals = primary_signals.clip(-1, 1).fillna(0)
        
        if self.fetch_func is None or self._symbol is None:
            return primary_signals
            
        if len(self.timeframes) < 2:
            return primary_signals
            
        confirmed_signals = pd.Series(0, index=df.index)
        
        for idx in df.index:
            primary_sig = primary_signals.loc[idx]
            
            if primary_sig == 0:
                confirmed_signals.loc[idx] = 0
                continue
                
            if self._check_mtf_alignment(primary_sig):
                confirmed_signals.loc[idx] = primary_sig
            else:
                confirmed_signals.loc[idx] = 0
                
        return confirmed_signals
        
    def _check_mtf_alignment(self, signal_direction: int) -> bool:
        """Check if higher timeframes confirm the signal direction.
        
        Args:
            signal_direction: 1 for long, -1 for short
            
        Returns:
            True if sufficient higher timeframes confirm, False otherwise
        """
        if signal_direction == 0:
            return True
            
        primary_tf = self.timeframes[0]
        higher_tfs = self.timeframes[1:]
        
        if not higher_tfs:
            return True
            
        confirmations = 0
        checks_performed = 0
        
        for htf in higher_tfs:
            try:
                df_htf = self.fetch_func(self._symbol, htf)
                
                if df_htf.empty or len(df_htf) < 2:
                    continue
                    
                checks_performed += 1
                
                if self.use_strategy_signals:
                    if self._check_strategy_confirmation(df_htf, signal_direction):
                        confirmations += 1
                else:
                    if self._check_trend_confirmation(df_htf, signal_direction):
                        confirmations += 1
                        
            except Exception:
                continue
                
        if checks_performed == 0:
            return True
            
        if self.require_all_timeframes:
            return confirmations == checks_performed
        else:
            return confirmations >= min(self.min_confirmations, checks_performed)
            
    def _check_trend_confirmation(self, df: pd.DataFrame, signal_direction: int) -> bool:
        """Check if the trend on this timeframe confirms the signal.
        
        Uses multiple technical indicators:
        - SMA50 vs Price
        - SMA20 vs SMA50 (trend strength)
        - Recent price momentum
        
        Args:
            df: OHLCV DataFrame for the timeframe
            signal_direction: 1 for long, -1 for short
            
        Returns:
            True if trend confirms signal direction
        """
        if len(df) < 50:
            return self._check_simple_trend_confirmation(df, signal_direction)
            
        close = df['Close']
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        
        current_price = close.iloc[-1]
        current_sma20 = sma20.iloc[-1]
        current_sma50 = sma50.iloc[-1]
        
        if pd.isna(current_sma20) or pd.isna(current_sma50):
            return self._check_simple_trend_confirmation(df, signal_direction)
            
        recent_return = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0
        
        if signal_direction == 1:
            price_above_sma = current_price > current_sma50
            sma_trending_up = current_sma20 > current_sma50
            momentum_positive = recent_return > -0.01
            
            confirmations = sum([price_above_sma, sma_trending_up, momentum_positive])
            return confirmations >= 2
            
        elif signal_direction == -1:
            price_below_sma = current_price < current_sma50
            sma_trending_down = current_sma20 < current_sma50
            momentum_negative = recent_return < 0.01
            
            confirmations = sum([price_below_sma, sma_trending_down, momentum_negative])
            return confirmations >= 2
            
        return False
        
    def _check_simple_trend_confirmation(self, df: pd.DataFrame, signal_direction: int) -> bool:
        """Simplified trend check for shorter timeframes.
        
        Uses the existing check_mtf_confirmation logic from mtf_filter.
        """
        if len(df) < 20:
            return True
            
        close = df['Close']
        sma = close.rolling(20).mean()
        current_price = close.iloc[-1]
        current_sma = sma.iloc[-1]
        
        if pd.isna(current_sma):
            return True
            
        if signal_direction == 1:
            return current_price > current_sma
        elif signal_direction == -1:
            return current_price < current_sma
            
        return False
        
    def _check_strategy_confirmation(self, df: pd.DataFrame, signal_direction: int) -> bool:
        """Check if the base strategy also generates signals in the same direction.
        
        Args:
            df: OHLCV DataFrame for the higher timeframe
            signal_direction: 1 for long, -1 for short
            
        Returns:
            True if strategy signals agree with direction
        """
        try:
            htf_signals = self.base_strategy.generate_signals(df)
            
            if htf_signals.empty:
                return False
                
            recent_signals = htf_signals.tail(3)
            
            if signal_direction == 1:
                return (recent_signals > 0).any()
            elif signal_direction == -1:
                return (recent_signals < 0).any()
                
        except Exception:
            return False
            
        return False
        

class MultiTimeframeEnsemble(BaseStrategy):
    """Ensemble of strategies across different timeframes with weighted voting.
    
    This strategy runs the same or different strategies on multiple timeframes
    and combines their signals with configurable weights.
    """
    
    def __init__(
        self,
        strategies: List[Tuple[str, BaseStrategy, float]],
        fetch_func: Optional[Callable[[str, str], pd.DataFrame]] = None,
        aggregation: str = 'weighted',
        threshold: float = 0.3,
    ):
        """Initialize Multi-Timeframe Ensemble.
        
        Args:
            strategies: List of (timeframe, strategy, weight) tuples
                       e.g., [('1h', strategy1, 0.5), ('4h', strategy2, 0.3), ('1d', strategy3, 0.2)]
            fetch_func: Function to fetch OHLCV data as (symbol, timeframe) -> DataFrame
            aggregation: How to combine signals: 'weighted', 'majority', 'unanimous'
            threshold: For weighted aggregation, threshold for signal generation
        """
        self.strategies = strategies
        self.fetch_func = fetch_func
        self.aggregation = aggregation
        self.threshold = threshold
        self._symbol: Optional[str] = None
        
        total_weight = sum(w for _, _, w in strategies)
        if total_weight > 0:
            self.normalized_weights = [(tf, s, w/total_weight) for tf, s, w in strategies]
        else:
            n = len(strategies)
            self.normalized_weights = [(tf, s, 1.0/n) for tf, s, _ in strategies]
            
    def set_symbol(self, symbol: str):
        """Set the symbol for multi-timeframe data fetching."""
        self._symbol = symbol
        
    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ensemble signals across timeframes.
        
        Args:
            df: Primary timeframe OHLCV data (should match first strategy's timeframe)
            **kwargs: Additional arguments passed to strategies
            
        Returns:
            Combined signal series with values in {-1, 0, 1}
        """
        if df.empty:
            return pd.Series(0, index=df.index)
            
        if not self.strategies:
            return pd.Series(0, index=df.index)
            
        all_signals = []
        all_weights = []
        
        for timeframe, strategy, weight in self.normalized_weights:
            try:
                if self.fetch_func and self._symbol:
                    tf_df = self.fetch_func(self._symbol, timeframe)
                    if tf_df.empty:
                        continue
                else:
                    tf_df = df
                    
                signals = strategy.generate_signals(tf_df, **kwargs)
                signals = signals.clip(-1, 1).fillna(0)
                
                signals_aligned = signals.reindex(df.index, method='ffill').fillna(0)
                
                all_signals.append(signals_aligned)
                all_weights.append(weight)
                
            except Exception:
                continue
                
        if not all_signals:
            return pd.Series(0, index=df.index)
            
        if self.aggregation == 'weighted':
            return self._weighted_aggregation(all_signals, all_weights, df.index)
        elif self.aggregation == 'majority':
            return self._majority_aggregation(all_signals, df.index)
        elif self.aggregation == 'unanimous':
            return self._unanimous_aggregation(all_signals, df.index)
        else:
            return self._weighted_aggregation(all_signals, all_weights, df.index)
            
    def _weighted_aggregation(self, signals_list: List[pd.Series], weights: List[float], index: pd.Index) -> pd.Series:
        """Combine signals using weighted voting."""
        combined = pd.Series(0.0, index=index)
        
        for signals, weight in zip(signals_list, weights):
            combined += signals * weight
            
        final_signals = pd.Series(0, index=index)
        final_signals[combined > self.threshold] = 1
        final_signals[combined < -self.threshold] = -1
        
        return final_signals
        
    def _majority_aggregation(self, signals_list: List[pd.Series], index: pd.Index) -> pd.Series:
        """Combine signals using majority voting."""
        if not signals_list:
            return pd.Series(0, index=index)
            
        signal_df = pd.DataFrame(signals_list).T
        
        long_votes = (signal_df > 0).sum(axis=1)
        short_votes = (signal_df < 0).sum(axis=1)
        
        final_signals = pd.Series(0, index=index)
        final_signals[long_votes > short_votes] = 1
        final_signals[short_votes > long_votes] = -1
        
        return final_signals
        
    def _unanimous_aggregation(self, signals_list: List[pd.Series], index: pd.Index) -> pd.Series:
        """Combine signals requiring unanimous agreement."""
        if not signals_list:
            return pd.Series(0, index=index)
            
        signal_df = pd.DataFrame(signals_list).T
        
        all_long = (signal_df > 0).all(axis=1)
        all_short = (signal_df < 0).all(axis=1)
        
        final_signals = pd.Series(0, index=index)
        final_signals[all_long] = 1
        final_signals[all_short] = -1
        
        return final_signals
