"""Ensemble Strategy - Combines Multiple Strategies for Diversification.

Combines signals from multiple strategies using voting or weighted average
to reduce variance and improve consistency.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .base import BaseStrategy
from .ml_strategy import MLStrategy
from ..quant.regime_detector import RegimeDetector, RegimeType
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

class CombineMethod(Enum):
    """How to combine signals from multiple strategies."""
    VOTING = "voting"  # Majority vote
    AVERAGE = "average"  # Average weights
    WEIGHTED = "weighted"  # Performance-weighted average
    CONFIDENT = "confident"  # Take most confident signal

@dataclass
class StrategySummary:
    """Summary of a strategy's recent performance."""
    name: str
    win_rate: float
    avg_return: float
    sharpe: float
    weight: float

class EnsembleStrategy(BaseStrategy):
    """
    Ensemble Strategy combining multiple signal sources.
    
    Features:
    - Combines ML, momentum, and mean-reversion strategies
    - Uses voting or weighted averaging
    - Regime-aware: activates different sub-strategies per regime
    - Tracks individual strategy performance for adaptive weighting
    """
    
    def __init__(
        self,
        combine_method: CombineMethod = CombineMethod.VOTING,
        min_agreement: float = 0.4,  # % of strategies that must agree (lowered for more trades)
        use_regime_filter: bool = True,
        probability_threshold: float = 0.52  # Lower threshold for more activity
    ):
        super().__init__()
        self.combine_method = combine_method
        self.min_agreement = min_agreement
        self.use_regime_filter = use_regime_filter
        self.probability_threshold = probability_threshold
        
        # Sub-strategies
        self.ml_strategy = MLStrategy(
            lookback=500,
            probability_threshold=probability_threshold
        )
        
        # Regime detector for adaptive strategy selection
        self.regime_detector = RegimeDetector(lookback=100)
        
        # Performance tracking per strategy
        self.strategy_stats: Dict[str, List[float]] = {
            "ml": [],
            "momentum": [],
            "mean_reversion": []
        }
        
    def _momentum_signal(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Simple momentum strategy: trend following.
        Returns (signal, confidence) where signal is 1 (long), -1 (short), 0 (neutral)
        """
        if len(df) < 50:
            return 0, 0.5
            
        close = df['Close']
        
        # Multi-timeframe momentum
        sma_fast = close.rolling(10).mean().iloc[-1]
        sma_slow = close.rolling(50).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        # Price above both SMAs = bullish
        # Price below both SMAs = bearish
        
        signal = 0
        confidence = 0.5
        
        if current_price > sma_fast > sma_slow:
            # Strong uptrend
            signal = 1
            confidence = 0.65
        elif current_price < sma_fast < sma_slow:
            # Strong downtrend
            signal = -1
            confidence = 0.65
        elif current_price > sma_slow:
            # Weak uptrend
            signal = 1
            confidence = 0.55
        elif current_price < sma_slow:
            # Weak downtrend
            signal = -1
            confidence = 0.55
            
        return signal, confidence
        
    def _mean_reversion_signal(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        Mean reversion strategy: trade back to mean.
        Best in ranging markets (Hurst < 0.5).
        """
        if len(df) < 30:
            return 0, 0.5
            
        close = df['Close']
        
        # Bollinger Bands
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        current_price = close.iloc[-1]
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        sma_val = sma.iloc[-1]
        
        # Calculate position in band (0 = lower, 1 = upper)
        if upper_val == lower_val:
            return 0, 0.5
            
        band_position = (current_price - lower_val) / (upper_val - lower_val)
        
        signal = 0
        confidence = 0.5
        
        if band_position > 0.9:
            # Near upper band - expect reversion down
            signal = -1
            confidence = 0.60 + (band_position - 0.9) * 2  # Higher confidence at extreme
        elif band_position < 0.1:
            # Near lower band - expect reversion up
            signal = 1
            confidence = 0.60 + (0.1 - band_position) * 2
            
        return signal, min(confidence, 0.75)
        
    def _rsi_signal(self, df: pd.DataFrame, period: int = 14) -> Tuple[int, float]:
        """RSI-based signal - oversold/overbought."""
        if len(df) < period + 5:
            return 0, 0.5
            
        close = df['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < 30:
            # Oversold - potential bounce up
            return 1, 0.60 + (30 - current_rsi) / 100
        elif current_rsi > 70:
            # Overbought - potential drop
            return -1, 0.60 + (current_rsi - 70) / 100
        elif current_rsi < 40:
            return 1, 0.55
        elif current_rsi > 60:
            return -1, 0.55
            
        return 0, 0.5
        
    def _macd_signal(self, df: pd.DataFrame) -> Tuple[int, float]:
        """MACD crossover signal."""
        if len(df) < 30:
            return 0, 0.5
            
        close = df['Close']
        
        # MACD calculation
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        
        current_macd = macd.iloc[-1]
        prev_macd = macd.iloc[-2]
        current_signal = signal_line.iloc[-1]
        prev_signal = signal_line.iloc[-2]
        
        # Crossover detection
        if prev_macd <= prev_signal and current_macd > current_signal:
            # Bullish crossover
            return 1, 0.65
        elif prev_macd >= prev_signal and current_macd < current_signal:
            # Bearish crossover
            return -1, 0.65
        elif current_macd > current_signal:
            # Above signal line
            return 1, 0.55
        elif current_macd < current_signal:
            # Below signal line
            return -1, 0.55
            
        return 0, 0.5
        
    def _stochastic_signal(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[int, float]:
        """Stochastic oscillator signal."""
        if len(df) < k_period + d_period:
            return 0, 0.5
            
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(d_period).mean()
        
        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        prev_k = k.iloc[-2]
        prev_d = d.iloc[-2]
        
        # Oversold/overbought with crossover
        if current_k < 20 and prev_k <= prev_d and current_k > current_d:
            # Bullish crossover in oversold
            return 1, 0.70
        elif current_k > 80 and prev_k >= prev_d and current_k < current_d:
            # Bearish crossover in overbought
            return -1, 0.70
        elif current_k < 30:
            return 1, 0.55
        elif current_k > 70:
            return -1, 0.55
            
        return 0, 0.5
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble signals by combining sub-strategies.
        """
        LOGGER.info("EnsembleStrategy: Generating signals...")
        
        # Initialize signals dataframe
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['probability'] = 0.5
        
        if len(df) < 100:
            return signals
            
        # Get regime
        regime_info = self.regime_detector.detect_regime(df)
        trend_regime = regime_info['trend_regime']
        hurst = regime_info['hurst_exponent']
        
        LOGGER.info(f"Market regime: {trend_regime}, Hurst: {hurst:.2f}")
        
        # Collect signals from each strategy
        strategy_signals: List[Tuple[str, int, float]] = []  # (name, signal, confidence)
        
        # 1. ML Strategy signals (skip if failing)
        try:
            ml_signals = self.ml_strategy.generate_signals(df)
            if not ml_signals.empty and 'signal' in ml_signals.columns:
                last_ml_signal = int(ml_signals['signal'].iloc[-1])
                ml_prob = float(ml_signals.get('probability', pd.Series([0.5])).iloc[-1])
                strategy_signals.append(("ml", last_ml_signal, ml_prob))
        except Exception as e:
            LOGGER.debug(f"ML strategy skipped: {e}")
            
        # 2. Momentum signals (better in trending markets)
        mom_signal, mom_conf = self._momentum_signal(df)
        if self.use_regime_filter and hurst < 0.50:
            mom_conf *= 0.7
        strategy_signals.append(("momentum", mom_signal, mom_conf))
        
        # 3. Mean reversion signals (better in ranging markets)
        mr_signal, mr_conf = self._mean_reversion_signal(df)
        if self.use_regime_filter and hurst > 0.55:
            mr_conf *= 0.7
        strategy_signals.append(("mean_reversion", mr_signal, mr_conf))
        
        # 4. RSI signal
        rsi_signal, rsi_conf = self._rsi_signal(df)
        strategy_signals.append(("rsi", rsi_signal, rsi_conf))
        
        # 5. MACD signal
        macd_signal, macd_conf = self._macd_signal(df)
        strategy_signals.append(("macd", macd_signal, macd_conf))
        
        # 6. Stochastic signal
        stoch_signal, stoch_conf = self._stochastic_signal(df)
        strategy_signals.append(("stochastic", stoch_signal, stoch_conf))
        
        # Combine signals
        if self.combine_method == CombineMethod.VOTING:
            final_signal, final_prob = self._voting_combine(strategy_signals)
        elif self.combine_method == CombineMethod.AVERAGE:
            final_signal, final_prob = self._average_combine(strategy_signals)
        elif self.combine_method == CombineMethod.CONFIDENT:
            final_signal, final_prob = self._confident_combine(strategy_signals)
        else:
            final_signal, final_prob = self._weighted_combine(strategy_signals)
            
        # Apply probability threshold
        if final_prob < self.probability_threshold:
            final_signal = 0
            
        # Set final signals for last bar (walk-forward style)
        signals.loc[signals.index[-1], 'signal'] = final_signal
        signals.loc[signals.index[-1], 'probability'] = final_prob
        
        LOGGER.info(f"Ensemble signal: {final_signal} (prob: {final_prob:.2%})")
        
        return signals
        
    def _voting_combine(
        self, 
        strategies: List[Tuple[str, int, float]]
    ) -> Tuple[int, float]:
        """Majority voting combination - more active version."""
        # Count signals with positive direction
        longs = sum(1 for _, sig, conf in strategies if sig > 0 and conf > 0.50)
        shorts = sum(1 for _, sig, conf in strategies if sig < 0 and conf > 0.50)
        total = len(strategies)
        
        # Lower threshold for signals - need at least 40% agreement
        effective_agreement = max(0.4, self.min_agreement)
        
        if longs / total >= effective_agreement:
            long_confs = [conf for _, sig, conf in strategies if sig > 0 and conf > 0.5]
            # Boost confidence if more signals agree
            bonus = min(0.1, (longs / total - effective_agreement) * 0.5)
            return 1, np.mean(long_confs) + bonus if long_confs else 0.55
            
        if shorts / total >= effective_agreement:
            short_confs = [conf for _, sig, conf in strategies if sig < 0 and conf > 0.5]
            bonus = min(0.1, (shorts / total - effective_agreement) * 0.5)
            return -1, np.mean(short_confs) + bonus if short_confs else 0.55
        
        # If close to agreement, still take position with lower confidence
        if longs / total >= 0.3 and longs > shorts:
            long_confs = [conf for _, sig, conf in strategies if sig > 0]
            return 1, np.mean(long_confs) * 0.9 if long_confs else 0.52
            
        if shorts / total >= 0.3 and shorts > longs:
            short_confs = [conf for _, sig, conf in strategies if sig < 0]
            return -1, np.mean(short_confs) * 0.9 if short_confs else 0.52
            
        return 0, 0.5
        
    def _average_combine(
        self,
        strategies: List[Tuple[str, int, float]]
    ) -> Tuple[int, float]:
        """Simple average of signals."""
        if not strategies:
            return 0, 0.5
            
        avg_signal = np.mean([sig for _, sig, _ in strategies])
        avg_conf = np.mean([conf for _, _, conf in strategies])
        
        if avg_signal > 0.3:
            return 1, avg_conf
        elif avg_signal < -0.3:
            return -1, avg_conf
        return 0, 0.5
        
    def _confident_combine(
        self,
        strategies: List[Tuple[str, int, float]]
    ) -> Tuple[int, float]:
        """Take the signal with highest confidence."""
        if not strategies:
            return 0, 0.5
            
        best = max(strategies, key=lambda x: x[2])
        return best[1], best[2]
        
    def _weighted_combine(
        self,
        strategies: List[Tuple[str, int, float]]
    ) -> Tuple[int, float]:
        """Performance-weighted average (based on recent win rates)."""
        if not strategies:
            return 0, 0.5
            
        weighted_signal = 0.0
        total_weight = 0.0
        
        for name, sig, conf in strategies:
            # Get historical performance weight
            history = self.strategy_stats.get(name, [])
            if history:
                win_rate = sum(1 for x in history if x > 0) / len(history)
            else:
                win_rate = 0.5  # Default
                
            weight = conf * (0.5 + win_rate)  # Boost by win rate
            weighted_signal += sig * weight
            total_weight += weight
            
        if total_weight == 0:
            return 0, 0.5
            
        avg_signal = weighted_signal / total_weight
        avg_conf = np.mean([conf for _, _, conf in strategies])
        
        if avg_signal > 0.3:
            return 1, avg_conf
        elif avg_signal < -0.3:
            return -1, avg_conf
        return 0, 0.5
        
    def update_performance(self, strategy_name: str, pnl: float):
        """Update strategy performance for weighted combining."""
        if strategy_name in self.strategy_stats:
            self.strategy_stats[strategy_name].append(pnl)
            # Keep only last 100 trades
            self.strategy_stats[strategy_name] = self.strategy_stats[strategy_name][-100:]
