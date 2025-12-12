"""Trade Filter for Pre-Trade Validation.

Filters trades based on market conditions to improve win rate and reduce losses.
Only allows trades when conditions are favorable.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import get_logger
from ..quant.regime_detector import RegimeDetector, RegimeType

LOGGER = get_logger(__name__)

class FilterResult(Enum):
    """Trade filter result."""
    APPROVED = "approved"
    REJECTED_REGIME = "rejected_regime"
    REJECTED_VOLATILITY = "rejected_volatility"
    REJECTED_PROBABILITY = "rejected_probability"
    REJECTED_CORRELATION = "rejected_correlation"
    REJECTED_TIME = "rejected_time"

@dataclass
class TradeSignal:
    """Trade signal with metadata."""
    symbol: str
    side: str  # "LONG" or "SHORT"
    probability: float
    features: Dict[str, float]
    
@dataclass 
class FilterConfig:
    """Configuration for trade filters."""
    min_probability: float = 0.60  # Minimum ML probability
    min_hurst: float = 0.55  # Minimum Hurst for trending trades
    max_hurst: float = 0.45  # Maximum Hurst for mean reversion
    max_volatility_percentile: float = 0.90  # Reject if vol > 90th percentile
    min_volatility_percentile: float = 0.10  # Reject if vol < 10th percentile
    allow_counter_trend: bool = False  # Allow trades against regime
    max_correlation: float = 0.70  # Max correlation with existing positions

class TradeFilter:
    """
    Pre-trade validation filter to improve trade quality.
    
    Checks:
    1. Market regime (trending vs ranging)
    2. Volatility (not too high or too low)
    3. ML probability threshold
    4. Correlation with existing positions
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self.regime_detector = RegimeDetector(lookback=100)
        self._volatility_history = []
        
    def check_trade(
        self,
        signal: TradeSignal,
        df: pd.DataFrame,
        existing_positions: Optional[Dict[str, float]] = None
    ) -> Tuple[FilterResult, str]:
        """
        Check if a trade should be taken.
        
        Args:
            signal: The trade signal to validate.
            df: Recent OHLCV data for the symbol.
            existing_positions: Current open positions {symbol: qty}.
            
        Returns:
            Tuple of (FilterResult, reason_string)
        """
        # Check 1: ML Probability
        if signal.probability < self.config.min_probability:
            return (
                FilterResult.REJECTED_PROBABILITY,
                f"Probability {signal.probability:.2%} < {self.config.min_probability:.2%}"
            )
            
        # Check 2: Market Regime
        if len(df) >= 20:
            regime = self.regime_detector.detect_regime(df)
            hurst = regime['hurst_exponent']
            trend = regime['trend_regime']
            
            # For LONG trades, prefer bullish trending markets
            if signal.side == "LONG":
                if hurst < self.config.min_hurst:
                    if not self.config.allow_counter_trend:
                        return (
                            FilterResult.REJECTED_REGIME,
                            f"Hurst {hurst:.2f} < {self.config.min_hurst} (not trending enough for LONG)"
                        )
                        
                if trend == RegimeType.TRENDING_DOWN and not self.config.allow_counter_trend:
                    return (
                        FilterResult.REJECTED_REGIME,
                        f"Market in downtrend, rejecting LONG"
                    )
                    
            # For SHORT trades, prefer bearish trending markets
            elif signal.side == "SHORT":
                if hurst < self.config.min_hurst:
                    if not self.config.allow_counter_trend:
                        return (
                            FilterResult.REJECTED_REGIME,
                            f"Hurst {hurst:.2f} < {self.config.min_hurst} (not trending enough for SHORT)"
                        )
                        
                if trend == RegimeType.TRENDING_UP and not self.config.allow_counter_trend:
                    return (
                        FilterResult.REJECTED_REGIME,
                        f"Market in uptrend, rejecting SHORT"
                    )
                    
            # Check 3: Volatility
            vol = regime['volatility']
            
            if regime['volatility_regime'] == RegimeType.HIGH_VOLATILITY:
                # High vol can be good for momentum but risky
                # Allow but log warning
                LOGGER.warning(f"High volatility detected for {signal.symbol}")
                
        # Check 4: Correlation with existing positions
        if existing_positions:
            # Simple check: don't over-concentrate
            symbol_exposure = abs(existing_positions.get(signal.symbol, 0))
            total_exposure = sum(abs(v) for v in existing_positions.values())
            
            if total_exposure > 0:
                concentration = symbol_exposure / total_exposure
                if concentration > 0.3:  # More than 30% in one symbol
                    return (
                        FilterResult.REJECTED_CORRELATION,
                        f"Already {concentration:.1%} exposure to {signal.symbol}"
                    )
                    
        # All checks passed
        LOGGER.info(f"Trade APPROVED: {signal.side} {signal.symbol} (prob: {signal.probability:.2%})")
        return (FilterResult.APPROVED, "All filters passed")
        
    def get_position_size_multiplier(
        self,
        signal: TradeSignal,
        df: pd.DataFrame
    ) -> float:
        """
        Get a position size multiplier based on signal quality.
        
        Returns:
            Multiplier between 0.5 and 2.0
        """
        multiplier = 1.0
        
        # Boost for high probability
        if signal.probability > 0.70:
            multiplier *= 1.3
        elif signal.probability > 0.65:
            multiplier *= 1.15
        elif signal.probability < 0.55:
            multiplier *= 0.7
            
        # Check regime alignment
        if len(df) >= 20:
            regime = self.regime_detector.detect_regime(df)
            hurst = regime['hurst_exponent']
            
            # Boost for strong trend
            if hurst > 0.65:
                multiplier *= 1.2
            elif hurst > 0.60:
                multiplier *= 1.1
                
            # Reduce for choppy market
            if hurst < 0.50:
                multiplier *= 0.8
                
        # Cap multiplier
        return max(0.5, min(2.0, multiplier))
