"""Dynamic TP/SL calculation based on market conditions.

Provides multiple methods for setting Take Profit and Stop Loss levels:
- ATR-based (volatility adaptive)
- Support/Resistance levels
- Volatility-adjusted
- Time-based decay
"""
from __future__ import annotations
from typing import Tuple, Optional
from enum import Enum
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class ExitMethod(Enum):
    """Available methods for calculating TP/SL."""
    ATR = "atr"
    SUPPORT_RESISTANCE = "support_resistance"
    VOLATILITY = "volatility"
    TIME_BASED = "time_based"


class DynamicExitCalculator:
    """Calculate dynamic TP/SL levels based on market conditions."""
    
    def __init__(
        self,
        method: str = "atr",
        tp_atr_multiplier: float = 3.0,
        sl_atr_multiplier: float = 1.5,
        atr_period: int = 14,
        lookback_bars: int = 50
    ):
        """Initialize the exit calculator.
        
        Args:
            method: Calculation method (atr, support_resistance, volatility, time_based)
            tp_atr_multiplier: TP distance in ATR multiples
            sl_atr_multiplier: SL distance in ATR multiples  
            atr_period: Period for ATR calculation
            lookback_bars: Number of bars to look back for S/R levels
        """
        try:
            self.method = ExitMethod(method.lower())
        except ValueError:
            LOGGER.warning(f"Unknown method {method}, defaulting to ATR")
            self.method = ExitMethod.ATR
            
        self.tp_atr_mult = tp_atr_multiplier
        self.sl_atr_mult = sl_atr_multiplier
        self.atr_period = atr_period
        self.lookback = lookback_bars
        
    def calculate_exits(
        self,
        df: pd.DataFrame,
        entry_price: float,
        side: str,
        position_age_hours: float = 0.0
    ) -> Tuple[float, float]:
        """Calculate TP and SL levels.
        
        Args:
            df: OHLCV DataFrame with recent price data
            entry_price: Position entry price
            side: "LONG" or "SHORT"
            position_age_hours: Hours since position opened (for time-based method)
            
        Returns:
            Tuple of (take_profit, stop_loss) prices
        """
        if df.empty or len(df) < self.atr_period:
            # Fallback: use simple percentage
            LOGGER.warning("Insufficient data for dynamic exits, using fallback")
            return self._fallback_exits(entry_price, side)
            
        if self.method == ExitMethod.ATR:
            return self._atr_based(df, entry_price, side)
        elif self.method == ExitMethod.SUPPORT_RESISTANCE:
            return self._support_resistance(df, entry_price, side)
        elif self.method == ExitMethod.VOLATILITY:
            return self._volatility_based(df, entry_price, side)
        elif self.method == ExitMethod.TIME_BASED:
            return self._time_based(df, entry_price, side, position_age_hours)
        else:
            return self._fallback_exits(entry_price, side)
            
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range."""
        try:
            high = df['High'].values
            low = df['Low'].values
            close = df['Close'].values
            
            # True Range
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))
                )
            )
            
            # Skip first element (roll artifact)
            tr = tr[1:]
            
            # ATR as simple moving average of TR
            atr = np.mean(tr[-self.atr_period:]) if len(tr) >= self.atr_period else np.mean(tr)
            
            return float(atr)
            
        except Exception as e:
            LOGGER.error(f"Error calculating ATR: {e}")
            # Fallback to percentage of price
            return float(df['Close'].iloc[-1] * 0.01)
            
    def _atr_based(
        self,
        df: pd.DataFrame,
        entry_price: float,
        side: str
    ) -> Tuple[float, float]:
        """ATR-based TP/SL calculation."""
        atr = self._calculate_atr(df)
        
        if side == "LONG":
            tp = entry_price + (atr * self.tp_atr_mult)
            sl = entry_price - (atr * self.sl_atr_mult)
        else:  # SHORT
            tp = entry_price - (atr * self.tp_atr_mult)
            sl = entry_price + (atr * self.sl_atr_mult)
            
        return (float(tp), float(sl))
        
    def _support_resistance(
        self,
        df: pd.DataFrame,
        entry_price: float,
        side: str
    ) -> Tuple[float, float]:
        """TP/SL based on recent swing highs/lows."""
        try:
            recent = df.tail(self.lookback)
            
            # Find swing highs and lows (local extrema)
            highs = recent['High'].values
            lows = recent['Low'].values
            
            # Simple approach: highest high and lowest low in lookback period
            resistance = float(np.max(highs))
            support = float(np.min(lows))
            
            if side == "LONG":
                # TP at resistance, SL below support
                tp = resistance
                sl = support * 0.998  # Slightly below support
            else:  # SHORT
                # TP at support, SL above resistance
                tp = support
                sl = resistance * 1.002  # Slightly above resistance
                
            # Sanity check: ensure TP is profitable and SL is protective
            if side == "LONG":
                tp = max(tp, entry_price * 1.005)  # At least 0.5% profit
                sl = min(sl, entry_price * 0.985)  # At least 1.5% risk
            else:
                tp = min(tp, entry_price * 0.995)
                sl = max(sl, entry_price * 1.015)
                
            return (tp, sl)
            
        except Exception as e:
            LOGGER.error(f"Error in support/resistance calc: {e}")
            return self._fallback_exits(entry_price, side)
            
    def _volatility_based(
        self,
        df: pd.DataFrame,
        entry_price: float,
        side: str
    ) -> Tuple[float, float]:
        """Volatility-adjusted TP/SL (wider in high vol, tighter in low vol)."""
        try:
            # Calculate recent volatility
            returns = df['Close'].pct_change().dropna()
            recent_vol = float(returns.tail(self.lookback).std())
            
            # Historical average volatility
            avg_vol = float(returns.std())
            
            # Volatility ratio (current / average)
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
            
            # Adjust multipliers based on volatility
            # High vol -> wider stops, Low vol -> tighter stops
            adjusted_tp_mult = self.tp_atr_mult * vol_ratio
            adjusted_sl_mult = self.sl_atr_mult * vol_ratio
            
            # Calculate ATR-based exits with adjusted multipliers
            atr = self._calculate_atr(df)
            
            if side == "LONG":
                tp = entry_price + (atr * adjusted_tp_mult)
                sl = entry_price - (atr * adjusted_sl_mult)
            else:
                tp = entry_price - (atr * adjusted_tp_mult)
                sl = entry_price + (atr * adjusted_sl_mult)
                
            return (float(tp), float(sl))
            
        except Exception as e:
            LOGGER.error(f"Error in volatility calc: {e}")
            return self._fallback_exits(entry_price, side)
            
    def _time_based(
        self,
        df: pd.DataFrame,
        entry_price: float,
        side: str,
        position_age_hours: float
    ) -> Tuple[float, float]:
        """Time-based TP/SL (reduce TP distance as position ages)."""
        try:
            # Base calculation (ATR)
            base_tp, base_sl = self._atr_based(df, entry_price, side)
            
            # Decay factor: reduce TP distance over time
            # After 24 hours, TP is at 70% of original distance
            # After 72 hours, TP is at 50% of original distance
            max_age_hours = 72.0
            decay_rate = 0.5  # Final TP distance as fraction of original
            
            time_factor = 1.0 - ((1.0 - decay_rate) * min(position_age_hours / max_age_hours, 1.0))
            
            # Apply decay to TP distance
            tp_distance = abs(base_tp - entry_price) * time_factor
            
            if side == "LONG":
                tp = entry_price + tp_distance
            else:
                tp = entry_price - tp_distance
                
            # SL remains unchanged (always protective)
            sl = base_sl
            
            return (float(tp), float(sl))
            
        except Exception as e:
            LOGGER.error(f"Error in time-based calc: {e}")
            return self._fallback_exits(entry_price, side)
            
    def _fallback_exits(
        self,
        entry_price: float,
        side: str
    ) -> Tuple[float, float]:
        """Fallback TP/SL using simple percentages."""
        tp_pct = 0.02  # 2% TP
        sl_pct = 0.01  # 1% SL
        
        if side == "LONG":
            tp = entry_price * (1.0 + tp_pct)
            sl = entry_price * (1.0 - sl_pct)
        else:
            tp = entry_price * (1.0 - tp_pct)
            sl = entry_price * (1.0 + sl_pct)
            
        return (float(tp), float(sl))
