"""
Risk Management Module

Single Responsibility: Only handles risk calculations (ATR, position sizing, stops).
"""
from typing import Tuple
import pandas as pd
import numpy as np


class RiskManager:
    """
    Risk management for position sizing and stop loss.
    
    Implements:
    - ATR-based stop loss / take profit
    - Fixed fractional position sizing (risk X% per trade)
    """
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).
        
        ATR = Average of True Range over N periods
        True Range = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        # Handle NaN: return a safe fallback based on recent price range
        if pd.isna(atr) or atr <= 0:
            # Fallback: use simple high-low range of last bar
            atr = float(df['High'].iloc[-1] - df['Low'].iloc[-1])
            if atr <= 0:
                # Ultimate fallback: 1% of current price
                atr = float(df['Close'].iloc[-1]) * 0.01
        
        return float(atr)
    
    @staticmethod
    def calculate_position_size(
        equity: float,
        entry_price: float,
        stop_loss_price: float,
        risk_percent: float = 0.02
    ) -> float:
        """
        Calculate position size based on risk.
        
        Position Size = (Equity * Risk%) / (Entry - StopLoss)
        
        Args:
            equity: Account equity
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_percent: Fraction of equity to risk (e.g., 0.02 = 2%)
            
        Returns:
            Position size in units
        """
        risk_amount = equity * risk_percent
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            return 0.0
            
        position_size = risk_amount / risk_per_unit
        return position_size
    
    @staticmethod
    def calculate_stops(
        entry_price: float,
        atr: float,
        side: str,  # "LONG" or "SHORT"
        sl_mult: float = 2.0,
        tp_mult: float = 3.0
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            side: "LONG" or "SHORT"
            sl_mult: Stop loss ATR multiplier
            tp_mult: Take profit ATR multiplier
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if side.upper() == "LONG":
            stop_loss = entry_price - (atr * sl_mult)
            take_profit = entry_price + (atr * tp_mult)
        else:
            stop_loss = entry_price + (atr * sl_mult)
            take_profit = entry_price - (atr * tp_mult)
            
        return stop_loss, take_profit


