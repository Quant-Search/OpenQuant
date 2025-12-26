"""
Unit Tests for Risk Manager Module
===================================
Tests for position sizing, ATR calculation, and stop levels.
"""
import pytest
import pandas as pd
import numpy as np
from robot.risk_manager import RiskManager


class TestATRCalculation:
    """Test Average True Range calculation."""
    
    def test_atr_positive(self, sample_ohlcv_data):
        """Test that ATR is always positive."""
        atr = RiskManager.calculate_atr(sample_ohlcv_data)
        assert atr > 0
    
    def test_atr_reasonable_value(self, sample_ohlcv_data):
        """Test that ATR is within reasonable bounds."""
        atr = RiskManager.calculate_atr(sample_ohlcv_data)
        price = sample_ohlcv_data['Close'].iloc[-1]
        
        # ATR should be less than 10% of price for normal markets
        assert atr < price * 0.1
    
    def test_atr_with_different_periods(self, sample_ohlcv_data):
        """Test ATR with different periods."""
        atr_14 = RiskManager.calculate_atr(sample_ohlcv_data, period=14)
        atr_7 = RiskManager.calculate_atr(sample_ohlcv_data, period=7)
        atr_21 = RiskManager.calculate_atr(sample_ohlcv_data, period=21)
        
        # All should be positive
        assert all(atr > 0 for atr in [atr_14, atr_7, atr_21])
    
    def test_atr_fallback_insufficient_data(self, small_ohlcv_data):
        """Test ATR fallback when insufficient data."""
        atr = RiskManager.calculate_atr(small_ohlcv_data, period=14)
        
        # Should return fallback value (high-low or 1% of close)
        assert atr > 0


class TestPositionSizing:
    """Test position size calculations."""
    
    def test_basic_position_size(self):
        """Test basic position size calculation."""
        size = RiskManager.calculate_position_size(
            equity=10000,
            entry_price=1.10,
            stop_loss_price=1.08,
            risk_percent=0.02
        )
        
        # Risk $200, risk per unit = 0.02, size = 200/0.02 = 10000
        expected = 200 / 0.02  # 10000 units
        assert abs(size - expected) < 0.01
    
    def test_position_size_scales_with_equity(self):
        """Test that position size scales with equity."""
        size_10k = RiskManager.calculate_position_size(
            equity=10000, entry_price=1.10, 
            stop_loss_price=1.08, risk_percent=0.02
        )
        size_20k = RiskManager.calculate_position_size(
            equity=20000, entry_price=1.10,
            stop_loss_price=1.08, risk_percent=0.02
        )
        
        assert abs(size_20k - 2 * size_10k) < 0.01
    
    def test_position_size_scales_with_risk(self):
        """Test that position size scales with risk percent."""
        size_1pct = RiskManager.calculate_position_size(
            equity=10000, entry_price=1.10,
            stop_loss_price=1.08, risk_percent=0.01
        )
        size_2pct = RiskManager.calculate_position_size(
            equity=10000, entry_price=1.10,
            stop_loss_price=1.08, risk_percent=0.02
        )
        
        assert abs(size_2pct - 2 * size_1pct) < 0.01
    
    def test_position_size_zero_stop_distance(self):
        """Test position size when stop = entry (edge case)."""
        size = RiskManager.calculate_position_size(
            equity=10000, entry_price=1.10,
            stop_loss_price=1.10, risk_percent=0.02
        )
        
        assert size == 0.0
    
    def test_position_size_short_trade(self):
        """Test position size for short trades."""
        # For shorts, stop is above entry
        size = RiskManager.calculate_position_size(
            equity=10000, entry_price=1.10,
            stop_loss_price=1.12, risk_percent=0.02
        )
        
        assert size > 0


class TestStopLevels:
    """Test stop loss and take profit calculations."""
    
    def test_long_stops(self):
        """Test stops for long position."""
        sl, tp = RiskManager.calculate_stops(
            entry_price=1.10, atr=0.01,
            side="LONG", sl_mult=2.0, tp_mult=3.0
        )
        
        assert sl < 1.10  # SL below entry
        assert tp > 1.10  # TP above entry
        assert abs(sl - 1.08) < 0.001  # 1.10 - 2*0.01
        assert abs(tp - 1.13) < 0.001  # 1.10 + 3*0.01
    
    def test_short_stops(self):
        """Test stops for short position."""
        sl, tp = RiskManager.calculate_stops(
            entry_price=1.10, atr=0.01,
            side="SHORT", sl_mult=2.0, tp_mult=3.0
        )
        
        assert sl > 1.10  # SL above entry
        assert tp < 1.10  # TP below entry
        assert abs(sl - 1.12) < 0.001  # 1.10 + 2*0.01
        assert abs(tp - 1.07) < 0.001  # 1.10 - 3*0.01
    
    def test_stops_case_insensitive(self):
        """Test that side is case insensitive."""
        sl1, tp1 = RiskManager.calculate_stops(1.10, 0.01, "LONG")
        sl2, tp2 = RiskManager.calculate_stops(1.10, 0.01, "long")
        sl3, tp3 = RiskManager.calculate_stops(1.10, 0.01, "Long")
        
        assert sl1 == sl2 == sl3
        assert tp1 == tp2 == tp3
    
    def test_stops_with_different_multipliers(self):
        """Test stops with different ATR multipliers."""
        sl_tight, tp_tight = RiskManager.calculate_stops(
            1.10, 0.01, "LONG", sl_mult=1.0, tp_mult=1.5
        )
        sl_wide, tp_wide = RiskManager.calculate_stops(
            1.10, 0.01, "LONG", sl_mult=3.0, tp_mult=5.0
        )
        
        # Wider stops = further from entry
        assert abs(1.10 - sl_wide) > abs(1.10 - sl_tight)
        assert abs(tp_wide - 1.10) > abs(tp_tight - 1.10)


class TestEdgeCases:
    """Test edge cases for risk management."""
    
    def test_atr_single_bar(self):
        """Test ATR with minimal data."""
        df = pd.DataFrame({
            'Open': [1.10],
            'High': [1.11],
            'Low': [1.09],
            'Close': [1.10],
            'Volume': [1000]
        })
        
        atr = RiskManager.calculate_atr(df)
        assert atr > 0
    
    def test_position_size_negative_equity(self):
        """Test position size with negative equity (should still calculate)."""
        # This is a safeguard - negative equity shouldn't happen but test anyway
        size = RiskManager.calculate_position_size(
            equity=-1000, entry_price=1.10,
            stop_loss_price=1.08, risk_percent=0.02
        )
        
        # Should be negative (or we could make it 0)
        assert isinstance(size, float)

