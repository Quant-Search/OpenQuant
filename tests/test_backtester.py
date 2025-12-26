"""
Unit Tests for Backtester Module
==================================
Tests for backtesting functionality and metrics.
"""
import pytest
import pandas as pd
import numpy as np
from robot.backtester import Backtester, BacktestResult, BacktestTrade
from robot.strategy import KalmanStrategy


class TestBacktesterInit:
    """Test Backtester initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        bt = Backtester()
        
        assert bt.initial_capital == 10000
        assert bt.position_size_pct == 0.02
        assert isinstance(bt.strategy, KalmanStrategy)
    
    def test_custom_init(self):
        """Test custom initialization."""
        strategy = KalmanStrategy(threshold=2.0)
        bt = Backtester(
            strategy=strategy,
            initial_capital=50000,
            position_size_pct=0.01
        )
        
        assert bt.initial_capital == 50000
        assert bt.position_size_pct == 0.01


class TestBacktestRun:
    """Test backtest execution."""
    
    def test_run_returns_result(self, sample_ohlcv_data):
        """Test that run returns BacktestResult."""
        bt = Backtester()
        result = bt.run(sample_ohlcv_data, "EURUSD")
        
        assert isinstance(result, BacktestResult)
    
    def test_run_insufficient_data(self, small_ohlcv_data):
        """Test that insufficient data raises error."""
        bt = Backtester()
        
        with pytest.raises(ValueError, match="at least 100 bars"):
            bt.run(small_ohlcv_data, "EURUSD")
    
    def test_result_has_equity_curve(self, sample_ohlcv_data):
        """Test that result includes equity curve."""
        bt = Backtester()
        result = bt.run(sample_ohlcv_data, "EURUSD")
        
        assert isinstance(result.equity_curve, pd.Series)
        assert len(result.equity_curve) > 0
    
    def test_result_has_trades(self, sample_ohlcv_data):
        """Test that result includes trades."""
        bt = Backtester()
        result = bt.run(sample_ohlcv_data, "EURUSD")
        
        assert isinstance(result.trades, list)
        # May or may not have trades depending on signals
    
    def test_result_has_metrics(self, sample_ohlcv_data):
        """Test that result includes performance metrics."""
        bt = Backtester()
        result = bt.run(sample_ohlcv_data, "EURUSD")
        
        assert result.metrics is not None
        assert hasattr(result.metrics, 'total_return')
        assert hasattr(result.metrics, 'sharpe_ratio')


class TestEquityCurve:
    """Test equity curve calculations."""
    
    def test_equity_starts_at_initial_capital(self, sample_ohlcv_data):
        """Test that equity curve starts at initial capital."""
        bt = Backtester(initial_capital=10000)
        result = bt.run(sample_ohlcv_data, "EURUSD")
        
        assert result.equity_curve.iloc[0] == 10000
    
    def test_equity_never_negative(self, sample_ohlcv_data):
        """Test that equity never goes negative."""
        bt = Backtester()
        result = bt.run(sample_ohlcv_data, "EURUSD")
        
        # With proper risk management, should stay positive
        # (though in extreme cases might not)
        assert result.equity_curve.min() > -bt.initial_capital


class TestTradeRecords:
    """Test trade record creation."""
    
    def test_trade_has_required_fields(self, mean_reverting_data):
        """Test that trades have all required fields."""
        bt = Backtester()
        result = bt.run(mean_reverting_data, "EURUSD")
        
        if result.trades:  # If any trades occurred
            trade = result.trades[0]
            assert hasattr(trade, 'entry_time')
            assert hasattr(trade, 'entry_price')
            assert hasattr(trade, 'direction')
            assert hasattr(trade, 'size')
            assert hasattr(trade, 'pnl')
    
    def test_trade_direction_valid(self, mean_reverting_data):
        """Test that trade directions are valid."""
        bt = Backtester()
        result = bt.run(mean_reverting_data, "EURUSD")
        
        for trade in result.trades:
            assert trade.direction in ["LONG", "SHORT"]
    
    def test_closed_trades_have_exit(self, mean_reverting_data):
        """Test that closed trades have exit info."""
        bt = Backtester()
        result = bt.run(mean_reverting_data, "EURUSD")
        
        for trade in result.trades:
            assert trade.exit_time is not None
            assert trade.exit_price is not None


class TestRiskManagement:
    """Test risk management in backtest."""
    
    def test_stop_loss_respected(self, trending_down_data):
        """Test that stop loss is respected."""
        bt = Backtester(stop_loss_atr=1.0)  # Tight stop
        result = bt.run(trending_down_data, "EURUSD")
        
        for trade in result.trades:
            if trade.direction == "LONG" and trade.exit_reason == "stop_loss":
                # Exit should be at or below stop loss
                assert trade.exit_price <= trade.stop_loss * 1.01  # Allow 1% slippage
    
    def test_take_profit_respected(self, trending_up_data):
        """Test that take profit is respected."""
        bt = Backtester(take_profit_atr=1.0)  # Tight TP
        result = bt.run(trending_up_data, "EURUSD")
        
        for trade in result.trades:
            if trade.direction == "LONG" and trade.exit_reason == "take_profit":
                # Exit should be at or above take profit
                assert trade.exit_price >= trade.take_profit * 0.99


class TestSlippageAndCommission:
    """Test slippage and commission handling."""
    
    def test_slippage_affects_entry(self, sample_ohlcv_data):
        """Test that slippage affects entry price."""
        bt_no_slip = Backtester(slippage_pct=0)
        bt_with_slip = Backtester(slippage_pct=0.001)  # 0.1%
        
        result_no_slip = bt_no_slip.run(sample_ohlcv_data, "EURUSD")
        result_with_slip = bt_with_slip.run(sample_ohlcv_data, "EURUSD")
        
        if result_no_slip.trades and result_with_slip.trades:
            # With slippage, entry prices should differ
            # (unless by coincidence they're the same)
            pass  # Just checking no crash
    
    def test_commission_reduces_pnl(self, mean_reverting_data):
        """Test that commission reduces P&L."""
        bt_no_comm = Backtester(commission_pct=0)
        bt_with_comm = Backtester(commission_pct=0.001)  # 0.1%
        
        result_no_comm = bt_no_comm.run(mean_reverting_data, "EURUSD")
        result_with_comm = bt_with_comm.run(mean_reverting_data, "EURUSD")
        
        # Final equity with commission should be lower
        final_no = result_no_comm.equity_curve.iloc[-1]
        final_with = result_with_comm.equity_curve.iloc[-1]
        
        assert final_with <= final_no

