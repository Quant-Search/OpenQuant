"""
Integration Tests for OpenQuant
================================
End-to-end tests that verify the complete trading workflow.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from robot.strategy import KalmanStrategy
from robot.data_fetcher import DataFetcher
from robot.risk_manager import RiskManager
from robot.trader import Trader
from robot.backtester import Backtester


class TestTradingWorkflow:
    """Test complete trading workflow from data to execution."""
    
    def test_signal_generation_to_order(self, sample_ohlcv_data):
        """Test the complete flow from signal to order placement."""
        # 1. Create strategy
        strategy = KalmanStrategy(threshold=1.5)
        
        # 2. Generate signals
        signals = strategy.generate_signals(sample_ohlcv_data)
        
        # 3. Find a signal
        signal_indices = signals[signals != 0].index
        
        if len(signal_indices) > 0:
            # 4. Get data for risk management
            last_idx = signal_indices[-1]
            idx_loc = sample_ohlcv_data.index.get_loc(last_idx)
            df_slice = sample_ohlcv_data.iloc[:idx_loc+1]
            
            # 5. Calculate ATR and stops
            atr = RiskManager.calculate_atr(df_slice)
            current_price = float(df_slice['Close'].iloc[-1])
            signal = int(signals.loc[last_idx])
            side = "LONG" if signal == 1 else "SHORT"
            
            sl, tp = RiskManager.calculate_stops(current_price, atr, side)
            
            # 6. Calculate position size
            equity = 10000
            position_size = RiskManager.calculate_position_size(
                equity, current_price, sl, 0.02
            )
            
            # 7. Place order (paper)
            trader = Trader(mode="paper")
            volume = max(0.01, round(position_size / 100000, 2))
            
            result = trader.place_order("EURUSD", "BUY" if side == "LONG" else "SELL", 
                                        volume, sl, tp)
            
            assert result is True
            assert "EURUSD" in trader.get_positions()
    
    def test_backtest_to_optimization(self, sample_ohlcv_data):
        """Test backtest integration with different parameters."""
        results = []
        
        for threshold in [1.0, 1.5, 2.0]:
            strategy = KalmanStrategy(threshold=threshold)
            bt = Backtester(strategy=strategy)
            
            try:
                result = bt.run(sample_ohlcv_data, "EURUSD")
                results.append({
                    'threshold': threshold,
                    'return': result.metrics.total_return,
                    'sharpe': result.metrics.sharpe_ratio,
                    'trades': result.metrics.total_trades
                })
            except ValueError:
                pass
        
        # Should have at least some results
        assert len(results) > 0
    
    def test_position_tracking(self, sample_ohlcv_data):
        """Test that positions are tracked correctly through trades."""
        trader = Trader(mode="paper")
        
        # Initial state
        assert trader.get_equity() == 10000
        assert len(trader.get_positions()) == 0
        
        # Open position
        trader.place_order("EURUSD", "BUY", 0.1, 1.08, 1.15)
        
        # Check position exists
        positions = trader.get_positions()
        assert "EURUSD" in positions
        assert positions["EURUSD"] > 0
        
        # Update price and check equity
        trader.update_paper_prices({"EURUSD": 1.11})
        equity = trader.get_equity()
        
        # Equity should have changed
        assert equity != 10000


class TestDataToSignals:
    """Test data fetching to signal generation pipeline."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_live_data_signal_generation(self):
        """Test generating signals from live data."""
        fetcher = DataFetcher(use_mt5=False)
        strategy = KalmanStrategy()
        
        # Fetch data
        df = fetcher.fetch("EURUSD", "1d", 200)
        
        if not df.empty and len(df) >= 50:
            # Generate signals
            signals = strategy.generate_signals(df)
            
            assert len(signals) == len(df)
            assert signals.dtype == int


class TestRiskWorkflow:
    """Test risk management in complete workflow."""
    
    def test_position_sizing_integration(self, sample_ohlcv_data):
        """Test position sizing with real ATR."""
        # Get ATR from data
        atr = RiskManager.calculate_atr(sample_ohlcv_data)
        current_price = float(sample_ohlcv_data['Close'].iloc[-1])
        
        # Calculate stops
        sl, tp = RiskManager.calculate_stops(current_price, atr, "LONG")
        
        # Calculate position size
        equity = 10000
        risk_pct = 0.02
        position_size = RiskManager.calculate_position_size(
            equity, current_price, sl, risk_pct
        )
        
        # Verify risk amount
        risk_per_unit = abs(current_price - sl)
        expected_risk = equity * risk_pct
        actual_risk = position_size * risk_per_unit
        
        assert abs(actual_risk - expected_risk) < 1  # Within $1


class TestEquityCurveIntegration:
    """Test equity curve through full backtest."""
    
    def test_equity_curve_consistency(self, sample_ohlcv_data):
        """Test that equity curve is consistent with trades."""
        bt = Backtester(initial_capital=10000)
        result = bt.run(sample_ohlcv_data, "EURUSD")
        
        # Starting equity should be initial capital
        assert result.equity_curve.iloc[0] == 10000
        
        # Final equity should reflect trade results
        if result.trades:
            total_pnl = sum(t.pnl for t in result.trades)
            final_equity = result.equity_curve.iloc[-1]
            
            # Should be close (commissions may cause slight difference)
            assert abs(final_equity - 10000 - total_pnl) < 100


class TestMultiSymbolWorkflow:
    """Test workflow with multiple symbols."""
    
    def test_multiple_symbols_independent(self):
        """Test that multiple symbols are handled independently."""
        trader = Trader(mode="paper")
        
        # Open positions in multiple symbols
        trader.place_order("EURUSD", "BUY", 0.1)
        trader.place_order("GBPUSD", "SELL", 0.1)
        trader.place_order("USDJPY", "BUY", 0.1)
        
        positions = trader.get_positions()
        
        assert len(positions) == 3
        assert "EURUSD" in positions
        assert "GBPUSD" in positions
        assert "USDJPY" in positions
        
        # Each should have correct direction
        assert positions["EURUSD"] > 0  # Long
        assert positions["GBPUSD"] < 0  # Short
        assert positions["USDJPY"] > 0  # Long

