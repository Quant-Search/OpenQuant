"""
Unit Tests for Trader Module
=============================
Tests for paper trading and order execution.
"""
import pytest
from datetime import datetime, timezone
from robot.trader import Trader


class TestTraderInit:
    """Test Trader initialization."""
    
    def test_paper_mode_init(self):
        """Test initialization in paper mode."""
        trader = Trader(mode="paper")
        
        assert trader.mode == "paper"
        assert trader._paper_cash == 10000.0
        assert trader._paper_pnl == 0.0
        assert trader._paper_positions == {}
    
    def test_live_mode_init(self):
        """Test initialization in live mode."""
        trader = Trader(mode="live")
        
        assert trader.mode == "live"
        assert trader._mt5 is None


class TestPaperEquity:
    """Test paper trading equity calculations."""
    
    def test_initial_equity(self):
        """Test initial equity is starting cash."""
        trader = Trader(mode="paper")
        
        assert trader.get_equity() == 10000.0
    
    def test_equity_with_long_position_profit(self):
        """Test equity with profitable long position."""
        trader = Trader(mode="paper")

        # Open long position
        trader._paper_positions["EURUSD"] = {
            "volume": 0.1,  # 0.1 lots
            "entry_price": 1.1000,
            "current_price": 1.1100,  # +100 pips
            "side": "BUY"
        }

        # Unrealized P&L = (1.1100 - 1.1000) * 0.1 * 100000 = $100
        expected = 10000 + 100
        assert abs(trader.get_equity() - expected) < 0.01
    
    def test_equity_with_long_position_loss(self):
        """Test equity with losing long position."""
        trader = Trader(mode="paper")

        trader._paper_positions["EURUSD"] = {
            "volume": 0.1,
            "entry_price": 1.1000,
            "current_price": 1.0900,  # -100 pips
            "side": "BUY"
        }

        # Unrealized P&L = (1.0900 - 1.1000) * 0.1 * 100000 = -$100
        expected = 10000 - 100
        assert abs(trader.get_equity() - expected) < 0.01
    
    def test_equity_with_short_position_profit(self):
        """Test equity with profitable short position."""
        trader = Trader(mode="paper")

        trader._paper_positions["EURUSD"] = {
            "volume": -0.1,  # Negative for short
            "entry_price": 1.1000,
            "current_price": 1.0900,  # Price down = profit
            "side": "SELL"
        }

        # Unrealized P&L = (1.1000 - 1.0900) * 0.1 * 100000 = $100
        expected = 10000 + 100
        assert abs(trader.get_equity() - expected) < 0.01


class TestPaperPositions:
    """Test paper trading positions."""
    
    def test_get_positions_empty(self):
        """Test getting positions when none exist."""
        trader = Trader(mode="paper")
        positions = trader.get_positions()
        
        assert positions == {}
    
    def test_get_positions_with_data(self):
        """Test getting positions with existing positions."""
        trader = Trader(mode="paper")
        
        trader._paper_positions = {
            "EURUSD": {"volume": 0.1},
            "GBPUSD": {"volume": -0.2}
        }
        
        positions = trader.get_positions()
        
        assert positions == {"EURUSD": 0.1, "GBPUSD": -0.2}


class TestPaperOrders:
    """Test paper order execution."""
    
    def test_place_buy_order(self):
        """Test placing a buy order."""
        trader = Trader(mode="paper")
        
        result = trader.place_order(
            symbol="EURUSD",
            side="BUY",
            volume=0.1,
            stop_loss=1.08,
            take_profit=1.15
        )
        
        assert result is True
        assert "EURUSD" in trader._paper_positions
        assert trader._paper_positions["EURUSD"]["volume"] > 0
        assert trader._paper_positions["EURUSD"]["side"] == "BUY"
    
    def test_place_sell_order(self):
        """Test placing a sell order."""
        trader = Trader(mode="paper")
        
        result = trader.place_order(
            symbol="EURUSD",
            side="SELL",
            volume=0.1
        )
        
        assert result is True
        assert "EURUSD" in trader._paper_positions
        assert trader._paper_positions["EURUSD"]["volume"] < 0
        assert trader._paper_positions["EURUSD"]["side"] == "SELL"
    
    def test_order_records_entry_time(self):
        """Test that order records entry time."""
        trader = Trader(mode="paper")
        
        before = datetime.now(timezone.utc)
        trader.place_order("EURUSD", "BUY", 0.1)
        after = datetime.now(timezone.utc)
        
        entry_time = trader._paper_positions["EURUSD"]["entry_time"]
        assert before <= entry_time <= after
    
    def test_close_position_realizes_pnl(self):
        """Test that closing position realizes P&L."""
        trader = Trader(mode="paper")
        
        # Open long at 1.10
        trader._paper_positions["EURUSD"] = {
            "volume": 0.1,
            "entry_price": 1.1000,
            "current_price": 1.1100,
            "side": "BUY"
        }
        
        # "Close" by opening opposite (sell)
        # Note: place_order calculates from _get_paper_price which uses current_price
        initial_pnl = trader._paper_pnl
        trader.place_order("EURUSD", "SELL", 0.1)
        
        # P&L should have increased
        assert trader._paper_pnl > initial_pnl


class TestUpdatePrices:
    """Test price update functionality."""
    
    def test_update_paper_prices(self):
        """Test updating paper position prices."""
        trader = Trader(mode="paper")
        
        trader._paper_positions["EURUSD"] = {
            "volume": 0.1,
            "entry_price": 1.1000,
            "current_price": 1.1000
        }
        
        trader.update_paper_prices({"EURUSD": 1.1050})
        
        assert trader._paper_positions["EURUSD"]["current_price"] == 1.1050
    
    def test_update_prices_ignores_non_positions(self):
        """Test that price update ignores symbols without positions."""
        trader = Trader(mode="paper")
        
        # No exception should be raised
        trader.update_paper_prices({"EURUSD": 1.1050})
        
        assert "EURUSD" not in trader._paper_positions

