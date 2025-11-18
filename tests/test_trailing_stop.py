import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock MetaTrader5 before importing
sys.modules["MetaTrader5"] = MagicMock()

from openquant.risk.trailing_stop import TrailingStopManager, PositionInfo


class TestTrailingStopManager(unittest.TestCase):
    """Test the TrailingStopManager class.
    
    These tests verify the mathematical calculations for trailing stop logic:
    - Long positions: SL trails below the price
    - Short positions: SL trails above the price
    - Activation threshold enforcement
    - Minimum update threshold
    """
    
    def test_long_position_trailing(self):
        """Test trailing stop for a LONG position.
        
        Mathematical scenario:
        - Entry: 1.0000
        - Current price: 1.1000 (10% profit = 1000 bps)
        - Trailing: 100 bps (1%)
        - Expected new SL: 1.1000 × (1 - 0.01) = 1.0890
        - Current SL: 1.0000
        - Should update: Yes (1.0890 > 1.0000)
        """
        manager = TrailingStopManager(trailing_bps=100, activation_bps=0, min_update_bps=0)
        
        pos = PositionInfo(
            symbol="EURUSD",
            ticket=12345,
            volume=1.0,
            type=0,  # Long
            sl=1.0000,
            tp=0.0,
            open_price=1.0000
        )
        
        current_price = 1.1000
        new_sl = manager.calculate_new_sl(pos, current_price)
        
        # Expected: 1.1000 * (1 - 100/10000) = 1.1000 * 0.99 = 1.0890
        self.assertIsNotNone(new_sl)
        self.assertAlmostEqual(new_sl, 1.0890, places=4)
        
    def test_short_position_trailing(self):
        """Test trailing stop for a SHORT position.
        
        Mathematical scenario:
        - Entry: 1.1000
        - Current price: 1.0000 (9.09% profit ≈ 909 bps)
        - Trailing: 100 bps (1%)
        - Expected new SL: 1.0000 × (1 + 0.01) = 1.0100
        - Current SL: 1.1500
        - Should update: Yes (1.0100 < 1.1500, tightening)
        """
        manager = TrailingStopManager(trailing_bps=100, activation_bps=0, min_update_bps=0)
        
        pos = PositionInfo(
            symbol="EURUSD",
            ticket=12345,
            volume=1.0,
            type=1,  # Short
            sl=1.1500,
            tp=0.0,
            open_price=1.1000
        )
        
        current_price = 1.0000
        new_sl = manager.calculate_new_sl(pos, current_price)
        
        # Expected: 1.0000 * (1 + 100/10000) = 1.0000 * 1.01 = 1.0100
        self.assertIsNotNone(new_sl)
        self.assertAlmostEqual(new_sl, 1.0100, places=4)
        
    def test_activation_threshold_not_met(self):
        """Test that trailing doesn't activate until profit threshold is met.
        
        Mathematical scenario:
        - Entry: 1.0000
        - Current price: 1.0050 (only 0.5% = 50 bps profit)
        - Activation threshold: 100 bps
        - Should update: No (50 < 100)
        """
        manager = TrailingStopManager(trailing_bps=100, activation_bps=100, min_update_bps=0)
        
        pos = PositionInfo(
            symbol="EURUSD",
            ticket=12345,
            volume=1.0,
            type=0,  # Long
            sl=1.0000,
            tp=0.0,
            open_price=1.0000
        )
        
        current_price = 1.0050
        new_sl = manager.calculate_new_sl(pos, current_price)
        
        # No update because profit (50 bps) < activation (100 bps)
        self.assertIsNone(new_sl)
        
    def test_min_update_threshold(self):
        """Test that small SL changes are ignored.
        
        Mathematical scenario:
        - Current SL: 1.0890
        - New calculated SL: 1.0895 (improvement: ~4.6 bps)
        - Min update: 5 bps
        - Should update: No (4.6 < 5)
        """
        manager = TrailingStopManager(trailing_bps=100, activation_bps=0, min_update_bps=5)
        
        pos = PositionInfo(
            symbol="EURUSD",
            ticket=12345,
            volume=1.0,
            type=0,  # Long
            sl=1.0890,
            tp=0.0,
            open_price=1.0000
        )
        
        # Price: 1.1045 → new SL would be 1.1045 * 0.99 = 1.09346
        # Improvement: (1.09346 - 1.0890) / 1.0890 * 10000 ≈ 40.9 bps
        # This should trigger update
        current_price = 1.1045
        new_sl = manager.calculate_new_sl(pos, current_price)
        self.assertIsNotNone(new_sl)
        
        # Now test with tiny movement
        # Price: 1.0895 → new SL would be 1.0895 * 0.99 = 1.07860
        # But this is actually worse than current, so would be rejected anyway
        # Let me pick a better example
        pos.sl = 1.0850
        current_price = 1.0955  # new SL = 1.0955 * 0.99 = 1.08455
        # Improvement: (1.08455 - 1.0850) / 1.0850 * 10000 ≈ -4.2 bps
        # Wait, that's negative. Let me recalculate.
        # Actually 1.08455 < 1.0850, so this wouldn't update for a long anyway.
        
        # For a proper test, I need current_price such that new_sl is just barely above current sl
        # new_sl = current_price * 0.99
        # We want: new_sl = current_sl + epsilon
        # So: current_price = (current_sl + epsilon) / 0.99
        current_sl = 1.0850
        epsilon = 0.0001  # Very small improvement
        current_price = (current_sl + epsilon) / 0.99
        new_sl = manager.calculate_new_sl(pos, current_price)
        
        # Improvement in bps: (epsilon / current_sl) * 10000 ≈ 0.92 bps < 5 bps
        self.assertIsNone(new_sl)
        
    def test_long_no_update_when_price_drops(self):
        """Test that LONG position SL doesn't move down when price drops.
        
        Mathematical scenario:
        - Current SL: 1.0890
        - Price drops to 1.0500
        - New calculated SL: 1.0500 × 0.99 = 1.0395
        - Should update: No (1.0395 < 1.0890, would loosen the SL)
        """
        manager = TrailingStopManager(trailing_bps=100, activation_bps=0, min_update_bps=0)
        
        pos = PositionInfo(
            symbol="EURUSD",
            ticket=12345,
            volume=1.0,
            type=0,  # Long
            sl=1.0890,
            tp=0.0,
            open_price=1.0000
        )
        
        current_price = 1.0500  # Price dropped
        new_sl = manager.calculate_new_sl(pos, current_price)
        
        # No update because new SL (1.0395) < current SL (1.0890)
        self.assertIsNone(new_sl)


if __name__ == "__main__":
    unittest.main()
