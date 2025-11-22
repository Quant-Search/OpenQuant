import unittest
from openquant.paper.state import PortfolioState
from openquant.paper.simulator import check_daily_loss, MarketSnapshot

class TestDailyLoss(unittest.TestCase):
    def test_daily_reset(self):
        state = PortfolioState(cash=100000.0)
        # Day 1
        state.check_daily_reset("2025-01-01", 100000.0)
        self.assertEqual(state.daily_start_equity, 100000.0)
        self.assertEqual(state.last_reset_date, "2025-01-01")
        
        # Same Day - No Reset
        state.check_daily_reset("2025-01-01", 90000.0) # Lost 10k
        self.assertEqual(state.daily_start_equity, 100000.0)
        
        # Day 2 - Reset
        state.check_daily_reset("2025-01-02", 90000.0)
        self.assertEqual(state.daily_start_equity, 90000.0)
        self.assertEqual(state.last_reset_date, "2025-01-02")

    def test_limit_check(self):
        state = PortfolioState(cash=100000.0)
        state.daily_start_equity = 100000.0
        snap = MarketSnapshot(prices={})
        
        # No loss
        self.assertFalse(check_daily_loss(state, snap, 0.02))
        
        # Small loss (1%)
        state.cash = 99000.0
        self.assertFalse(check_daily_loss(state, snap, 0.02))
        
        # Big loss (3%)
        state.cash = 97000.0
        self.assertTrue(check_daily_loss(state, snap, 0.02))

if __name__ == "__main__":
    unittest.main()
