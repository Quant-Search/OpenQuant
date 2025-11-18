import unittest
import shutil
import json
from pathlib import Path
from datetime import date, timedelta
from openquant.risk.portfolio_guard import PortfolioGuard

class TestPortfolioGuard(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_risk")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.test_dir / "risk_state.json"
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        guard = PortfolioGuard(self.state_file)
        self.assertEqual(guard.high_water_mark, 0.0)
        self.assertEqual(guard.start_day_equity, 0.0)

    def test_hwm_update(self):
        guard = PortfolioGuard(self.state_file)
        
        # Day 1: Start with 100k
        is_safe, _ = guard.on_cycle_start(100000.0)
        self.assertTrue(is_safe)
        self.assertEqual(guard.high_water_mark, 100000.0)
        
        # Profit: 110k
        is_safe, _ = guard.on_cycle_start(110000.0)
        self.assertTrue(is_safe)
        self.assertEqual(guard.high_water_mark, 110000.0)
        
        # Loss: 105k (HWM stays 110k)
        is_safe, _ = guard.on_cycle_start(105000.0)
        self.assertTrue(is_safe)
        self.assertEqual(guard.high_water_mark, 110000.0)

    def test_drawdown_breach(self):
        guard = PortfolioGuard(self.state_file)
        guard.update_config({"dd_limit": 0.10}) # 10% limit
        
        # Peak 100k
        guard.on_cycle_start(100000.0)
        
        # Drop to 89k (11% DD)
        is_safe, msg = guard.on_cycle_start(89000.0)
        self.assertFalse(is_safe)
        self.assertIn("MAX DRAWDOWN BREACHED", msg)

    def test_daily_loss_breach(self):
        guard = PortfolioGuard(self.state_file)
        guard.update_config({"daily_loss_cap": 0.05}) # 5% limit
        
        # Start day 100k
        guard.on_cycle_start(100000.0)
        
        # Drop to 94k (6% loss)
        is_safe, msg = guard.on_cycle_start(94000.0)
        self.assertFalse(is_safe)
        self.assertIn("DAILY LOSS CAP BREACHED", msg)

    def test_daily_reset(self):
        guard = PortfolioGuard(self.state_file)
        
        # Day 1: 100k -> 96k (4% loss, safe)
        guard.on_cycle_start(100000.0)
        guard.on_cycle_start(96000.0)
        self.assertEqual(guard.start_day_equity, 100000.0)
        
        # Force new day
        guard.current_date = date.today() - timedelta(days=1)
        
        # Day 2 Start: 96k. This should reset start_day_equity to 96k
        guard.on_cycle_start(96000.0)
        self.assertEqual(guard.start_day_equity, 96000.0)
        
        # Day 2 Loss: 96k -> 95k (1% loss from 96k, safe)
        is_safe, _ = guard.on_cycle_start(95000.0)
        self.assertTrue(is_safe)

    def test_cvar_calculation(self):
        guard = PortfolioGuard(self.state_file)
        # Add 20 returns, mostly small, one big loss
        for _ in range(19):
            guard.record_daily_return(0.01)
        guard.record_daily_return(-0.10) # 10% loss
        
        # 95% CVaR should pick up the tail
        cvar = guard.calculate_cvar(0.95)
        self.assertGreater(cvar, 0.05)

    def test_cvar_breach(self):
        guard = PortfolioGuard(self.state_file)
        guard.update_config({"cvar_limit": 0.05})
        
        # Simulate history with heavy losses
        for _ in range(30):
            guard.record_daily_return(-0.06) # Consistent 6% loss
            
        is_safe, msg = guard.on_cycle_start(100000.0)
        self.assertFalse(is_safe)
        self.assertIn("CVaR LIMIT BREACHED", msg)

    def test_exposure_check(self):
        guard = PortfolioGuard(self.state_file)
        guard.update_config({"max_exposure_per_symbol": 0.20})
        
        # Safe portfolio
        positions = {"AAPL": 15000, "GOOG": 10000} # 15%, 10%
        equity = 100000.0
        is_safe, msg = guard.check_exposure(positions, equity)
        self.assertTrue(is_safe)
        
        # Unsafe portfolio
        positions = {"AAPL": 25000, "GOOG": 10000} # 25% > 20%
        is_safe, msg = guard.check_exposure(positions, equity)
        self.assertFalse(is_safe)
        self.assertIn("EXPOSURE LIMIT BREACHED", msg)
        self.assertIn("AAPL", msg)

if __name__ == "__main__":
    unittest.main()
