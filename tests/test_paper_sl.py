import unittest
from openquant.paper.state import PortfolioState
from openquant.paper.simulator import execute_orders, check_exits, MarketSnapshot

class TestPaperStopLoss(unittest.TestCase):
    def test_sl_execution(self):
        # 1. Setup State
        state = PortfolioState()
        key = ("BINANCE", "BTC/USDT", "1h", "mom")
        
        # 2. Open Position with SL
        # Order: (key, delta, price, sl, tp)
        orders = [(key, 1.0, 50000.0, 49000.0, 55000.0)]
        execute_orders(state, orders)
        
        self.assertEqual(state.position(key), 1.0)
        self.assertEqual(state.avg_price[key], 50000.0)
        self.assertEqual(state.sl_levels[key], 49000.0)
        
        # 3. Check Exits - Price above SL
        snap = MarketSnapshot(prices={key: 49500.0})
        exits = check_exits(state, snap)
        self.assertEqual(len(exits), 0)
        
        # 4. Check Exits - Price below SL
        snap = MarketSnapshot(prices={key: 48900.0})
        exits = check_exits(state, snap)
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0][0], key)
        self.assertEqual(exits[0][1], -1.0) # Close 1.0 units
        
        # 5. Execute Exit
        execute_orders(state, exits)
        self.assertEqual(state.position(key), 0.0)
        # Aux data should be cleared
        self.assertNotIn(key, state.avg_price)
        self.assertNotIn(key, state.sl_levels)

    def test_tp_execution(self):
        state = PortfolioState()
        key = ("BINANCE", "ETH/USDT", "1h", "meanrev")
        
        # Short position: Sell 10 @ 3000, SL 3100, TP 2800
        orders = [(key, -10.0, 3000.0, 3100.0, 2800.0)]
        execute_orders(state, orders)
        
        self.assertEqual(state.position(key), -10.0)
        
        # Price drops to 2700 (hit TP for short)
        snap = MarketSnapshot(prices={key: 2700.0})
        exits = check_exits(state, snap)
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0][1], 10.0) # Buy back 10
        
        execute_orders(state, exits)
        self.assertEqual(state.position(key), 0.0)

if __name__ == "__main__":
    unittest.main()
