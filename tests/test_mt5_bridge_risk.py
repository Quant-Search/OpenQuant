import unittest
from unittest.mock import MagicMock, patch
import sys

# Import the module under test
from openquant.paper.mt5_bridge import apply_allocation_to_mt5, modify_position
import openquant.paper.mt5_bridge as bridge_module

class TestMT5BridgeRisk(unittest.TestCase):
    def setUp(self):
        # Create a fresh mock for each test
        self.mt5_mock = MagicMock()
        
        # Patch sys.modules to return our mock when MetaTrader5 is imported
        self.modules_patcher = patch.dict(sys.modules, {"MetaTrader5": self.mt5_mock})
        self.modules_patcher.start()
        
        # Reset the global _MT5 in the bridge module to force re-import
        bridge_module._MT5 = None
        
        # Configure common mock behaviors
        self.mt5_mock.initialize.return_value = True
        self.mt5_mock.login.return_value = True
        self.mt5_mock.symbol_info.return_value = MagicMock(trade_contract_size=1.0, volume_min=0.01, volume_step=0.01, volume_max=100.0)
        self.mt5_mock.symbol_info_tick.return_value = MagicMock(ask=1.0, bid=1.0, last=1.0)
        self.mt5_mock.positions_get.return_value = []
        self.mt5_mock.account_info.return_value = MagicMock(equity=10000.0)
        
        # Mock constants
        self.mt5_mock.ORDER_TYPE_BUY = 0
        self.mt5_mock.ORDER_TYPE_SELL = 1
        self.mt5_mock.TRADE_ACTION_DEAL = 1
        self.mt5_mock.TRADE_ACTION_SLTP = 6
        self.mt5_mock.TRADE_RETCODE_DONE = 10009
        
        # Mock alerts
        self.alert_patcher = patch("openquant.utils.alerts.send_alert")
        self.mock_send_alert = self.alert_patcher.start()

    def tearDown(self):
        self.modules_patcher.stop()
        self.alert_patcher.stop()

    def test_order_with_sltp(self):
        # Setup: Allocation with SL/TP
        allocation = [{"symbol": "EURUSD", "weight": 0.1, "sl": 0.95, "tp": 1.05}]
        
        # Mock order_send success
        success_res = MagicMock()
        success_res.retcode = self.mt5_mock.TRADE_RETCODE_DONE
        success_res.price = 1.0
        self.mt5_mock.order_send.return_value = success_res
        
        # Run
        apply_allocation_to_mt5(allocation)
        
        # Verify order_send called with SL/TP
        # We expect one call. Get the args.
        args, _ = self.mt5_mock.order_send.call_args
        req = args[0]
        
        self.assertEqual(req["sl"], 0.95)
        self.assertEqual(req["tp"], 1.05)
        self.assertEqual(req["action"], self.mt5_mock.TRADE_ACTION_DEAL)

    def test_modify_position_success(self):
        # Setup: Mock existing position
        mock_pos = MagicMock()
        mock_pos.ticket = 12345
        mock_pos.sl = 0.0
        mock_pos.tp = 0.0
        self.mt5_mock.positions_get.return_value = [mock_pos]
        
        # Mock order_send success
        success_res = MagicMock()
        success_res.retcode = self.mt5_mock.TRADE_RETCODE_DONE
        self.mt5_mock.order_send.return_value = success_res
        
        # Run
        res = modify_position("EURUSD", sl=0.98, tp=1.02)
        
        # Verify
        self.assertTrue(res)
        args, _ = self.mt5_mock.order_send.call_args
        req = args[0]
        self.assertEqual(req["action"], self.mt5_mock.TRADE_ACTION_SLTP)
        self.assertEqual(req["position"], 12345)
        self.assertEqual(req["sl"], 0.98)
        self.assertEqual(req["tp"], 1.02)

    def test_modify_position_fail(self):
        # Setup: Mock existing position
        mock_pos = MagicMock()
        mock_pos.ticket = 12345
        self.mt5_mock.positions_get.return_value = [mock_pos]
        
        # Mock order_send failure
        fail_res = MagicMock()
        fail_res.retcode = 10001
        fail_res.comment = "Invalid SL"
        self.mt5_mock.order_send.return_value = fail_res
        
        # Run
        res = modify_position("EURUSD", sl=0.98)
        
        # Verify
        self.assertFalse(res)
        self.mock_send_alert.assert_called_with(
            subject="MT5 Modify Failed: EURUSD",
            body="Modify SL/TP failed. Code: 10001, Comment: Invalid SL",
            severity="ERROR"
        )

if __name__ == "__main__":
    unittest.main()
