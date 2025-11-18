import unittest
from unittest.mock import MagicMock, patch
import sys

# Import the module under test
from openquant.paper.mt5_bridge import apply_allocation_to_mt5
import openquant.paper.mt5_bridge as bridge_module

class TestMT5BridgeAlerts(unittest.TestCase):
    def setUp(self):
        # Create a fresh mock for each test
        self.mt5_mock = MagicMock()
        
        # Patch sys.modules
        self.modules_patcher = patch.dict(sys.modules, {"MetaTrader5": self.mt5_mock})
        self.modules_patcher.start()
        
        # Reset the global _MT5
        bridge_module._MT5 = None
        
        # Configure common mock behaviors
        self.mt5_mock.initialize.return_value = True
        self.mt5_mock.login.return_value = True
        self.mt5_mock.symbol_info.return_value = MagicMock(trade_contract_size=1.0, volume_min=0.01, volume_step=0.01, volume_max=100.0)
        self.mt5_mock.symbol_info_tick.return_value = MagicMock(ask=1.0, bid=1.0, last=1.0)
        self.mt5_mock.positions_get.return_value = []
        self.mt5_mock.account_info.return_value = MagicMock(equity=10000.0)
        self.mt5_mock.ORDER_TYPE_BUY = 0
        self.mt5_mock.ORDER_TYPE_SELL = 1
        self.mt5_mock.TRADE_RETCODE_DONE = 10009
        
        # Mock alerts
        self.alert_patcher = patch("openquant.utils.alerts.send_alert")
        self.mock_send_alert = self.alert_patcher.start()

    def tearDown(self):
        self.modules_patcher.stop()
        self.alert_patcher.stop()
        mt5.login.return_value = True
        mt5.symbol_info.return_value = MagicMock(trade_contract_size=1.0, volume_min=0.01, volume_step=0.01, volume_max=100.0)
        mt5.symbol_info_tick.return_value = MagicMock(ask=1.0, bid=1.0, last=1.0)
        mt5.positions_get.return_value = []
        mt5.account_info.return_value = MagicMock(equity=10000.0)
        mt5.ORDER_TYPE_BUY = 0
        mt5.ORDER_TYPE_SELL = 1
        
        # Mock alerts
        self.alert_patcher = patch("openquant.utils.alerts.send_alert")
        self.mock_send_alert = self.alert_patcher.start()

    def tearDown(self):
        self.alert_patcher.stop()

    def test_order_rejection_alert(self):
        # Setup: Allocation that triggers a buy
        allocation = [{"symbol": "EURUSD", "weight": 0.1}]
        
        # Mock order_send to fail
        fail_res = MagicMock()
        fail_res.retcode = 10001  # Not TRADE_RETCODE_DONE (10009)
        fail_res.comment = "General error"
        self.mt5_mock.order_send.return_value = fail_res
        
        # Run
        apply_allocation_to_mt5(allocation)
        
        # Verify alert
        # Volume = (0.1 * 10000) / 1.0 = 1000.0. Max volume is 100.0. So volume should be 100.0.
        self.mock_send_alert.assert_called_with(
            subject="MT5 Order Failed: EURUSD",
            body="Order 0 100.0 failed. Code: 10001, Comment: General error",
            severity="ERROR"
        )

    def test_high_slippage_alert(self):
        # Setup: Allocation that triggers a buy
        allocation = [{"symbol": "EURUSD", "weight": 0.1}]
        
        # Mock tick price (expected)
        self.mt5_mock.symbol_info_tick.return_value = MagicMock(ask=1.0, bid=1.0)
        
        # Mock order_send to succeed but with high slippage (1.05 vs 1.00 = 5%)
        success_res = MagicMock()
        success_res.retcode = self.mt5_mock.TRADE_RETCODE_DONE
        success_res.price = 1.05
        self.mt5_mock.order_send.return_value = success_res
        
        # Run
        apply_allocation_to_mt5(allocation)
        
        # Verify alert
        self.mock_send_alert.assert_called_with(
            subject="MT5 High Slippage: EURUSD",
            body="Slippage 5.0000%. Expected 1.0, Got 1.05",
            severity="WARNING"
        )

    def test_normal_execution_no_alert(self):
        # Setup: Allocation that triggers a buy
        allocation = [{"symbol": "EURUSD", "weight": 0.1}]
        
        # Mock tick price
        self.mt5_mock.symbol_info_tick.return_value = MagicMock(ask=1.0, bid=1.0)
        
        # Mock order_send to succeed with matching price
        success_res = MagicMock()
        success_res.retcode = self.mt5_mock.TRADE_RETCODE_DONE
        success_res.price = 1.0
        self.mt5_mock.order_send.return_value = success_res
        
        # Run
        apply_allocation_to_mt5(allocation)
        
        # Verify NO alert
        self.mock_send_alert.assert_not_called()

if __name__ == "__main__":
    unittest.main()
