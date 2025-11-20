"""
Test for Alpaca Broker (Mocked).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock alpaca modules BEFORE importing the broker
# This ensures the try/except block in alpaca_broker.py succeeds
mock_alpaca = MagicMock()
sys.modules["alpaca"] = mock_alpaca
sys.modules["alpaca.trading"] = mock_alpaca.trading
sys.modules["alpaca.trading.client"] = mock_alpaca.trading.client
sys.modules["alpaca.trading.requests"] = mock_alpaca.trading.requests
sys.modules["alpaca.trading.enums"] = mock_alpaca.trading.enums

# Define the classes that are imported
mock_alpaca.trading.client.TradingClient = MagicMock()
mock_alpaca.trading.requests.MarketOrderRequest = MagicMock()
mock_alpaca.trading.requests.LimitOrderRequest = MagicMock()
mock_alpaca.trading.enums.OrderSide = MagicMock()
mock_alpaca.trading.enums.TimeInForce = MagicMock()

sys.path.append(str(Path(__file__).parent.parent))
from openquant.broker.alpaca_broker import AlpacaBroker

def test_alpaca_broker():
    print("\n--- Testing Alpaca Broker (Mock) ---")
    
    # Setup Mock Client Instance
    mock_client_instance = MagicMock()
    
    # Mock Account
    mock_account = MagicMock()
    mock_account.cash = "100000.00"
    mock_account.equity = "105000.00"
    mock_client_instance.get_account.return_value = mock_account
    
    # Mock Positions
    p1 = MagicMock()
    p1.symbol = "AAPL"
    p1.qty = "10"
    mock_client_instance.get_all_positions.return_value = [p1]
    
    # Mock Order
    mock_order = MagicMock()
    mock_order.id = "12345"
    mock_order.status = "accepted"
    mock_client_instance.submit_order.return_value = mock_order
    
    # Patch the TradingClient class inside the module to return our mock instance
    with patch("openquant.broker.alpaca_broker.TradingClient", return_value=mock_client_instance):
        # Initialize Broker
        broker = AlpacaBroker(api_key="test", secret_key="test", paper=True)
        
        # 1. Test Cash/Equity
        cash = broker.get_cash()
        equity = broker.get_equity()
        print(f"Cash: {cash}, Equity: {equity}")
        assert cash == 100000.0
        assert equity == 105000.0
        
        # 2. Test Positions
        pos = broker.get_positions()
        print(f"Positions: {pos}")
        assert pos["AAPL"] == 10.0
        
        # 3. Test Place Order
        print("Placing Order...")
        res = broker.place_order("TSLA", 5, "buy", "market")
        print(f"Order Result: {res}")
        assert res["id"] == "12345"
        assert res["status"] == "accepted"
        
        # Verify calls
        mock_client_instance.submit_order.assert_called_once()
    
    print("\n✅ Alpaca Broker Test Passed!")

if __name__ == "__main__":
    test_alpaca_broker()
