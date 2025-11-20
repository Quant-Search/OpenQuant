"""
Alpaca Broker Implementation.
"""
import os
from typing import Dict, List, Any, Optional
from openquant.broker.abstract import Broker

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

class AlpacaBroker(Broker):
    """
    Alpaca Broker implementation using alpaca-py SDK.
    """
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        if not ALPACA_AVAILABLE:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")
            
        self.api_key = api_key or os.getenv("APCA_API_KEY_ID")
        self.secret_key = secret_key or os.getenv("APCA_API_SECRET_KEY")
        self.paper = paper
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials missing.")
            
        self.client = TradingClient(self.api_key, self.secret_key, paper=self.paper)

    def get_cash(self) -> float:
        account = self.client.get_account()
        return float(account.cash)

    def get_equity(self) -> float:
        account = self.client.get_account()
        return float(account.equity)

    def get_positions(self) -> Dict[str, float]:
        positions = self.client.get_all_positions()
        pos_dict = {}
        for p in positions:
            pos_dict[p.symbol] = float(p.qty)
        return pos_dict

    def place_order(self, 
                   symbol: str, 
                   quantity: float, 
                   side: str, 
                   order_type: str = "market", 
                   limit_price: Optional[float] = None) -> Dict[str, Any]:
        
        alpaca_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        
        if order_type.lower() == "market":
            req = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY
            )
        elif order_type.lower() == "limit":
            if limit_price is None:
                raise ValueError("Limit price required for limit orders")
            req = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=alpaca_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
            
        order = self.client.submit_order(order_data=req)
        return {"id": str(order.id), "status": order.status}

    def close_all_positions(self):
        self.client.close_all_positions(cancel_orders=True)
