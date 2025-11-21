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
        
        # Initialize TCA
        from openquant.analysis.tca import TCAMonitor
        self.tca = TCAMonitor()

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
        
        # Capture arrival price (snapshot) for TCA
        arrival_price = 0.0
        try:
            # Try to get a quick quote if possible, or use last trade
            # For now, we assume 0.0 if we can't get it easily without data API
            # In a real system, we'd pass the decision price from the strategy
            pass 
        except Exception:
            pass

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
            arrival_price = limit_price # For limit orders, decision price is limit
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
            
        order = self.client.submit_order(order_data=req)
        
        # Log to TCA
        # If market order, we might not have arrival price yet. 
        # Ideally, the strategy passes the price it saw.
        self.tca.log_order(
            order_id=str(order.id),
            symbol=symbol,
            side=side,
            quantity=quantity,
            arrival_price=arrival_price
        )
        
        return {"id": str(order.id), "status": order.status}

    def close_all_positions(self):
        self.client.close_all_positions(cancel_orders=True)

    def sync_tca(self):
        """Check for filled orders and update TCA records."""
        try:
            # Get closed orders from today
            # In a real high-volume system, we'd filter by time or ID
            orders = self.client.get_orders(filter=GetOrdersRequest(status="closed", limit=50))
            for o in orders:
                if o.status == "filled":
                    self.tca.update_fill(
                        order_id=str(o.id),
                        fill_price=float(o.filled_avg_price) if o.filled_avg_price else 0.0,
                        fill_qty=float(o.filled_qty),
                        fee=0.0 # Alpaca paper has 0 fees usually, but we could calc estimated
                    )
        except Exception as e:
            print(f"TCA Sync Error: {e}")
