"""
Broker Abstraction Layer.
Defines the interface for all broker implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class Broker(ABC):
    """
    Abstract Base Class for Brokers (Alpaca, MT5, IBKR, etc.)
    """
    
    @abstractmethod
    def get_cash(self) -> float:
        """Return available cash balance."""
        pass

    @abstractmethod
    def get_equity(self) -> float:
        """Return total account equity (cash + positions)."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """
        Return current positions.
        Format: {"BTCUSD": 0.5, "AAPL": 10}
        """
        pass

    @abstractmethod
    def place_order(self, 
                   symbol: str, 
                   quantity: float, 
                   side: str, 
                   order_type: str = "market", 
                   limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an order.
        side: 'buy' or 'sell'
        order_type: 'market', 'limit', 'stop'
        """
        pass

    @abstractmethod
    def close_all_positions(self):
        """Liquidate all positions immediately."""
        pass
