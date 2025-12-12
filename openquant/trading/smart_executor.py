"""Smart Order Execution.

Optimized order execution with:
- Limit orders instead of market orders
- TWAP (Time-Weighted Average Price)
- Slippage reduction
- Order retry logic
"""
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import threading

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    LIMIT_IOC = "limit_ioc"  # Immediate or Cancel
    TWAP = "twap"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class OrderResult:
    """Result of order execution."""
    order_id: str
    status: OrderStatus
    filled_qty: float
    avg_price: float
    slippage_bps: float
    execution_time_ms: float
    
@dataclass
class ExecutionConfig:
    """Configuration for order execution."""
    order_type: OrderType = OrderType.LIMIT
    limit_offset_bps: float = 2.0  # Place limit order 2bps better than current
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0
    twap_slices: int = 5
    twap_interval_seconds: float = 60.0

class SmartExecutor:
    """
    Smart order execution engine.
    
    Features:
    - Limit orders with offset for better fills
    - Automatic retry on partial fills
    - TWAP for large orders
    - Slippage tracking
    """
    
    def __init__(
        self,
        broker,  # MT5Broker or similar
        config: Optional[ExecutionConfig] = None
    ):
        self.broker = broker
        self.config = config or ExecutionConfig()
        self._orders: Dict[str, Dict] = {}
        
    def execute(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: float,
        current_price: float,
        **kwargs
    ) -> OrderResult:
        """
        Execute an order with smart execution.
        
        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Order quantity
            current_price: Current market price
            
        Returns:
            OrderResult with execution details
        """
        start_time = time.time()
        
        order_type = kwargs.get("order_type", self.config.order_type)
        
        if order_type == OrderType.TWAP:
            return self._execute_twap(symbol, side, quantity, current_price)
        elif order_type == OrderType.LIMIT:
            return self._execute_limit(symbol, side, quantity, current_price)
        else:
            return self._execute_market(symbol, side, quantity, current_price)
            
    def _execute_market(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float
    ) -> OrderResult:
        """Execute market order."""
        start_time = time.time()
        
        try:
            LOGGER.info(f"Executing MARKET {side} {quantity} {symbol} @ ~{current_price:.5f}")
            
            result = self.broker.place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="MARKET"
            )
            
            if result.get("success"):
                fill_price = result.get("fill_price", current_price)
                slippage = abs(fill_price - current_price) / current_price * 10000
                
                return OrderResult(
                    order_id=result.get("order_id", ""),
                    status=OrderStatus.FILLED,
                    filled_qty=quantity,
                    avg_price=fill_price,
                    slippage_bps=slippage,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            else:
                return OrderResult(
                    order_id="",
                    status=OrderStatus.FAILED,
                    filled_qty=0,
                    avg_price=0,
                    slippage_bps=0,
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            LOGGER.error(f"Market order failed: {e}")
            return OrderResult(
                order_id="",
                status=OrderStatus.FAILED,
                filled_qty=0,
                avg_price=0,
                slippage_bps=0,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
    def _execute_limit(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float
    ) -> OrderResult:
        """Execute limit order with retry logic."""
        start_time = time.time()
        
        # Calculate limit price with offset
        offset = current_price * self.config.limit_offset_bps / 10000
        if side.upper() == "BUY":
            # Buy at slightly lower price
            limit_price = current_price - offset
        else:
            # Sell at slightly higher price
            limit_price = current_price + offset
            
        remaining_qty = quantity
        total_filled = 0.0
        total_cost = 0.0
        order_id = ""
        
        for attempt in range(self.config.max_retries):
            try:
                LOGGER.info(f"LIMIT {side} {remaining_qty} {symbol} @ {limit_price:.5f} (attempt {attempt + 1})")
                
                result = self.broker.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=remaining_qty,
                    order_type="LIMIT",
                    limit_price=limit_price
                )
                
                if result.get("success"):
                    filled = result.get("filled_qty", remaining_qty)
                    fill_price = result.get("fill_price", limit_price)
                    
                    total_filled += filled
                    total_cost += filled * fill_price
                    order_id = result.get("order_id", order_id)
                    
                    if filled >= remaining_qty * 0.99:  # Essentially complete
                        break
                        
                    remaining_qty -= filled
                    
                    # Adjust limit price and retry
                    if side.upper() == "BUY":
                        limit_price = min(limit_price + offset, current_price)
                    else:
                        limit_price = max(limit_price - offset, current_price)
                        
                time.sleep(self.config.retry_delay_seconds)
                
            except Exception as e:
                LOGGER.warning(f"Limit order attempt {attempt + 1} failed: {e}")
                time.sleep(self.config.retry_delay_seconds)
                
        # Determine final status
        if total_filled >= quantity * 0.99:
            status = OrderStatus.FILLED
        elif total_filled > 0:
            status = OrderStatus.PARTIAL
        else:
            status = OrderStatus.FAILED
            
        avg_price = total_cost / total_filled if total_filled > 0 else 0
        slippage = abs(avg_price - current_price) / current_price * 10000 if total_filled > 0 else 0
        
        return OrderResult(
            order_id=order_id,
            status=status,
            filled_qty=total_filled,
            avg_price=avg_price,
            slippage_bps=slippage,
            execution_time_ms=(time.time() - start_time) * 1000
        )
        
    def _execute_twap(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float
    ) -> OrderResult:
        """
        Execute using TWAP (Time-Weighted Average Price).
        
        Splits order into slices executed over time.
        """
        start_time = time.time()
        
        slices = self.config.twap_slices
        slice_qty = quantity / slices
        interval = self.config.twap_interval_seconds
        
        total_filled = 0.0
        total_cost = 0.0
        
        LOGGER.info(f"TWAP {side} {quantity} {symbol} in {slices} slices over {slices * interval}s")
        
        for i in range(slices):
            # Get current market price for this slice
            try:
                tick = self.broker.get_tick(symbol)
                slice_price = tick.get("bid" if side.upper() == "SELL" else "ask", current_price)
            except:
                slice_price = current_price
                
            # Execute slice
            result = self._execute_limit(symbol, side, slice_qty, slice_price)
            
            total_filled += result.filled_qty
            total_cost += result.filled_qty * result.avg_price
            
            LOGGER.info(f"TWAP slice {i + 1}/{slices}: filled {result.filled_qty:.4f} @ {result.avg_price:.5f}")
            
            if i < slices - 1:
                time.sleep(interval)
                
        avg_price = total_cost / total_filled if total_filled > 0 else 0
        slippage = abs(avg_price - current_price) / current_price * 10000 if total_filled > 0 else 0
        
        status = OrderStatus.FILLED if total_filled >= quantity * 0.95 else OrderStatus.PARTIAL
        
        return OrderResult(
            order_id=f"twap_{int(start_time)}",
            status=status,
            filled_qty=total_filled,
            avg_price=avg_price,
            slippage_bps=slippage,
            execution_time_ms=(time.time() - start_time) * 1000
        )
        
    def calculate_optimal_order_type(
        self,
        symbol: str,
        quantity: float,
        avg_volume: float,
        volatility: float
    ) -> OrderType:
        """
        Determine optimal order type based on conditions.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            avg_volume: Average daily volume
            volatility: Current volatility
            
        Returns:
            Recommended OrderType
        """
        # Large orders relative to volume -> TWAP
        if avg_volume > 0 and quantity > avg_volume * 0.01:
            LOGGER.info(f"{symbol}: Large order ({quantity}/{avg_volume*0.01:.0f}), using TWAP")
            return OrderType.TWAP
            
        # High volatility -> Limit with tight offset
        if volatility > 0.03:
            LOGGER.info(f"{symbol}: High vol ({volatility:.2%}), using LIMIT")
            return OrderType.LIMIT
            
        # Low volatility, small order -> Market is fine
        if volatility < 0.01 and quantity < avg_volume * 0.001:
            return OrderType.MARKET
            
        # Default to limit
        return OrderType.LIMIT
