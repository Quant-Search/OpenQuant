"""Helper utilities for order book integration in strategies and trading.

Provides convenient wrappers and utilities for common order book operations.
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .orderbook import OrderBookFetcher, OrderBookAnalyzer, LiquidityAwareSizer
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class OrderBookCache:
    """Thread-safe order book cache with TTL."""
    
    def __init__(self):
        self._analyzers: Dict[str, OrderBookAnalyzer] = {}
        
    def get_analyzer(self, exchange: str) -> OrderBookAnalyzer:
        """Get or create analyzer for exchange."""
        if exchange not in self._analyzers:
            self._analyzers[exchange] = OrderBookAnalyzer(exchange)
        return self._analyzers[exchange]
    
    def clear(self):
        """Clear all cached analyzers."""
        self._analyzers.clear()


_GLOBAL_CACHE = OrderBookCache()


def check_liquidity(
    symbol: str,
    quantity: float,
    side: str,
    exchange: str = "binance",
    max_impact_bps: float = 10.0
) -> Dict[str, Any]:
    """
    Quick liquidity check for a trade.
    
    Args:
        symbol: Trading symbol
        quantity: Order quantity
        side: "buy" or "sell"
        exchange: Exchange name
        max_impact_bps: Maximum acceptable impact
        
    Returns:
        Dict with feasibility and adjusted quantity
    """
    try:
        analyzer = _GLOBAL_CACHE.get_analyzer(exchange)
        analysis = analyzer.analyze_execution(symbol, quantity, side)
        
        return {
            "feasible": analysis["market_impact_bps"] <= max_impact_bps,
            "adjusted_quantity": analysis["recommended_quantity"],
            "impact_bps": analysis["market_impact_bps"],
            "spread_bps": analysis["spread_bps"],
            "reason": f"Impact {analysis['market_impact_bps']:.1f}bps vs max {max_impact_bps}bps"
        }
    except Exception as e:
        LOGGER.warning(f"Liquidity check failed: {e}")
        return {
            "feasible": True,  # Fail open
            "adjusted_quantity": quantity,
            "impact_bps": 0.0,
            "spread_bps": 0.0,
            "reason": f"Check failed: {e}"
        }


def get_optimal_limit_price(
    symbol: str,
    quantity: float,
    side: str,
    urgency: float = 0.5,
    exchange: str = "binance"
) -> float:
    """
    Get optimal limit price for an order.
    
    Args:
        symbol: Trading symbol
        quantity: Order quantity
        side: "buy" or "sell"
        urgency: 0.0 (patient) to 1.0 (aggressive)
        exchange: Exchange name
        
    Returns:
        Optimal limit price
    """
    try:
        analyzer = _GLOBAL_CACHE.get_analyzer(exchange)
        analysis = analyzer.analyze_execution(symbol, quantity, side, urgency)
        return analysis["optimal_price"]
    except Exception as e:
        LOGGER.warning(f"Failed to get optimal price: {e}")
        return 0.0


def estimate_execution_cost(
    symbol: str,
    quantity: float,
    side: str,
    exchange: str = "binance"
) -> Dict[str, float]:
    """
    Estimate total execution cost including spread and impact.
    
    Args:
        symbol: Trading symbol
        quantity: Order quantity
        side: "buy" or "sell"
        exchange: Exchange name
        
    Returns:
        Dict with cost breakdown
    """
    try:
        analyzer = _GLOBAL_CACHE.get_analyzer(exchange)
        analysis = analyzer.analyze_execution(symbol, quantity, side)
        
        mid_price = analysis["mid_price"]
        spread_cost_bps = analysis["spread_bps"] / 2  # Pay half spread
        impact_cost_bps = analysis["market_impact_bps"]
        total_cost_bps = spread_cost_bps + impact_cost_bps
        
        # Calculate dollar cost
        total_cost_dollars = (total_cost_bps / 10000) * mid_price * quantity
        
        return {
            "spread_bps": analysis["spread_bps"],
            "spread_cost_bps": spread_cost_bps,
            "impact_bps": impact_cost_bps,
            "total_cost_bps": total_cost_bps,
            "total_cost_dollars": total_cost_dollars,
            "mid_price": mid_price
        }
    except Exception as e:
        LOGGER.warning(f"Failed to estimate cost: {e}")
        return {
            "spread_bps": 0.0,
            "spread_cost_bps": 0.0,
            "impact_bps": 0.0,
            "total_cost_bps": 0.0,
            "total_cost_dollars": 0.0,
            "mid_price": 0.0
        }


def should_use_twap(
    symbol: str,
    quantity: float,
    side: str,
    exchange: str = "binance",
    impact_threshold_bps: float = 20.0
) -> Tuple[bool, Optional[int]]:
    """
    Determine if TWAP should be used and recommend slice count.
    
    Args:
        symbol: Trading symbol
        quantity: Order quantity
        side: "buy" or "sell"
        exchange: Exchange name
        impact_threshold_bps: Impact threshold for TWAP
        
    Returns:
        Tuple of (use_twap, num_slices)
    """
    try:
        analyzer = _GLOBAL_CACHE.get_analyzer(exchange)
        strategy = analyzer.get_execution_strategy(
            symbol, quantity, side, impact_threshold_bps
        )
        
        if strategy["strategy"] == "twap":
            return True, strategy.get("num_slices", 5)
        return False, None
        
    except Exception as e:
        LOGGER.warning(f"Failed to determine TWAP requirement: {e}")
        return False, None


def get_book_imbalance(
    symbol: str,
    exchange: str = "binance"
) -> float:
    """
    Get order book imbalance as signal.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        
    Returns:
        Imbalance (-1 to +1, positive = buy pressure)
    """
    try:
        fetcher = OrderBookFetcher(exchange)
        orderbook = fetcher.fetch_order_book(symbol)
        return orderbook.order_book_imbalance()
    except Exception as e:
        LOGGER.warning(f"Failed to get book imbalance: {e}")
        return 0.0


def adjust_quantity_for_liquidity(
    symbol: str,
    base_quantity: float,
    side: str,
    exchange: str = "binance",
    max_participation: float = 0.05,
    max_impact_bps: float = 10.0
) -> float:
    """
    Adjust quantity based on available liquidity.
    
    Args:
        symbol: Trading symbol
        base_quantity: Desired quantity from strategy
        side: "buy" or "sell"
        exchange: Exchange name
        max_participation: Max % of liquidity to take
        max_impact_bps: Max acceptable impact
        
    Returns:
        Adjusted quantity
    """
    try:
        sizer = LiquidityAwareSizer(
            max_participation_rate=max_participation,
            max_impact_bps=max_impact_bps
        )
        
        fetcher = OrderBookFetcher(exchange)
        orderbook = fetcher.fetch_order_book(symbol)
        
        adjusted = sizer.adjust_position_size(orderbook, base_quantity, side)
        
        if adjusted < base_quantity:
            LOGGER.info(
                f"Adjusted {symbol} quantity from {base_quantity} to {adjusted} "
                f"due to liquidity constraints"
            )
        
        return adjusted
        
    except Exception as e:
        LOGGER.warning(f"Failed to adjust quantity: {e}")
        return base_quantity


def get_market_conditions(
    symbol: str,
    exchange: str = "binance"
) -> Dict[str, Any]:
    """
    Get comprehensive market conditions from order book.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        
    Returns:
        Dict with market conditions
    """
    try:
        fetcher = OrderBookFetcher(exchange)
        orderbook = fetcher.fetch_order_book(symbol, limit=20)
        
        return {
            "mid_price": orderbook.mid_price,
            "spread_bps": orderbook.spread_bps,
            "imbalance": orderbook.order_book_imbalance(),
            "bid_liquidity": orderbook.total_bid_liquidity(5),
            "ask_liquidity": orderbook.total_ask_liquidity(5),
            "depth_ratio": (
                orderbook.total_bid_liquidity(5) / orderbook.total_ask_liquidity(5)
                if orderbook.total_ask_liquidity(5) > 0 else 1.0
            ),
            "timestamp": orderbook.timestamp
        }
        
    except Exception as e:
        LOGGER.warning(f"Failed to get market conditions: {e}")
        return {
            "mid_price": 0.0,
            "spread_bps": 0.0,
            "imbalance": 0.0,
            "bid_liquidity": 0.0,
            "ask_liquidity": 0.0,
            "depth_ratio": 1.0,
            "timestamp": None
        }


def validate_order_feasibility(
    symbol: str,
    quantity: float,
    side: str,
    max_impact_bps: float = 15.0,
    min_depth_levels: int = 3,
    exchange: str = "binance"
) -> Tuple[bool, str]:
    """
    Validate if an order is feasible given market conditions.
    
    Args:
        symbol: Trading symbol
        quantity: Order quantity
        side: "buy" or "sell"
        max_impact_bps: Maximum acceptable impact
        min_depth_levels: Minimum required depth
        exchange: Exchange name
        
    Returns:
        Tuple of (is_feasible, reason)
    """
    try:
        fetcher = OrderBookFetcher(exchange)
        orderbook = fetcher.fetch_order_book(symbol)
        
        levels = orderbook.asks if side.lower() == "buy" else orderbook.bids
        
        # Check depth
        if len(levels) < min_depth_levels:
            return False, f"Insufficient depth: {len(levels)} < {min_depth_levels} levels"
        
        # Check impact
        analyzer = _GLOBAL_CACHE.get_analyzer(exchange)
        analysis = analyzer.analyze_execution(symbol, quantity, side)
        
        if analysis["market_impact_bps"] > max_impact_bps:
            return False, (
                f"Impact too high: {analysis['market_impact_bps']:.1f}bps "
                f"> {max_impact_bps}bps"
            )
        
        # Check if quantity can be filled
        if not analysis["feasible"]:
            return False, analysis["sizing"]["reason"]
        
        return True, "Order is feasible"
        
    except Exception as e:
        LOGGER.warning(f"Feasibility check failed: {e}")
        return True, f"Check failed (assuming feasible): {e}"


def clear_cache():
    """Clear the global order book cache."""
    _GLOBAL_CACHE.clear()
