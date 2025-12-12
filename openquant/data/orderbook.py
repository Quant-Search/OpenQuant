"""Order Book Depth Integration for OpenQuant.

Provides order book data fetching, liquidity analysis, market impact modeling,
and execution price estimation for exchanges that support it.

Key Features:
- Real-time order book fetching via CCXT
- Liquidity metrics (depth, spread, imbalance)
- Market impact estimation
- Optimal execution price calculation
- Liquidity-aware position sizing
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

import ccxt  # type: ignore

from ..utils.logging import get_logger
from ..utils.rate_limit import get_rate_limiter

LOGGER = get_logger(__name__)


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    size: float
    
    @property
    def notional(self) -> float:
        """Notional value (price * size)."""
        return self.price * self.size


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot with analytics."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """Best bid (highest buy price)."""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """Best ask (lowest sell price)."""
        return self.asks[0] if self.asks else None
    
    @property
    def mid_price(self) -> float:
        """Mid price between best bid and ask."""
        if not self.best_bid or not self.best_ask:
            return 0.0
        return (self.best_bid.price + self.best_ask.price) / 2.0
    
    @property
    def spread(self) -> float:
        """Bid-ask spread in absolute terms."""
        if not self.best_bid or not self.best_ask:
            return 0.0
        return self.best_ask.price - self.best_bid.price
    
    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points."""
        if self.mid_price <= 0:
            return 0.0
        return (self.spread / self.mid_price) * 10000
    
    def total_bid_liquidity(self, levels: Optional[int] = None) -> float:
        """Total bid liquidity (notional value)."""
        bids = self.bids[:levels] if levels else self.bids
        return sum(level.notional for level in bids)
    
    def total_ask_liquidity(self, levels: Optional[int] = None) -> float:
        """Total ask liquidity (notional value)."""
        asks = self.asks[:levels] if levels else self.asks
        return sum(level.notional for level in asks)
    
    def order_book_imbalance(self, levels: int = 5) -> float:
        """
        Calculate order book imbalance.
        
        Returns value between -1 (all asks) and +1 (all bids).
        Positive = more buy pressure, Negative = more sell pressure.
        """
        bid_liq = self.total_bid_liquidity(levels)
        ask_liq = self.total_ask_liquidity(levels)
        
        total = bid_liq + ask_liq
        if total <= 0:
            return 0.0
        
        return (bid_liq - ask_liq) / total
    
    def depth_at_price(self, target_price: float, side: str = "buy") -> float:
        """
        Calculate cumulative depth available at or better than target price.
        
        Args:
            target_price: Target price level
            side: "buy" or "sell"
            
        Returns:
            Total size available
        """
        if side.lower() == "buy":
            # For buying, we want asks at or below target
            return sum(level.size for level in self.asks if level.price <= target_price)
        else:
            # For selling, we want bids at or above target
            return sum(level.size for level in self.bids if level.price >= target_price)
    
    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """Convert order book to DataFrames for analysis."""
        bids_df = pd.DataFrame([
            {"price": level.price, "size": level.size, "notional": level.notional}
            for level in self.bids
        ])
        
        asks_df = pd.DataFrame([
            {"price": level.price, "size": level.size, "notional": level.notional}
            for level in self.asks
        ])
        
        return {"bids": bids_df, "asks": asks_df}


class OrderBookFetcher:
    """Fetches and manages order book data from exchanges."""
    
    def __init__(self, exchange: str, cache_seconds: float = 1.0):
        """
        Initialize order book fetcher.
        
        Args:
            exchange: Exchange name (e.g., "binance", "kraken")
            cache_seconds: Cache duration for order book snapshots
        """
        self.exchange_name = exchange.lower()
        self._exchange_instance: Optional[ccxt.Exchange] = None
        self.cache_seconds = cache_seconds
        self._cache: Dict[str, Tuple[datetime, OrderBookSnapshot]] = {}
        
    @property
    def exchange(self) -> ccxt.Exchange:
        """Get or create CCXT exchange instance."""
        if self._exchange_instance is None:
            if not hasattr(ccxt, self.exchange_name):
                raise ValueError(f"Unknown ccxt exchange: {self.exchange_name}")
            self._exchange_instance = getattr(ccxt, self.exchange_name)({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"}
            })
        return self._exchange_instance
    
    def fetch_order_book(
        self,
        symbol: str,
        limit: int = 20,
        use_cache: bool = True
    ) -> OrderBookSnapshot:
        """
        Fetch order book for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC/USDT")
            limit: Number of levels to fetch per side
            use_cache: Whether to use cached data
            
        Returns:
            OrderBookSnapshot with bids and asks
        """
        # Check cache
        if use_cache and symbol in self._cache:
            cache_time, cached_snapshot = self._cache[symbol]
            age = (datetime.now(timezone.utc) - cache_time).total_seconds()
            if age < self.cache_seconds:
                return cached_snapshot
        
        try:
            # Rate limiting
            limiter = get_rate_limiter(self.exchange_name, rate_per_sec=10.0, capacity=10)
            limiter.acquire()
            
            # Fetch order book
            ob = self.exchange.fetch_order_book(symbol, limit=limit)
            
            # Parse response
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(ob["timestamp"] / 1000.0, tz=timezone.utc) if ob.get("timestamp") else datetime.now(timezone.utc),
                bids=[OrderBookLevel(price=float(bid[0]), size=float(bid[1])) for bid in ob.get("bids", [])],
                asks=[OrderBookLevel(price=float(ask[0]), size=float(ask[1])) for ask in ob.get("asks", [])]
            )
            
            # Update cache
            self._cache[symbol] = (datetime.now(timezone.utc), snapshot)
            
            LOGGER.info(f"Fetched order book for {symbol}: mid={snapshot.mid_price:.5f}, spread={snapshot.spread_bps:.2f}bps")
            
            return snapshot
            
        except Exception as e:
            LOGGER.error(f"Failed to fetch order book for {symbol}: {e}")
            # Return cached data if available
            if symbol in self._cache:
                LOGGER.warning(f"Using stale cached order book for {symbol}")
                return self._cache[symbol][1]
            # Return empty snapshot
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bids=[],
                asks=[]
            )
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear order book cache."""
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()


class MarketImpactModel:
    """Models market impact for trade execution."""
    
    def __init__(self, impact_coefficient: float = 0.1):
        """
        Initialize market impact model.
        
        Args:
            impact_coefficient: Coefficient for market impact calculation
                               Higher = more conservative impact estimates
        """
        self.impact_coefficient = impact_coefficient
    
    def estimate_slippage(
        self,
        order_book: OrderBookSnapshot,
        quantity: float,
        side: str
    ) -> Dict[str, float]:
        """
        Estimate slippage for executing an order.
        
        Uses order book depth to calculate:
        - Expected fill price
        - Total slippage in absolute and basis points
        - Number of levels consumed
        
        Args:
            order_book: Order book snapshot
            quantity: Order quantity
            side: "buy" or "sell"
            
        Returns:
            Dict with execution metrics
        """
        levels = order_book.asks if side.lower() == "buy" else order_book.bids
        
        if not levels:
            return {
                "avg_price": 0.0,
                "slippage": 0.0,
                "slippage_bps": 0.0,
                "levels_consumed": 0,
                "filled_quantity": 0.0,
                "unfilled_quantity": quantity
            }
        
        reference_price = levels[0].price
        remaining_qty = quantity
        total_cost = 0.0
        levels_consumed = 0
        
        # Walk through order book levels
        for level in levels:
            if remaining_qty <= 0:
                break
            
            fill_qty = min(remaining_qty, level.size)
            total_cost += fill_qty * level.price
            remaining_qty -= fill_qty
            levels_consumed += 1
        
        filled_qty = quantity - remaining_qty
        
        if filled_qty <= 0:
            return {
                "avg_price": 0.0,
                "slippage": 0.0,
                "slippage_bps": 0.0,
                "levels_consumed": 0,
                "filled_quantity": 0.0,
                "unfilled_quantity": quantity
            }
        
        avg_price = total_cost / filled_qty
        slippage = avg_price - reference_price if side.lower() == "buy" else reference_price - avg_price
        slippage_bps = (slippage / reference_price) * 10000 if reference_price > 0 else 0.0
        
        return {
            "avg_price": avg_price,
            "slippage": slippage,
            "slippage_bps": slippage_bps,
            "levels_consumed": levels_consumed,
            "filled_quantity": filled_qty,
            "unfilled_quantity": remaining_qty
        }
    
    def estimate_market_impact(
        self,
        order_book: OrderBookSnapshot,
        quantity: float,
        side: str
    ) -> float:
        """
        Estimate total market impact including permanent price impact.
        
        Uses square-root model: Impact = k * sqrt(Q / ADV)
        Where Q is order size and ADV is average daily volume (approximated by book depth).
        
        Args:
            order_book: Order book snapshot
            quantity: Order quantity
            side: "buy" or "sell"
            
        Returns:
            Estimated impact in basis points
        """
        levels = order_book.asks if side.lower() == "buy" else order_book.bids
        
        if not levels or order_book.mid_price <= 0:
            return 0.0
        
        # Approximate available liquidity from top 10 levels
        total_liquidity = sum(level.size for level in levels[:10])
        
        if total_liquidity <= 0:
            return 999.0  # Very high impact if no liquidity
        
        # Order size as fraction of available liquidity
        participation_rate = quantity / total_liquidity
        
        # Square root impact model (Almgren-Chriss)
        impact_bps = self.impact_coefficient * np.sqrt(participation_rate) * 10000
        
        return impact_bps
    
    def optimal_execution_price(
        self,
        order_book: OrderBookSnapshot,
        quantity: float,
        side: str,
        urgency: float = 0.5
    ) -> float:
        """
        Calculate optimal limit price for execution.
        
        Balances execution probability vs. price improvement.
        
        Args:
            order_book: Order book snapshot
            quantity: Order quantity
            side: "buy" or "sell"
            urgency: 0.0 (patient) to 1.0 (aggressive)
            
        Returns:
            Recommended limit price
        """
        if not order_book.best_bid or not order_book.best_ask:
            return order_book.mid_price
        
        # Calculate expected slippage
        slippage_metrics = self.estimate_slippage(order_book, quantity, side)
        
        if side.lower() == "buy":
            # Start from best ask
            base_price = order_book.best_ask.price
            
            # Aggressive: pay more (up to avg price needed for full fill)
            # Patient: offer less (down to mid price)
            if urgency >= 0.5:
                # Aggressive: willing to walk up the book
                improvement = (slippage_metrics["avg_price"] - base_price) * (urgency - 0.5) * 2
                return base_price + improvement
            else:
                # Patient: offer below best ask
                improvement = (base_price - order_book.mid_price) * (0.5 - urgency) * 2
                return base_price - improvement
        else:
            # Selling
            base_price = order_book.best_bid.price
            
            if urgency >= 0.5:
                # Aggressive: willing to walk down the book
                improvement = (base_price - slippage_metrics["avg_price"]) * (urgency - 0.5) * 2
                return base_price - improvement
            else:
                # Patient: ask for more
                improvement = (order_book.mid_price - base_price) * (0.5 - urgency) * 2
                return base_price + improvement


class LiquidityAwareSizer:
    """Position sizing based on order book liquidity."""
    
    def __init__(
        self,
        max_participation_rate: float = 0.05,
        max_impact_bps: float = 10.0,
        min_depth_levels: int = 3
    ):
        """
        Initialize liquidity-aware sizer.
        
        Args:
            max_participation_rate: Maximum fraction of visible liquidity to take
            max_impact_bps: Maximum acceptable market impact in bps
            min_depth_levels: Minimum number of price levels required
        """
        self.max_participation_rate = max_participation_rate
        self.max_impact_bps = max_impact_bps
        self.min_depth_levels = min_depth_levels
        self.impact_model = MarketImpactModel()
    
    def calculate_max_size(
        self,
        order_book: OrderBookSnapshot,
        side: str,
        desired_quantity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate maximum tradeable size given liquidity constraints.
        
        Args:
            order_book: Order book snapshot
            side: "buy" or "sell"
            desired_quantity: Desired quantity (optional, for checking feasibility)
            
        Returns:
            Dict with sizing recommendations
        """
        levels = order_book.asks if side.lower() == "buy" else order_book.bids
        
        if len(levels) < self.min_depth_levels:
            return {
                "max_quantity": 0.0,
                "recommended_quantity": 0.0,
                "feasible": False,
                "reason": f"Insufficient depth ({len(levels)} < {self.min_depth_levels} levels)"
            }
        
        # Calculate total available liquidity in top 10 levels
        total_liquidity = sum(level.size for level in levels[:10])
        
        # Max size based on participation rate
        max_by_participation = total_liquidity * self.max_participation_rate
        
        # Binary search for max size given impact constraint
        max_by_impact = 0.0
        low, high = 0.0, total_liquidity
        
        for _ in range(20):  # Binary search iterations
            mid = (low + high) / 2.0
            impact = self.impact_model.estimate_market_impact(order_book, mid, side)
            
            if impact <= self.max_impact_bps:
                max_by_impact = mid
                low = mid
            else:
                high = mid
        
        # Take minimum of constraints
        max_quantity = min(max_by_participation, max_by_impact)
        
        # If desired quantity specified, check feasibility
        if desired_quantity is not None:
            feasible = desired_quantity <= max_quantity
            if not feasible:
                impact = self.impact_model.estimate_market_impact(order_book, desired_quantity, side)
                return {
                    "max_quantity": max_quantity,
                    "recommended_quantity": max_quantity,
                    "feasible": False,
                    "reason": f"Desired size {desired_quantity} exceeds max {max_quantity:.4f} (impact would be {impact:.2f}bps)"
                }
        
        return {
            "max_quantity": max_quantity,
            "recommended_quantity": max_quantity * 0.8,  # 80% of max for safety margin
            "feasible": True,
            "total_liquidity": total_liquidity,
            "participation_rate": max_quantity / total_liquidity if total_liquidity > 0 else 0.0,
            "estimated_impact_bps": self.impact_model.estimate_market_impact(order_book, max_quantity, side)
        }
    
    def adjust_position_size(
        self,
        order_book: OrderBookSnapshot,
        base_size: float,
        side: str
    ) -> float:
        """
        Adjust position size based on current liquidity.
        
        Args:
            order_book: Order book snapshot
            base_size: Base position size from strategy
            side: "buy" or "sell"
            
        Returns:
            Adjusted position size
        """
        sizing = self.calculate_max_size(order_book, side, base_size)
        
        if not sizing["feasible"]:
            LOGGER.warning(f"Position size {base_size} adjusted to {sizing['recommended_quantity']}: {sizing['reason']}")
            return sizing["recommended_quantity"]
        
        # If base size is within limits, use it
        return min(base_size, sizing["recommended_quantity"])


class OrderBookAnalyzer:
    """High-level analyzer combining all order book functionality."""
    
    def __init__(self, exchange: str):
        """
        Initialize order book analyzer.
        
        Args:
            exchange: Exchange name
        """
        self.fetcher = OrderBookFetcher(exchange)
        self.impact_model = MarketImpactModel()
        self.sizer = LiquidityAwareSizer()
    
    def analyze_execution(
        self,
        symbol: str,
        quantity: float,
        side: str,
        urgency: float = 0.5
    ) -> Dict[str, Any]:
        """
        Complete execution analysis including price, impact, and sizing.
        
        Args:
            symbol: Trading symbol
            quantity: Desired order quantity
            side: "buy" or "sell"
            urgency: Execution urgency (0.0 to 1.0)
            
        Returns:
            Complete analysis with recommendations
        """
        # Fetch order book
        order_book = self.fetcher.fetch_order_book(symbol)
        
        # Check liquidity constraints
        sizing = self.sizer.calculate_max_size(order_book, side, quantity)
        
        # Calculate optimal execution price
        optimal_price = self.impact_model.optimal_execution_price(
            order_book, quantity, side, urgency
        )
        
        # Estimate slippage
        slippage = self.impact_model.estimate_slippage(order_book, quantity, side)
        
        # Estimate total market impact
        impact = self.impact_model.estimate_market_impact(order_book, quantity, side)
        
        return {
            "symbol": symbol,
            "timestamp": order_book.timestamp,
            "mid_price": order_book.mid_price,
            "spread_bps": order_book.spread_bps,
            "book_imbalance": order_book.order_book_imbalance(),
            "optimal_price": optimal_price,
            "slippage_estimate": slippage,
            "market_impact_bps": impact,
            "sizing": sizing,
            "recommended_quantity": sizing["recommended_quantity"],
            "feasible": sizing["feasible"]
        }
    
    def get_execution_strategy(
        self,
        symbol: str,
        quantity: float,
        side: str,
        max_impact_bps: float = 10.0
    ) -> Dict[str, Any]:
        """
        Recommend execution strategy (TWAP, limit, market) based on conditions.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: "buy" or "sell"
            max_impact_bps: Maximum acceptable impact
            
        Returns:
            Execution strategy recommendation
        """
        order_book = self.fetcher.fetch_order_book(symbol)
        
        # Estimate impact
        impact = self.impact_model.estimate_market_impact(order_book, quantity, side)
        
        # Determine strategy
        if impact > max_impact_bps * 2:
            # Very high impact - use TWAP
            num_slices = int(np.ceil(impact / max_impact_bps))
            return {
                "strategy": "twap",
                "num_slices": min(num_slices, 10),
                "slice_size": quantity / num_slices,
                "estimated_impact": impact,
                "reason": f"High impact ({impact:.1f}bps) requires TWAP execution"
            }
        elif impact > max_impact_bps:
            # Moderate impact - use limit with patience
            return {
                "strategy": "limit",
                "limit_price": self.impact_model.optimal_execution_price(
                    order_book, quantity, side, urgency=0.3
                ),
                "estimated_impact": impact,
                "reason": f"Moderate impact ({impact:.1f}bps) use patient limit order"
            }
        elif order_book.spread_bps > 5.0:
            # Wide spread - use limit
            return {
                "strategy": "limit",
                "limit_price": self.impact_model.optimal_execution_price(
                    order_book, quantity, side, urgency=0.5
                ),
                "estimated_impact": impact,
                "reason": f"Wide spread ({order_book.spread_bps:.1f}bps) use limit order"
            }
        else:
            # Low impact, tight spread - market is fine
            return {
                "strategy": "market",
                "estimated_impact": impact,
                "reason": f"Low impact ({impact:.1f}bps) and tight spread, market order acceptable"
            }
