"""Example: Order Book Depth Integration Usage.

Demonstrates how to use order book data for:
1. Fetching and analyzing order book depth
2. Estimating market impact and slippage
3. Liquidity-aware position sizing
4. Optimal execution price calculation
5. Integration with SmartExecutor

Usage:
    python scripts/example_orderbook_usage.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.data.orderbook import (
    OrderBookFetcher,
    OrderBookAnalyzer,
    MarketImpactModel,
    LiquidityAwareSizer
)
from openquant.trading.smart_executor import SmartExecutor, ExecutionConfig


def example_1_fetch_orderbook():
    """Example 1: Fetch and analyze order book data."""
    print("\n=== Example 1: Fetching Order Book ===")
    
    fetcher = OrderBookFetcher("binance")
    
    symbol = "BTC/USDT"
    orderbook = fetcher.fetch_order_book(symbol, limit=10)
    
    print(f"Symbol: {orderbook.symbol}")
    print(f"Timestamp: {orderbook.timestamp}")
    print(f"Mid Price: ${orderbook.mid_price:,.2f}")
    print(f"Spread: ${orderbook.spread:.2f} ({orderbook.spread_bps:.2f} bps)")
    print(f"\nBest Bid: ${orderbook.best_bid.price:,.2f} x {orderbook.best_bid.size:.4f}")
    print(f"Best Ask: ${orderbook.best_ask.price:,.2f} x {orderbook.best_ask.size:.4f}")
    
    print(f"\nTop 5 Bids:")
    for i, level in enumerate(orderbook.bids[:5]):
        print(f"  {i+1}. ${level.price:,.2f} x {level.size:.4f} BTC = ${level.notional:,.2f}")
    
    print(f"\nTop 5 Asks:")
    for i, level in enumerate(orderbook.asks[:5]):
        print(f"  {i+1}. ${level.price:,.2f} x {level.size:.4f} BTC = ${level.notional:,.2f}")
    
    print(f"\nLiquidity Metrics:")
    print(f"  Total Bid Liquidity (5 levels): ${orderbook.total_bid_liquidity(5):,.2f}")
    print(f"  Total Ask Liquidity (5 levels): ${orderbook.total_ask_liquidity(5):,.2f}")
    print(f"  Order Book Imbalance: {orderbook.order_book_imbalance():.3f}")


def example_2_market_impact():
    """Example 2: Estimate market impact and slippage."""
    print("\n=== Example 2: Market Impact Estimation ===")
    
    fetcher = OrderBookFetcher("binance")
    impact_model = MarketImpactModel(impact_coefficient=0.1)
    
    symbol = "ETH/USDT"
    orderbook = fetcher.fetch_order_book(symbol, limit=20)
    
    test_sizes = [0.5, 1.0, 5.0, 10.0]
    
    print(f"\nBuying {symbol}:")
    print(f"Current Ask: ${orderbook.best_ask.price:,.2f}")
    print(f"\n{'Size (ETH)':<12} {'Avg Price':<12} {'Slippage':<12} {'Impact (bps)':<15} {'Levels'}")
    print("-" * 70)
    
    for size in test_sizes:
        slippage = impact_model.estimate_slippage(orderbook, size, "buy")
        impact = impact_model.estimate_market_impact(orderbook, size, "buy")
        
        print(f"{size:<12.2f} ${slippage['avg_price']:<11.2f} "
              f"${slippage['slippage']:<11.2f} {impact:<14.2f} {slippage['levels_consumed']}")
    
    print(f"\nSelling {symbol}:")
    print(f"Current Bid: ${orderbook.best_bid.price:,.2f}")
    print(f"\n{'Size (ETH)':<12} {'Avg Price':<12} {'Slippage':<12} {'Impact (bps)':<15} {'Levels'}")
    print("-" * 70)
    
    for size in test_sizes:
        slippage = impact_model.estimate_slippage(orderbook, size, "sell")
        impact = impact_model.estimate_market_impact(orderbook, size, "sell")
        
        print(f"{size:<12.2f} ${slippage['avg_price']:<11.2f} "
              f"${slippage['slippage']:<11.2f} {impact:<14.2f} {slippage['levels_consumed']}")


def example_3_liquidity_sizing():
    """Example 3: Liquidity-aware position sizing."""
    print("\n=== Example 3: Liquidity-Aware Position Sizing ===")
    
    sizer = LiquidityAwareSizer(
        max_participation_rate=0.05,
        max_impact_bps=10.0,
        min_depth_levels=3
    )
    
    fetcher = OrderBookFetcher("binance")
    symbol = "BTC/USDT"
    orderbook = fetcher.fetch_order_book(symbol, limit=20)
    
    desired_sizes = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print(f"\nAnalyzing {symbol} order book:")
    print(f"Mid Price: ${orderbook.mid_price:,.2f}")
    print(f"\n{'Desired Size':<15} {'Max Size':<15} {'Recommended':<15} {'Feasible':<12} {'Impact (bps)'}")
    print("-" * 85)
    
    for desired in desired_sizes:
        sizing = sizer.calculate_max_size(orderbook, "buy", desired)
        
        feasible = "Yes" if sizing["feasible"] else "No"
        impact = sizing.get("estimated_impact_bps", 0)
        
        print(f"{desired:<15.4f} {sizing['max_quantity']:<15.4f} "
              f"{sizing['recommended_quantity']:<15.4f} {feasible:<12} {impact:.2f}")
        
        if not sizing["feasible"]:
            print(f"  → {sizing['reason']}")


def example_4_optimal_execution():
    """Example 4: Calculate optimal execution prices."""
    print("\n=== Example 4: Optimal Execution Pricing ===")
    
    impact_model = MarketImpactModel()
    fetcher = OrderBookFetcher("binance")
    
    symbol = "BTC/USDT"
    orderbook = fetcher.fetch_order_book(symbol, limit=20)
    
    quantity = 0.5
    urgency_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print(f"\nBuying {quantity} {symbol.split('/')[0]}:")
    print(f"Mid Price: ${orderbook.mid_price:,.2f}")
    print(f"Best Ask: ${orderbook.best_ask.price:,.2f}")
    print(f"\n{'Urgency':<12} {'Optimal Price':<15} {'vs Mid (%)':<15} {'vs Best Ask (%)'}")
    print("-" * 60)
    
    for urgency in urgency_levels:
        optimal = impact_model.optimal_execution_price(orderbook, quantity, "buy", urgency)
        vs_mid = ((optimal - orderbook.mid_price) / orderbook.mid_price) * 100
        vs_ask = ((optimal - orderbook.best_ask.price) / orderbook.best_ask.price) * 100
        
        urgency_label = ["Patient", "Low", "Medium", "High", "Aggressive"][int(urgency * 4)]
        print(f"{urgency_label:<12} ${optimal:<14.2f} {vs_mid:>+13.3f}% {vs_ask:>+16.3f}%")


def example_5_complete_analysis():
    """Example 5: Complete execution analysis."""
    print("\n=== Example 5: Complete Execution Analysis ===")
    
    analyzer = OrderBookAnalyzer("binance")
    
    symbol = "ETH/USDT"
    quantity = 2.0
    side = "buy"
    
    analysis = analyzer.analyze_execution(
        symbol=symbol,
        quantity=quantity,
        side=side,
        urgency=0.5
    )
    
    print(f"\nExecution Analysis for: {side.upper()} {quantity} {symbol}")
    print(f"Timestamp: {analysis['timestamp']}")
    print(f"\nMarket Conditions:")
    print(f"  Mid Price: ${analysis['mid_price']:,.2f}")
    print(f"  Spread: {analysis['spread_bps']:.2f} bps")
    print(f"  Book Imbalance: {analysis['book_imbalance']:+.3f} (positive = buy pressure)")
    
    print(f"\nExecution Metrics:")
    print(f"  Optimal Price: ${analysis['optimal_price']:,.2f}")
    print(f"  Expected Slippage: {analysis['slippage_estimate']['slippage_bps']:.2f} bps")
    print(f"  Market Impact: {analysis['market_impact_bps']:.2f} bps")
    print(f"  Levels Consumed: {analysis['slippage_estimate']['levels_consumed']}")
    
    print(f"\nPosition Sizing:")
    print(f"  Desired Quantity: {quantity:.4f}")
    print(f"  Recommended Quantity: {analysis['recommended_quantity']:.4f}")
    print(f"  Feasible: {'Yes' if analysis['feasible'] else 'No'}")
    
    if not analysis['feasible']:
        print(f"  Reason: {analysis['sizing']['reason']}")
    
    strategy = analyzer.get_execution_strategy(symbol, quantity, side)
    print(f"\nRecommended Strategy: {strategy['strategy'].upper()}")
    print(f"  Reason: {strategy['reason']}")
    if strategy['strategy'] == 'twap':
        print(f"  TWAP Slices: {strategy['num_slices']}")
        print(f"  Slice Size: {strategy['slice_size']:.4f}")
    elif strategy['strategy'] == 'limit':
        print(f"  Limit Price: ${strategy['limit_price']:,.2f}")


def example_6_smart_executor_integration():
    """Example 6: Integration with SmartExecutor."""
    print("\n=== Example 6: SmartExecutor Integration ===")
    
    config = ExecutionConfig(
        use_orderbook=True,
        orderbook_exchange="binance",
        order_type=None  # Let order book analysis decide
    )
    
    print(f"\nSmartExecutor Configuration:")
    print(f"  Order Book Integration: {'Enabled' if config.use_orderbook else 'Disabled'}")
    print(f"  Exchange: {config.orderbook_exchange}")
    
    print(f"\nHow to use:")
    print("""
    # Initialize with your broker
    executor = SmartExecutor(broker=my_broker, config=config)
    
    # Execute with automatic order book optimization
    result = executor.execute(
        symbol="BTC/USDT",
        side="BUY",
        quantity=0.5,
        current_price=50000.0,
        urgency=0.5  # Optional: 0.0=patient, 1.0=aggressive
    )
    
    # The executor will:
    # - Fetch real-time order book
    # - Adjust quantity based on liquidity
    # - Calculate optimal limit price
    # - Select best execution strategy (market/limit/TWAP)
    # - Estimate and minimize slippage
    """)


def main():
    """Run all examples."""
    print("=" * 80)
    print("Order Book Depth Integration - Usage Examples")
    print("=" * 80)
    
    try:
        example_1_fetch_orderbook()
        example_2_market_impact()
        example_3_liquidity_sizing()
        example_4_optimal_execution()
        example_5_complete_analysis()
        example_6_smart_executor_integration()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: These examples require:")
        print("  1. Internet connection")
        print("  2. Access to exchange APIs (e.g., Binance)")
        print("  3. ccxt library installed")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
