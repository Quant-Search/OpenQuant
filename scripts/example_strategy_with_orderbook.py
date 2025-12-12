"""Example: Trading Strategy with Order Book Integration.

Demonstrates how to integrate order book analysis into a trading strategy
for better execution and risk management.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.data import (
    check_liquidity,
    get_optimal_limit_price,
    estimate_execution_cost,
    adjust_quantity_for_liquidity,
    get_market_conditions,
    validate_order_feasibility
)


def example_strategy_decision():
    """Example: Make a trading decision with order book validation."""
    print("\n=== Example: Strategy with Order Book Integration ===\n")
    
    # Strategy generates a signal
    symbol = "BTC/USDT"
    side = "buy"
    base_quantity = 0.5  # Strategy wants to buy 0.5 BTC
    
    print(f"Strategy Signal: {side.upper()} {base_quantity} {symbol}")
    
    # Step 1: Check market conditions
    print("\n1. Checking market conditions...")
    conditions = get_market_conditions(symbol, exchange="binance")
    
    print(f"   Mid Price: ${conditions['mid_price']:,.2f}")
    print(f"   Spread: {conditions['spread_bps']:.2f} bps")
    print(f"   Book Imbalance: {conditions['imbalance']:+.3f}")
    print(f"   Bid Liquidity: ${conditions['bid_liquidity']:,.2f}")
    print(f"   Ask Liquidity: ${conditions['ask_liquidity']:,.2f}")
    
    # Step 2: Validate order feasibility
    print("\n2. Validating order feasibility...")
    feasible, reason = validate_order_feasibility(
        symbol, base_quantity, side, 
        max_impact_bps=15.0,
        exchange="binance"
    )
    
    if feasible:
        print(f"   ✓ {reason}")
    else:
        print(f"   ✗ {reason}")
        print("   → Order rejected or needs adjustment")
        return
    
    # Step 3: Adjust quantity for liquidity
    print("\n3. Adjusting quantity for liquidity...")
    adjusted_quantity = adjust_quantity_for_liquidity(
        symbol, base_quantity, side,
        exchange="binance",
        max_participation=0.05,
        max_impact_bps=10.0
    )
    
    if adjusted_quantity < base_quantity:
        print(f"   Quantity adjusted: {base_quantity} → {adjusted_quantity}")
    else:
        print(f"   Quantity unchanged: {adjusted_quantity}")
    
    # Step 4: Estimate execution cost
    print("\n4. Estimating execution cost...")
    cost = estimate_execution_cost(symbol, adjusted_quantity, side, exchange="binance")
    
    print(f"   Spread Cost: {cost['spread_cost_bps']:.2f} bps")
    print(f"   Impact Cost: {cost['impact_bps']:.2f} bps")
    print(f"   Total Cost: {cost['total_cost_bps']:.2f} bps (${cost['total_cost_dollars']:.2f})")
    
    # Step 5: Calculate optimal limit price
    print("\n5. Calculating optimal limit price...")
    
    # Try different urgency levels
    urgency_levels = [0.3, 0.5, 0.7]
    for urgency in urgency_levels:
        optimal_price = get_optimal_limit_price(
            symbol, adjusted_quantity, side, 
            urgency=urgency,
            exchange="binance"
        )
        
        urgency_label = {0.3: "Patient", 0.5: "Balanced", 0.7: "Aggressive"}[urgency]
        vs_mid = ((optimal_price - conditions['mid_price']) / conditions['mid_price']) * 100
        
        print(f"   {urgency_label:>11} (urgency={urgency}): ${optimal_price:,.2f} ({vs_mid:+.3f}% vs mid)")
    
    # Step 6: Final decision
    print("\n6. Final execution decision:")
    print(f"   → {side.upper()} {adjusted_quantity} {symbol}")
    print(f"   → Use limit order at optimal price")
    print(f"   → Expected total cost: {cost['total_cost_bps']:.2f} bps")
    
    return {
        "symbol": symbol,
        "side": side,
        "quantity": adjusted_quantity,
        "optimal_price": get_optimal_limit_price(symbol, adjusted_quantity, side, 0.5, "binance"),
        "estimated_cost_bps": cost['total_cost_bps']
    }


def example_risk_check():
    """Example: Pre-trade risk check using order book."""
    print("\n=== Example: Pre-Trade Risk Check ===\n")
    
    # Multiple symbols to check
    orders = [
        {"symbol": "BTC/USDT", "quantity": 0.3, "side": "buy"},
        {"symbol": "ETH/USDT", "quantity": 2.0, "side": "sell"},
        {"symbol": "BTC/USDT", "quantity": 5.0, "side": "buy"},  # Large order
    ]
    
    print("Checking order feasibility:\n")
    print(f"{'Symbol':<12} {'Side':<6} {'Quantity':<10} {'Impact':<12} {'Feasible':<10} {'Reason'}")
    print("-" * 85)
    
    for order in orders:
        liquidity = check_liquidity(
            order["symbol"],
            order["quantity"],
            order["side"],
            exchange="binance",
            max_impact_bps=15.0
        )
        
        feasible = "✓ Yes" if liquidity["feasible"] else "✗ No"
        
        print(f"{order['symbol']:<12} {order['side']:<6} {order['quantity']:<10.4f} "
              f"{liquidity['impact_bps']:<11.2f}bps {feasible:<10} "
              f"{liquidity['reason']}")
        
        if not liquidity["feasible"]:
            print(f"  → Adjust to: {liquidity['adjusted_quantity']:.4f}")


def example_adaptive_execution():
    """Example: Adaptive execution based on market conditions."""
    print("\n=== Example: Adaptive Execution Strategy ===\n")
    
    symbol = "ETH/USDT"
    quantity = 3.0
    side = "buy"
    
    print(f"Order: {side.upper()} {quantity} {symbol}\n")
    
    # Get market conditions
    conditions = get_market_conditions(symbol, exchange="binance")
    
    # Decision logic based on conditions
    print("Market Analysis:")
    print(f"  Spread: {conditions['spread_bps']:.2f} bps")
    print(f"  Imbalance: {conditions['imbalance']:+.3f}")
    
    if conditions['spread_bps'] > 10.0:
        print("\n  → Wide spread detected")
        print("  → Strategy: Use patient limit order")
        urgency = 0.2
    elif abs(conditions['imbalance']) > 0.3:
        if conditions['imbalance'] > 0 and side == "buy":
            print("\n  → Strong buy pressure (positive imbalance)")
            print("  → Strategy: Use aggressive order to front-run")
            urgency = 0.8
        elif conditions['imbalance'] < 0 and side == "sell":
            print("\n  → Strong sell pressure (negative imbalance)")
            print("  → Strategy: Use aggressive order")
            urgency = 0.8
        else:
            print("\n  → Order against prevailing pressure")
            print("  → Strategy: Use patient order for better price")
            urgency = 0.3
    else:
        print("\n  → Balanced market")
        print("  → Strategy: Use balanced approach")
        urgency = 0.5
    
    # Get optimal price based on adaptive urgency
    optimal_price = get_optimal_limit_price(
        symbol, quantity, side, urgency, exchange="binance"
    )
    
    print(f"\nExecution Plan:")
    print(f"  Urgency: {urgency} ({'Patient' if urgency < 0.4 else 'Aggressive' if urgency > 0.6 else 'Balanced'})")
    print(f"  Limit Price: ${optimal_price:,.2f}")
    
    # Estimate cost
    cost = estimate_execution_cost(symbol, quantity, side, exchange="binance")
    print(f"  Expected Cost: {cost['total_cost_bps']:.2f} bps")


def example_portfolio_execution():
    """Example: Execute multiple orders with liquidity consideration."""
    print("\n=== Example: Portfolio Execution with Liquidity Management ===\n")
    
    portfolio = [
        {"symbol": "BTC/USDT", "quantity": 0.2, "side": "buy"},
        {"symbol": "ETH/USDT", "quantity": 1.5, "side": "buy"},
        {"symbol": "SOL/USDT", "quantity": 50.0, "side": "buy"},
    ]
    
    print("Portfolio Orders:\n")
    print(f"{'Symbol':<12} {'Orig Qty':<10} {'Adj Qty':<10} {'Cost (bps)':<12} {'Status'}")
    print("-" * 70)
    
    for order in portfolio:
        # Adjust for liquidity
        adjusted_qty = adjust_quantity_for_liquidity(
            order["symbol"],
            order["quantity"],
            order["side"],
            exchange="binance"
        )
        
        # Estimate cost
        cost = estimate_execution_cost(
            order["symbol"],
            adjusted_qty,
            order["side"],
            exchange="binance"
        )
        
        # Validate
        feasible, _ = validate_order_feasibility(
            order["symbol"],
            adjusted_qty,
            order["side"],
            exchange="binance"
        )
        
        status = "✓ Ready" if feasible else "✗ Skip"
        
        print(f"{order['symbol']:<12} {order['quantity']:<10.4f} {adjusted_qty:<10.4f} "
              f"{cost['total_cost_bps']:<11.2f} {status}")


def main():
    """Run all examples."""
    print("=" * 85)
    print("Trading Strategy with Order Book Integration - Examples")
    print("=" * 85)
    
    try:
        example_strategy_decision()
        example_risk_check()
        example_adaptive_execution()
        example_portfolio_execution()
        
        print("\n" + "=" * 85)
        print("All examples completed!")
        print("\nKey Takeaways:")
        print("  • Always validate order feasibility before execution")
        print("  • Adjust quantities based on available liquidity")
        print("  • Use market conditions to adapt execution strategy")
        print("  • Estimate and minimize total execution costs")
        print("  • Consider order book imbalance as a signal")
        print("=" * 85)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: These examples require:")
        print("  1. Internet connection")
        print("  2. Access to Binance API")
        print("  3. ccxt library installed")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
