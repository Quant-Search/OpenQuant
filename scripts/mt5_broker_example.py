#!/usr/bin/env python3
"""
Example: Using MT5Broker for live trading.

Prerequisites:
1. MetaTrader 5 installed on Windows
2. Valid MT5 account (demo or live)
3. Credentials set in .env file
"""
from dotenv import load_dotenv
load_dotenv()

from openquant.broker import MT5Broker

def main():
    # Initialize MT5 Broker (will read credentials from .env)
    print("Connecting to MetaTrader 5...")
    broker = MT5Broker()
    
    # Get account information
    print(f"\nAccount Balance: ${broker.get_cash():.2f}")
    print(f"Account Equity: ${broker.get_equity():.2f}")
    
    # Get current positions
    positions = broker.get_positions()
    print(f"\nCurrent Positions:")
    if positions:
        for symbol, volume in positions.items():
            direction = "LONG" if volume > 0 else "SHORT"
            print(f"  {symbol}: {abs(volume):.2f} lots ({direction})")
    else:
        print("  No open positions")
    
    # Example: Place a market order (commented out for safety)
    # CAUTION: This will place a real order!
    """
    print("\nPlacing market order...")
    order = broker.place_order(
        symbol="EURUSD",
        quantity=0.01,  # 0.01 lots (micro lot)
        side="buy",
        order_type="market"
    )
    print(f"  Order placed: {order}")
    """
    
    # Example: Close all positions (commented out for safety)
    # CAUTION: This will close ALL positions!
    """
    print("\nClosing all positions...")
    result = broker.close_all_positions()
    print(f"  Closed {result['closed_positions']} positions")
    """
    
    # Shutdown connection when done
    print("\nShutting down MT5 connection...")
    broker.shutdown()
    print("Done!")

if __name__ == "__main__":
    main()
