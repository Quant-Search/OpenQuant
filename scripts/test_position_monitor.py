"""Test script for position monitoring system.

This script verifies that the position monitor:
1. Detects open positions
2. Calculates metrics correctly
3. Updates trailing stops
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openquant.trading.position_monitor import PositionMonitor
from openquant.risk.trailing_stop import TrailingStopManager
from openquant.broker.mt5_broker import MT5Broker
from dotenv import load_dotenv
import time

load_dotenv()

def test_position_monitor():
    """Test the position monitoring system."""
    print("=" * 60)
    print("Position Monitor Test")
    print("=" * 60)
    
    # Initialize MT5 Broker
    print("\n1. Connecting to MT5...")
    try:
        broker = MT5Broker()
        print(f"   Connected to account #{broker.login}")
        print(f"   Equity: ${broker.get_equity():,.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return
        
    # Check current positions
    print("\n2. Checking current positions...")
    positions = broker.get_positions()
    
    if not positions:
        print("   No open positions found.")
        print("\n   TIP: Open a position in MT5 and run this script again.")
        broker.shutdown()
        return
        
    print(f"   Found {len(positions)} positions:")
    for symbol, qty in positions.items():
        print(f"     - {symbol}: {qty:+.4f} lots")
        
    # Create position monitor
    print("\n3. Creating position monitor...")
    trailing_mgr = TrailingStopManager(
        trailing_bps=30,  # Trail 0.3% behind price
        activation_bps=50,  # Activate after 0.5% profit
        min_update_bps=5    # Update if SL improves by 0.05%
    )
    
    monitor = PositionMonitor(
        check_interval_seconds=5,  # Check every 5 seconds for testing
        trailing_stop_manager=trailing_mgr
    )
    
    # Start monitoring
    print("\n4. Starting position monitor...")
    monitor.start(broker)
    
    # Monitor for 30 seconds
    print("\n5. Monitoring positions for 30 seconds...")
    print("   (Position metrics will be logged)\n")
    
    def on_update(metrics):
        """Callback when position is updated."""
        print(f"   [{metrics.symbol}] "
              f"P&L: {metrics.unrealized_pnl_pct:+.2f}% (${metrics.unrealized_pnl_usd:+,.2f}) | "
              f"Price: {metrics.current_price:.5f} | "
              f"SL: {metrics.current_sl:.5f} | "
              f"TP: {metrics.current_tp:.5f}")
    
    monitor.on_position_update = on_update
    
    try:
        for i in range(6):  # 6 checks over 30 seconds
            time.sleep(5)
            
            # Get all metrics
            metrics = monitor.get_all_metrics()
            
            if not metrics:
                print(f"\n   Check {i+1}/6: No positions")
            else:
                print(f"\n   Check {i+1}/6:")
                for m in metrics:
                    print(f"     {m.symbol}: {m.unrealized_pnl_pct:+.2f}% | "
                          f"Age: {m.age_seconds/60:.1f}min | "
                          f"SL: {m.current_sl:.5f}")
                          
    except KeyboardInterrupt:
        print("\n\n   Interrupted by user")
        
    # Stop monitoring
    print("\n\n6. Stopping monitor...")
    monitor.stop()
    
    # Shutdown
    print("7. Disconnecting from MT5...")
    broker.shutdown()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_position_monitor()
