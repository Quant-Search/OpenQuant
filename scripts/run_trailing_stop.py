#!/usr/bin/env python3
"""Run the Trailing Stop Manager to monitor and update MT5 positions.

This script continuously monitors open MT5 positions and updates their
Stop Loss levels according to the trailing stop logic.

Usage:
    python scripts/run_trailing_stop.py [--interval SECONDS] [--trailing BPS]

Arguments:
    --interval: Update interval in seconds (default: 5)
    --trailing: Trailing distance in basis points (default: 50 = 0.5%)
    --activation: Minimum profit in bps before trailing starts (default: 0)
    --min-update: Minimum SL change in bps to trigger update (default: 5)
    
Example:
    # Trail 100 bps (1%) behind price, update every 10 seconds
    python scripts/run_trailing_stop.py --interval 10 --trailing 100
"""

import argparse
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.risk.trailing_stop import TrailingStopManager
from openquant.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for the trailing stop manager."""
    parser = argparse.ArgumentParser(description="Run Trailing Stop Manager for MT5")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Update interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--trailing",
        type=float,
        default=50.0,
        help="Trailing distance in basis points (default: 50 = 0.5%%)"
    )
    parser.add_argument(
        "--activation",
        type=float,
        default=0.0,
        help="Minimum profit in bps before trailing starts (default: 0)"
    )
    parser.add_argument(
        "--min-update",
        type=float,
        default=5.0,
        help="Minimum SL change in bps to trigger update (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Initialize the trailing stop manager
    manager = TrailingStopManager(
        trailing_bps=args.trailing,
        activation_bps=args.activation,
        min_update_bps=args.min_update
    )
    
    logger.info(f"Starting Trailing Stop Manager")
    logger.info(f"  Trailing: {args.trailing} bps ({args.trailing/100:.2f}%)")
    logger.info(f"  Activation: {args.activation} bps ({args.activation/100:.2f}%)")
    logger.info(f"  Min Update: {args.min_update} bps ({args.min_update/100:.2f}%)")
    logger.info(f"  Interval: {args.interval} seconds")
    
    # Lazy import MT5
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("MetaTrader5 module not available. Please install it.")
        return 1
        
    # Initialize MT5
    if not mt5.initialize():
        logger.error("Failed to initialize MT5 terminal")
        return 1
        
    try:
        # Main loop
        update_count = 0
        while True:
            try:
                # Update trailing stops
                results = manager.update_mt5_positions(mt5)
                
                if results:
                    update_count += 1
                    logger.info(f"Update #{update_count}: Modified {len(results)} position(s)")
                    for symbol, success in results.items():
                        status = "✓" if success else "✗"
                        logger.info(f"  {status} {symbol}")
                        
            except Exception as e:
                logger.error(f"Error during update: {e}")
                
            # Sleep until next update
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        logger.info("Shutting down Trailing Stop Manager...")
        
    finally:
        mt5.shutdown()
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
