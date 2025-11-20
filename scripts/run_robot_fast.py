#!/usr/bin/env python3
"""Fast mode CLI runner - optimized for speed over accuracy."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.research.universe_runner import run_universe

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenQuant Fast Mode (Speed Optimized)")
    parser.add_argument("--top-n", type=int, default=10, help="Top N symbols")
    parser.add_argument("--interval", type=int, default=60, help="Cycle interval (minutes)")
    
    args = parser.parse_args()
    
    print(f"⚡ Fast Mode Enabled")
    print(f"   Top N: {args.top_n}")
    print(f"   Strategies: 5 classic only (no pandas-ta)")
    print(f"   Optimization: Disabled")
    print(f"   WFO: Disabled")
    
    # Run with minimal settings
    run_universe(
        exchange="binance",
        top_n=args.top_n,
        timeframes=("4h",),  # Single timeframe only
        strategies=("sma", "ema", "rsi", "macd", "bollinger"),  # Classic 5 only
        optimize=False,  # No optimization
        run_wfo=False,  # No WFO
    )
    
    print(f"\n✅ Fast mode completed")

if __name__ == "__main__":
    main()
