#!/usr/bin/env python3
"""CLI runner for focused Forex pair trading."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.research.universe_runner import run_universe
from openquant.config.forex import FOREX_CONFIG, get_spread_bps

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenQuant Forex Pair Focused Runner")
    parser.add_argument("--pairs", type=str, default="EURUSD,GBPUSD", 
                        help="Comma-separated list of Forex pairs (e.g., EURUSD,USDJPY)")
    parser.add_argument("--timeframes", type=str, default="4h,1h",
                        help="Comma-separated timeframes")
    parser.add_argument("--leverage", type=float, default=50.0,
                        help="Leverage (default: 50x for Forex)")
    parser.add_argument("--strategies", type=str, default=None,
                        help="Comma-separated strategy names (default: use all)")
    
    parser.add_argument("--trials", type=int, default=30,
                        help="Optuna trials per strategy (default: 30)")
    
    args = parser.parse_args()
    
    # Parse pairs
    pairs = [p.strip().upper() for p in args.pairs.split(",")]
    timeframes = [tf.strip() for tf in args.timeframes.split(",")]
    
    # Parse strategies if provided
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]
    
    print(f"🔹 Forex Pair Focused Runner")
    print(f"   Pairs:      {', '.join(pairs)}")
    print(f"   Timeframes: {', '.join(timeframes)}")
    print(f"   Leverage:   {args.leverage}x")
    print(f"   Trials:     {args.trials}")
    
    # Check if pairs are configured
    for pair in pairs:
        if pair not in FOREX_CONFIG:
            print(f"   ⚠️  Warning: {pair} not in FOREX_CONFIG, using defaults")
        else:
            spread = get_spread_bps(pair)
            print(f"   {pair}: spread={spread} bps")
    
    # Run universe with focused pairs
    results_path = run_universe(
        exchange="mt5",  # Forex via MT5
        symbols=pairs,   # Specific pairs
        timeframes=timeframes,
        strategies=strategies or ("sma", "ema", "rsi", "macd", "bollinger"),
        optimize=True,
        optuna_trials=args.trials,  # Use CLI argument
    )
    
    print(f"\n✅ Completed. Results stored at: {results_path}")

if __name__ == "__main__":
    main()
