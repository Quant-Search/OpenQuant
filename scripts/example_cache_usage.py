#!/usr/bin/env python3
"""Example demonstrating cache usage in OpenQuant.

This script shows:
1. Basic OHLCV data caching
2. Indicator caching
3. Optimization result caching
4. Cache statistics and management
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.data import DataLoader, get_cache, CacheConfig
from openquant.quant.stationarity import classify_regime
from openquant.optimization.optuna_search import optuna_best_params
from openquant.utils.logging import get_logger
import pandas as pd

LOGGER = get_logger(__name__)


def demo_ohlcv_caching():
    """Demonstrate OHLCV data caching."""
    print("\n" + "="*60)
    print("1. OHLCV Data Caching Demo")
    print("="*60)
    
    loader = DataLoader(use_cache=True)
    cache = get_cache()
    
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # First fetch (will hit the API)
    print(f"\nFetching {symbol} {timeframe} data (first time)...")
    start = datetime.now()
    df1 = loader.get_ohlcv("ccxt:binance", symbol, timeframe, limit=100)
    elapsed1 = (datetime.now() - start).total_seconds()
    print(f"  Fetched {len(df1)} rows in {elapsed1:.2f}s")
    
    # Second fetch (will use cache)
    print(f"\nFetching {symbol} {timeframe} data (from cache)...")
    start = datetime.now()
    df2 = loader.get_ohlcv("ccxt:binance", symbol, timeframe, limit=100)
    elapsed2 = (datetime.now() - start).total_seconds()
    print(f"  Fetched {len(df2)} rows in {elapsed2:.2f}s")
    
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float('inf')
    print(f"\n  Cache speedup: {speedup:.1f}x faster")
    
    return df1


def demo_indicator_caching(df: pd.DataFrame):
    """Demonstrate indicator result caching."""
    print("\n" + "="*60)
    print("2. Indicator Caching Demo")
    print("="*60)
    
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    # First calculation (will compute)
    print(f"\nCalculating regime classification (first time)...")
    start = datetime.now()
    regime1 = classify_regime(df['Close'], symbol=symbol, timeframe=timeframe, use_cache=True)
    elapsed1 = (datetime.now() - start).total_seconds()
    print(f"  Regime: {regime1['regime']}, Hurst: {regime1['hurst']:.3f}")
    print(f"  Computed in {elapsed1:.3f}s")
    
    # Second calculation (will use cache)
    print(f"\nCalculating regime classification (from cache)...")
    start = datetime.now()
    regime2 = classify_regime(df['Close'], symbol=symbol, timeframe=timeframe, use_cache=True)
    elapsed2 = (datetime.now() - start).total_seconds()
    print(f"  Regime: {regime2['regime']}, Hurst: {regime2['hurst']:.3f}")
    print(f"  Retrieved in {elapsed2:.3f}s")
    
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float('inf')
    print(f"\n  Cache speedup: {speedup:.1f}x faster")


def demo_optimization_caching(df: pd.DataFrame):
    """Demonstrate optimization result caching."""
    print("\n" + "="*60)
    print("3. Optimization Caching Demo")
    print("="*60)
    
    symbol = "BTC/USDT"
    timeframe = "1h"
    
    param_grid = {
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.0, 0.5],
        "lookback": [50, 100]
    }
    
    # First optimization (will compute)
    print(f"\nRunning optimization (first time, 5 trials)...")
    start = datetime.now()
    try:
        best_params1 = optuna_best_params(
            strat_name="StatArb",
            df=df,
            grid=param_grid,
            fee_bps=2.0,
            weight=1.0,
            timeframe=timeframe,
            n_trials=5,
            symbol=symbol,
            use_cache=True
        )
        elapsed1 = (datetime.now() - start).total_seconds()
        print(f"  Best params: {best_params1}")
        print(f"  Optimized in {elapsed1:.2f}s")
        
        # Second optimization (will use cache)
        print(f"\nRunning optimization (from cache)...")
        start = datetime.now()
        best_params2 = optuna_best_params(
            strat_name="StatArb",
            df=df,
            grid=param_grid,
            fee_bps=2.0,
            weight=1.0,
            timeframe=timeframe,
            n_trials=5,
            symbol=symbol,
            use_cache=True
        )
        elapsed2 = (datetime.now() - start).total_seconds()
        print(f"  Best params: {best_params2}")
        print(f"  Retrieved in {elapsed2:.3f}s")
        
        speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float('inf')
        print(f"\n  Cache speedup: {speedup:.1f}x faster")
    except Exception as e:
        print(f"  Optimization skipped: {e}")


def demo_cache_management():
    """Demonstrate cache management operations."""
    print("\n" + "="*60)
    print("4. Cache Management Demo")
    print("="*60)
    
    cache = get_cache()
    
    # Show statistics
    print("\nCache Statistics:")
    stats = cache.get_stats()
    print(f"  Backend: {stats.get('backend', 'sqlite')}")
    print(f"  Status: {stats.get('status')}")
    print(f"  Total Keys: {stats.get('total_keys', 'N/A')}")
    
    if 'local_stats' in stats:
        local = stats['local_stats']
        print(f"  Cache Hits: {local.get('hits', 0)}")
        print(f"  Cache Misses: {local.get('misses', 0)}")
        print(f"  Hit Rate: {local.get('hit_rate', 'N/A')}")
    
    if 'size_mb' in stats:
        print(f"  Cache Size: {stats['size_mb']:.2f} MB")
    
    # Demonstrate invalidation
    print("\nCache Invalidation:")
    
    # Invalidate specific symbol
    count = cache.invalidate_symbol("BTC/USDT")
    print(f"  Invalidated {count} entries for BTC/USDT")
    
    # Invalidate specific timeframe
    count = cache.invalidate_timeframe("1h")
    print(f"  Invalidated {count} entries for 1h timeframe")


def main():
    """Run all cache demonstrations."""
    print("\n" + "="*60)
    print("OpenQuant Caching Layer Demo")
    print("="*60)
    
    # Configure cache (SQLite by default)
    config = CacheConfig(
        backend="sqlite",
        ohlcv_ttl=600,  # 10 minutes
        indicator_ttl=1200,  # 20 minutes
        optimization_ttl=86400,  # 24 hours
    )
    
    print(f"\nCache Configuration:")
    print(f"  Backend: {config.backend}")
    print(f"  OHLCV TTL: {config.ohlcv_ttl}s")
    print(f"  Indicator TTL: {config.indicator_ttl}s")
    print(f"  Optimization TTL: {config.optimization_ttl}s")
    
    try:
        # Run demos
        df = demo_ohlcv_caching()
        
        if df is not None and not df.empty:
            demo_indicator_caching(df)
            demo_optimization_caching(df)
        
        demo_cache_management()
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        LOGGER.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
