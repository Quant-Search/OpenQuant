#!/usr/bin/env python3
"""Cache warming utility for OpenQuant.

Pre-loads frequently used data into cache for faster application startup.
Supports both file-based and database cache warming.

Usage:
    python scripts/warm_cache.py --symbols BTC/USDT,ETH/USDT --timeframes 1d,1h
    python scripts/warm_cache.py --config cache_config.json
"""
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.data import DataLoader, get_cache, CacheConfig
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Warm OpenQuant cache with market data")
    
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT,ETH/USDT,SOL/USDT,AAPL,MSFT,SPY,QQQ",
        help="Comma-separated list of symbols to cache"
    )
    
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1d,4h,1h",
        help="Comma-separated list of timeframes to cache"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="ccxt:binance",
        help="Data source (e.g., 'yfinance', 'ccxt:binance')"
    )
    
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        help="Number of days of historical data to cache"
    )
    
    parser.add_argument(
        "--cache-backend",
        type=str,
        default="sqlite",
        choices=["sqlite", "redis"],
        help="Cache backend to use"
    )
    
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis host (if using Redis backend)"
    )
    
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port (if using Redis backend)"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cache before warming"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show cache statistics after warming"
    )
    
    return parser.parse_args()


def warm_cache_from_config(
    symbols: List[str],
    timeframes: List[str],
    source: str,
    lookback_days: int,
    cache_config: CacheConfig
):
    """Warm cache with specified configuration."""
    LOGGER.info(f"Cache warming started")
    LOGGER.info(f"  Backend: {cache_config.backend}")
    LOGGER.info(f"  Symbols: {len(symbols)}")
    LOGGER.info(f"  Timeframes: {timeframes}")
    LOGGER.info(f"  Lookback: {lookback_days} days")
    
    # Initialize cache with config
    from openquant.data.cache import _CACHE_INSTANCE, DataCache
    global _CACHE_INSTANCE
    _CACHE_INSTANCE = DataCache(cache_config)
    cache = _CACHE_INSTANCE
    
    # Initialize data loader (without cache to force fetch)
    loader = DataLoader(use_cache=False)
    
    # Warm the cache
    cache.warm_cache(
        symbols=symbols,
        timeframes=timeframes,
        data_loader=lambda src, sym, tf, start, end: loader.get_ohlcv(
            source=source,
            symbol=sym,
            timeframe=tf,
            start=start,
            end=end
        ),
        lookback_days=lookback_days
    )
    
    LOGGER.info("Cache warming completed successfully")
    
    return cache


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse symbols and timeframes
    symbols = [s.strip() for s in args.symbols.split(",")]
    timeframes = [tf.strip() for tf in args.timeframes.split(",")]
    
    # Create cache config
    cache_config = CacheConfig(
        backend=args.cache_backend,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
    )
    
    # Clear cache if requested
    if args.clear_cache:
        LOGGER.info("Clearing cache...")
        cache = get_cache()
        cleared = cache.clear_all()
        LOGGER.info(f"Cleared {cleared} cache entries")
    
    # Warm cache
    try:
        cache = warm_cache_from_config(
            symbols=symbols,
            timeframes=timeframes,
            source=args.source,
            lookback_days=args.lookback_days,
            cache_config=cache_config
        )
        
        # Show stats if requested
        if args.stats:
            stats = cache.get_stats()
            LOGGER.info("Cache Statistics:")
            LOGGER.info(f"  Status: {stats.get('status')}")
            LOGGER.info(f"  Total Keys: {stats.get('total_keys', 'N/A')}")
            if 'local_stats' in stats:
                local = stats['local_stats']
                LOGGER.info(f"  Hits: {local.get('hits', 0)}")
                LOGGER.info(f"  Misses: {local.get('misses', 0)}")
                LOGGER.info(f"  Hit Rate: {local.get('hit_rate', 'N/A')}")
            if 'size_mb' in stats:
                LOGGER.info(f"  Size: {stats['size_mb']:.2f} MB")
        
        print("\n✅ Cache warming completed successfully!")
        
    except Exception as e:
        LOGGER.error(f"Cache warming failed: {e}", exc_info=True)
        print(f"\n❌ Cache warming failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
