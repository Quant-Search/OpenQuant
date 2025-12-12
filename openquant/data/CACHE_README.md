# OpenQuant Caching Layer

Comprehensive caching layer for market data, indicators, and optimization results with Redis and SQLite support.

## Features

- **Multiple Backends**: Redis (production) and SQLite (development/offline)
- **TTL-Based Invalidation**: Automatic expiration of stale data
- **Cache Warming**: Pre-load frequently used data on startup
- **Type-Specific TTLs**: Different expiration times for OHLCV, indicators, and optimization results
- **Compression**: Optional Parquet compression for DataFrames
- **Statistics**: Cache hit/miss tracking and performance monitoring

## Quick Start

### Basic Usage

```python
from openquant.data import DataLoader, get_cache

# Use data loader with automatic caching
loader = DataLoader(use_cache=True)
df = loader.get_ohlcv("ccxt:binance", "BTC/USDT", "1h")

# Access cache directly
cache = get_cache()
stats = cache.get_stats()
print(f"Cache hit rate: {stats['local_stats']['hit_rate']}")
```

### Configuration

```python
from openquant.data import CacheConfig, DataCache

# SQLite configuration (default)
sqlite_config = CacheConfig(
    backend="sqlite",
    sqlite_path="data_cache/cache.db",
    ohlcv_ttl=300,  # 5 minutes
    indicator_ttl=600,  # 10 minutes
    optimization_ttl=86400,  # 24 hours
)

# Redis configuration
redis_config = CacheConfig(
    backend="redis",
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
    redis_password=None,
)

cache = DataCache(redis_config)
```

## Cache Types

### 1. OHLCV Data Cache

```python
from openquant.data import get_cache
from datetime import datetime, timedelta

cache = get_cache()

# Cache OHLCV data
df = loader.get_ohlcv("ccxt:binance", "BTC/USDT", "1h")
cache.set_ohlcv(df, "ccxt:binance", "BTC/USDT", "1h", 
                start=datetime.now() - timedelta(days=7),
                end=datetime.now())

# Retrieve cached OHLCV
cached_df = cache.get_ohlcv("ccxt:binance", "BTC/USDT", "1h",
                             start=datetime.now() - timedelta(days=7),
                             end=datetime.now())
```

### 2. Indicator Cache

```python
from openquant.quant.stationarity import classify_regime

# Automatically cached indicator
regime = classify_regime(
    series=df['Close'],
    symbol="BTC/USDT",
    timeframe="1h",
    use_cache=True  # Default
)

# Manual indicator caching
cache = get_cache()
result = {"rsi": 65.4, "macd": 0.23}
cache.set_indicator(result, "rsi_macd", "BTC/USDT", "1h", period=14)

# Retrieve cached indicator
cached_result = cache.get_indicator("rsi_macd", "BTC/USDT", "1h", period=14)
```

### 3. Optimization Results Cache

```python
from openquant.optimization.optuna_search import optuna_best_params

# Automatically cached optimization
best_params = optuna_best_params(
    strat_name="StatArb",
    df=df,
    grid=param_grid,
    fee_bps=2.0,
    weight=1.0,
    timeframe="1h",
    symbol="BTC/USDT",
    use_cache=True  # Default
)

# Manual optimization caching
cache = get_cache()
opt_result = {"best_params": {"entry_z": 2.5}, "sharpe": 1.8}
cache.set_optimization(opt_result, "StatArb", "BTC/USDT", "1h")

# Retrieve cached optimization
cached_opt = cache.get_optimization("StatArb", "BTC/USDT", "1h")
```

## Cache Warming

Pre-load frequently used data on application startup for improved performance.

### Command Line

```bash
# Warm cache with default symbols
python scripts/warm_cache.py

# Custom symbols and timeframes
python scripts/warm_cache.py \
    --symbols "BTC/USDT,ETH/USDT,SOL/USDT" \
    --timeframes "1d,4h,1h" \
    --lookback-days 90 \
    --stats

# Use Redis backend
python scripts/warm_cache.py \
    --cache-backend redis \
    --redis-host localhost \
    --redis-port 6379

# Clear and rebuild cache
python scripts/warm_cache.py --clear-cache --stats
```

### Programmatic

```python
from openquant.data import get_cache, DataLoader

cache = get_cache()
loader = DataLoader(use_cache=False)

# Warm cache with specific symbols
cache.warm_cache(
    symbols=["BTC/USDT", "ETH/USDT", "AAPL"],
    timeframes=["1d", "4h", "1h"],
    data_loader=loader.get_ohlcv,
    lookback_days=30
)
```

## Cache Management

### Invalidation

```python
cache = get_cache()

# Invalidate all data for a symbol
cache.invalidate_symbol("BTC/USDT")

# Invalidate all data for a timeframe
cache.invalidate_timeframe("1h")

# Clear entire cache
cache.clear_all()
```

### Statistics

```python
cache = get_cache()
stats = cache.get_stats()

print(f"Backend: {stats['status']}")
print(f"Total Keys: {stats['total_keys']}")
print(f"Hit Rate: {stats['local_stats']['hit_rate']}")
print(f"Cache Size: {stats.get('size_mb', 'N/A')} MB")
```

## TTL Configuration

Different cache types have different default TTLs:

| Cache Type | Default TTL | Rationale |
|------------|-------------|-----------|
| OHLCV Data | 300s (5min) | Market data updates frequently |
| Indicators | 600s (10min) | Derived from OHLCV, slightly longer |
| Optimization | 86400s (24h) | Expensive computations, longer TTL |

Override defaults in `CacheConfig`:

```python
config = CacheConfig(
    ohlcv_ttl=600,        # 10 minutes
    indicator_ttl=1800,   # 30 minutes
    optimization_ttl=172800,  # 48 hours
)
```

## Backend Selection

### SQLite (Default)

**Pros:**
- No external dependencies
- Simple setup
- Persistent across restarts
- Good for development

**Cons:**
- Single-process only
- Slower than Redis
- Limited scalability

### Redis

**Pros:**
- Multi-process support
- Very fast
- Network-accessible
- Production-ready

**Cons:**
- Requires Redis server
- More complex setup
- Memory-based (configure persistence)

### Setup Redis

```bash
# Install Redis
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
redis-server

# Configure OpenQuant to use Redis
export CACHE_BACKEND=redis
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

## Environment Variables

Configure caching via environment variables:

```bash
# Cache backend
export CACHE_BACKEND=redis  # or sqlite

# Redis settings
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0
export REDIS_PASSWORD=your_password

# SQLite settings
export SQLITE_CACHE_PATH=data_cache/cache.db

# TTL settings (in seconds)
export CACHE_OHLCV_TTL=300
export CACHE_INDICATOR_TTL=600
export CACHE_OPTIMIZATION_TTL=86400
```

## Integration Examples

### With Backtest Engine

```python
from openquant.data import DataLoader
from openquant.backtest.engine import backtest_signals
from openquant.strategies.quant.stat_arb import StatArbStrategy

# Data is automatically cached
loader = DataLoader(use_cache=True)
df = loader.get_ohlcv("ccxt:binance", "BTC/USDT", "1h")

# Strategy with cached indicators
strategy = StatArbStrategy(entry_z=2.0)
signals = strategy.generate_signals(df)

# Backtest
results = backtest_signals(df, signals, fee_bps=2.0, weight=1.0)
```

### With Research Pipeline

```python
from openquant.data import DataLoader, get_cache
from openquant.optimization.optuna_search import optuna_best_params

loader = DataLoader(use_cache=True)
cache = get_cache()

symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
timeframe = "1h"

for symbol in symbols:
    # Load data (cached after first load)
    df = loader.get_ohlcv("ccxt:binance", symbol, timeframe)
    
    # Optimize parameters (cached if already computed)
    best_params = optuna_best_params(
        strat_name="StatArb",
        df=df,
        grid=param_grid,
        fee_bps=2.0,
        weight=1.0,
        timeframe=timeframe,
        symbol=symbol,
        use_cache=True
    )
    
    print(f"{symbol}: {best_params}")

# Show cache performance
print(cache.get_stats())
```

## Performance Tips

1. **Warm cache on startup** for frequently accessed data
2. **Use appropriate TTLs** based on data update frequency
3. **Monitor cache hit rates** to optimize configuration
4. **Use Redis in production** for multi-process applications
5. **Batch operations** when warming cache to minimize API calls
6. **Consider cache size** when setting TTLs for large datasets

## Troubleshooting

### Cache not persisting

- SQLite: Check file permissions on `sqlite_path`
- Redis: Verify Redis server is running and persistence is configured

### Poor cache hit rates

- Increase TTLs if data doesn't change frequently
- Warm cache with relevant symbols/timeframes
- Check cache key generation (parameters must match exactly)

### Memory issues

- Reduce TTLs to expire old data faster
- Implement cache size limits
- Use compression for DataFrames

### Redis connection failures

```python
# Automatic fallback to SQLite
config = CacheConfig(backend="redis")
cache = DataCache(config)  # Will fallback to SQLite if Redis unavailable
```

## API Reference

See `openquant/data/cache.py` for complete API documentation.

Key classes:
- `CacheConfig`: Configuration dataclass
- `DataCache`: High-level cache interface
- `CacheBackend`: Abstract backend interface
- `RedisBackend`: Redis implementation
- `SQLiteBackend`: SQLite implementation

Key functions:
- `get_cache()`: Get global cache instance
- `invalidate_cache()`: Reset global cache
