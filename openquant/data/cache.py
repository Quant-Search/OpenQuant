"""Comprehensive caching layer for market data, indicators, and optimization results.

Supports Redis (production) and SQLite (development/offline) with TTL-based invalidation.
Includes cache warming on startup for frequently used data.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import json
import pickle
import hashlib
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    backend: str = "sqlite"  # "redis" or "sqlite"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    sqlite_path: str = "data_cache/cache.db"
    default_ttl: int = 3600  # 1 hour in seconds
    ohlcv_ttl: int = 300  # 5 minutes for OHLCV
    indicator_ttl: int = 600  # 10 minutes for indicators
    optimization_ttl: int = 86400  # 24 hours for optimization results
    enable_compression: bool = True
    max_cache_size_mb: int = 1000  # SQLite only


class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Get raw bytes from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: bytes, ttl: int) -> bool:
        """Set raw bytes in cache with TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries matching pattern."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class RedisBackend(CacheBackend):
    """Redis cache backend for production deployments."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._redis = None
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            import redis
            self._redis = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            self._redis.ping()
            LOGGER.info(f"Redis cache connected: {self.config.redis_host}:{self.config.redis_port}")
        except Exception as e:
            LOGGER.error(f"Redis connection failed: {e}")
            self._redis = None
            raise
    
    def get(self, key: str) -> Optional[bytes]:
        if self._redis is None:
            return None
        try:
            return self._redis.get(key)
        except Exception as e:
            LOGGER.warning(f"Redis get error for {key}: {e}")
            return None
    
    def set(self, key: str, value: bytes, ttl: int) -> bool:
        if self._redis is None:
            return False
        try:
            self._redis.setex(key, ttl, value)
            return True
        except Exception as e:
            LOGGER.warning(f"Redis set error for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        if self._redis is None:
            return False
        try:
            return self._redis.delete(key) > 0
        except Exception as e:
            LOGGER.warning(f"Redis delete error for {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        if self._redis is None:
            return False
        try:
            return self._redis.exists(key) > 0
        except Exception as e:
            LOGGER.warning(f"Redis exists error for {key}: {e}")
            return False
    
    def clear(self, pattern: Optional[str] = None) -> int:
        if self._redis is None:
            return 0
        try:
            if pattern:
                keys = self._redis.keys(pattern)
                if keys:
                    return self._redis.delete(*keys)
            else:
                self._redis.flushdb()
                return -1  # All keys deleted
            return 0
        except Exception as e:
            LOGGER.warning(f"Redis clear error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        if self._redis is None:
            return {"status": "disconnected"}
        try:
            info = self._redis.info("stats")
            return {
                "status": "connected",
                "total_keys": self._redis.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "memory_used": self._redis.info("memory").get("used_memory_human", "unknown"),
            }
        except Exception as e:
            LOGGER.warning(f"Redis stats error: {e}")
            return {"status": "error", "error": str(e)}


class SQLiteBackend(CacheBackend):
    """SQLite cache backend for development/offline use."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database."""
        import sqlite3
        
        db_path = Path(self.config.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
        self._conn.commit()
        LOGGER.info(f"SQLite cache initialized: {db_path}")
        
        # Clean expired entries on init
        self._cleanup_expired()
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        now = datetime.now(timezone.utc).timestamp()
        cursor = self._conn.execute("DELETE FROM cache WHERE expires_at < ?", (now,))
        deleted = cursor.rowcount
        self._conn.commit()
        if deleted > 0:
            LOGGER.debug(f"Cleaned {deleted} expired cache entries")
    
    def get(self, key: str) -> Optional[bytes]:
        now = datetime.now(timezone.utc).timestamp()
        cursor = self._conn.execute(
            "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
            (key, now)
        )
        row = cursor.fetchone()
        return row[0] if row else None
    
    def set(self, key: str, value: bytes, ttl: int) -> bool:
        now = datetime.now(timezone.utc).timestamp()
        expires_at = now + ttl
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (key, value, expires_at, now)
            )
            self._conn.commit()
            return True
        except Exception as e:
            LOGGER.warning(f"SQLite set error for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        try:
            cursor = self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            LOGGER.warning(f"SQLite delete error for {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        now = datetime.now(timezone.utc).timestamp()
        cursor = self._conn.execute(
            "SELECT 1 FROM cache WHERE key = ? AND expires_at > ?",
            (key, now)
        )
        return cursor.fetchone() is not None
    
    def clear(self, pattern: Optional[str] = None) -> int:
        try:
            if pattern:
                # SQLite LIKE pattern (convert glob to SQL)
                sql_pattern = pattern.replace("*", "%")
                cursor = self._conn.execute("DELETE FROM cache WHERE key LIKE ?", (sql_pattern,))
            else:
                cursor = self._conn.execute("DELETE FROM cache")
            self._conn.commit()
            return cursor.rowcount
        except Exception as e:
            LOGGER.warning(f"SQLite clear error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM cache")
            total = cursor.fetchone()[0]
            
            now = datetime.now(timezone.utc).timestamp()
            cursor = self._conn.execute("SELECT COUNT(*) FROM cache WHERE expires_at > ?", (now,))
            valid = cursor.fetchone()[0]
            
            cursor = self._conn.execute("SELECT SUM(LENGTH(value)) FROM cache")
            size = cursor.fetchone()[0] or 0
            
            return {
                "status": "connected",
                "total_keys": total,
                "valid_keys": valid,
                "expired_keys": total - valid,
                "size_bytes": size,
                "size_mb": size / (1024 * 1024),
            }
        except Exception as e:
            LOGGER.warning(f"SQLite stats error: {e}")
            return {"status": "error", "error": str(e)}


class DataCache:
    """High-level cache interface for market data, indicators, and optimization results."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.backend = self._create_backend()
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "errors": 0}
    
    def _create_backend(self) -> CacheBackend:
        """Create appropriate backend based on config."""
        if self.config.backend == "redis":
            try:
                return RedisBackend(self.config)
            except Exception as e:
                LOGGER.warning(f"Redis backend failed, falling back to SQLite: {e}")
                self.config.backend = "sqlite"
                return SQLiteBackend(self.config)
        else:
            return SQLiteBackend(self.config)
    
    def _make_key(self, namespace: str, *args, **kwargs) -> str:
        """Generate cache key from namespace and arguments."""
        parts = [namespace] + [str(a) for a in args]
        if kwargs:
            parts.append(json.dumps(kwargs, sort_keys=True))
        key_str = ":".join(parts)
        return f"openquant:{hashlib.md5(key_str.encode()).hexdigest()[:16]}:{key_str[:100]}"
    
    def _serialize(self, obj: Any) -> bytes:
        """Serialize object to bytes."""
        if isinstance(obj, pd.DataFrame):
            # Efficient DataFrame serialization
            if self.config.enable_compression:
                return obj.to_parquet(compression="snappy")
            else:
                return obj.to_parquet(compression=None)
        elif isinstance(obj, pd.Series):
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        elif isinstance(obj, np.ndarray):
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # Generic pickle for other objects
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize(self, data: bytes, obj_type: str = "auto") -> Any:
        """Deserialize bytes to object."""
        if obj_type == "dataframe" or (obj_type == "auto" and data.startswith(b"PAR1")):
            return pd.read_parquet(pd.io.common.BytesIO(data))
        else:
            return pickle.loads(data)
    
    def get_ohlcv(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """Get cached OHLCV data."""
        key = self._make_key("ohlcv", source, symbol, timeframe, start, end)
        data = self.backend.get(key)
        
        if data:
            self._stats["hits"] += 1
            try:
                return self._deserialize(data, "dataframe")
            except Exception as e:
                LOGGER.warning(f"Failed to deserialize OHLCV cache: {e}")
                self._stats["errors"] += 1
                return None
        
        self._stats["misses"] += 1
        return None
    
    def set_ohlcv(
        self,
        df: pd.DataFrame,
        source: str,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> bool:
        """Cache OHLCV data."""
        key = self._make_key("ohlcv", source, symbol, timeframe, start, end)
        try:
            data = self._serialize(df)
            success = self.backend.set(key, data, self.config.ohlcv_ttl)
            if success:
                self._stats["sets"] += 1
            return success
        except Exception as e:
            LOGGER.warning(f"Failed to cache OHLCV: {e}")
            self._stats["errors"] += 1
            return False
    
    def get_indicator(
        self,
        name: str,
        symbol: str,
        timeframe: str,
        **params
    ) -> Optional[Any]:
        """Get cached indicator calculation."""
        key = self._make_key("indicator", name, symbol, timeframe, **params)
        data = self.backend.get(key)
        
        if data:
            self._stats["hits"] += 1
            try:
                return self._deserialize(data)
            except Exception as e:
                LOGGER.warning(f"Failed to deserialize indicator cache: {e}")
                self._stats["errors"] += 1
                return None
        
        self._stats["misses"] += 1
        return None
    
    def set_indicator(
        self,
        result: Any,
        name: str,
        symbol: str,
        timeframe: str,
        **params
    ) -> bool:
        """Cache indicator calculation."""
        key = self._make_key("indicator", name, symbol, timeframe, **params)
        try:
            data = self._serialize(result)
            success = self.backend.set(key, data, self.config.indicator_ttl)
            if success:
                self._stats["sets"] += 1
            return success
        except Exception as e:
            LOGGER.warning(f"Failed to cache indicator: {e}")
            self._stats["errors"] += 1
            return False
    
    def get_optimization(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        **params
    ) -> Optional[Dict[str, Any]]:
        """Get cached optimization results."""
        key = self._make_key("optimization", strategy, symbol, timeframe, **params)
        data = self.backend.get(key)
        
        if data:
            self._stats["hits"] += 1
            try:
                return self._deserialize(data)
            except Exception as e:
                LOGGER.warning(f"Failed to deserialize optimization cache: {e}")
                self._stats["errors"] += 1
                return None
        
        self._stats["misses"] += 1
        return None
    
    def set_optimization(
        self,
        result: Dict[str, Any],
        strategy: str,
        symbol: str,
        timeframe: str,
        **params
    ) -> bool:
        """Cache optimization results."""
        key = self._make_key("optimization", strategy, symbol, timeframe, **params)
        try:
            data = self._serialize(result)
            success = self.backend.set(key, data, self.config.optimization_ttl)
            if success:
                self._stats["sets"] += 1
            return success
        except Exception as e:
            LOGGER.warning(f"Failed to cache optimization: {e}")
            self._stats["errors"] += 1
            return False
    
    def invalidate_symbol(self, symbol: str) -> int:
        """Invalidate all cache entries for a symbol."""
        pattern = f"*:{symbol}:*"
        return self.backend.clear(pattern)
    
    def invalidate_timeframe(self, timeframe: str) -> int:
        """Invalidate all cache entries for a timeframe."""
        pattern = f"*:{timeframe}:*"
        return self.backend.clear(pattern)
    
    def clear_all(self) -> int:
        """Clear entire cache."""
        return self.backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        backend_stats = self.backend.get_stats()
        hit_rate = self._stats["hits"] / (self._stats["hits"] + self._stats["misses"]) if (self._stats["hits"] + self._stats["misses"]) > 0 else 0
        
        return {
            **backend_stats,
            "local_stats": {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "errors": self._stats["errors"],
                "hit_rate": f"{hit_rate:.2%}",
            }
        }
    
    def warm_cache(
        self,
        symbols: List[str],
        timeframes: List[str],
        data_loader: Callable,
        lookback_days: int = 30
    ):
        """Warm cache with frequently used data on startup.
        
        Args:
            symbols: List of symbols to warm
            timeframes: List of timeframes to warm
            data_loader: Function(source, symbol, timeframe, start, end) -> DataFrame
            lookback_days: Number of days of historical data to cache
        """
        LOGGER.info(f"Starting cache warming for {len(symbols)} symbols, {len(timeframes)} timeframes")
        
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        
        success_count = 0
        error_count = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Check if already cached
                    cached = self.get_ohlcv("default", symbol, timeframe, start, end)
                    if cached is not None and not cached.empty:
                        LOGGER.debug(f"Cache already warm for {symbol} {timeframe}")
                        continue
                    
                    # Load and cache data
                    df = data_loader("default", symbol, timeframe, start, end)
                    if df is not None and not df.empty:
                        self.set_ohlcv(df, "default", symbol, timeframe, start, end)
                        success_count += 1
                        LOGGER.debug(f"Warmed cache: {symbol} {timeframe}")
                    
                except Exception as e:
                    LOGGER.warning(f"Failed to warm cache for {symbol} {timeframe}: {e}")
                    error_count += 1
        
        LOGGER.info(f"Cache warming complete: {success_count} successful, {error_count} errors")


# Global cache instance
_CACHE_INSTANCE: Optional[DataCache] = None


def get_cache(config: Optional[CacheConfig] = None) -> DataCache:
    """Get or create global cache instance."""
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        _CACHE_INSTANCE = DataCache(config)
    return _CACHE_INSTANCE


def invalidate_cache():
    """Reset global cache instance."""
    global _CACHE_INSTANCE
    _CACHE_INSTANCE = None


# Legacy API compatibility for existing code
_CACHE_ROOT = Path("data_cache")


def _sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "-")


def _sanitize_source(source: str) -> str:
    return source.lower().replace(":", "_")


def cache_path(source: str, symbol: str, timeframe: str) -> Path:
    root = _CACHE_ROOT / _sanitize_source(source) / timeframe.lower()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_sanitize_symbol(symbol)}.parquet"


def csv_cache_path(source: str, symbol: str, timeframe: str) -> Path:
    root = _CACHE_ROOT / _sanitize_source(source) / timeframe.lower()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_sanitize_symbol(symbol)}.csv"


def save_df(df: pd.DataFrame, source: str, symbol: str, timeframe: str) -> Path:
    p = cache_path(source, symbol, timeframe)
    try:
        df.to_parquet(p)
        return p
    except Exception:
        p_csv = csv_cache_path(source, symbol, timeframe)
        df.to_csv(p_csv)
        return p_csv


def load_df(source: str, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    p = cache_path(source, symbol, timeframe)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception:
            pass
    p_csv = csv_cache_path(source, symbol, timeframe)
    if p_csv.exists():
        try:
            df = pd.read_csv(p_csv, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            return df
        except Exception:
            return None
    return None


_FRESHNESS = {
    "1d": timedelta(days=1),
    "4h": timedelta(hours=4),
    "1h": timedelta(hours=1),
    "30m": timedelta(minutes=30),
    "15m": timedelta(minutes=15),
}


def is_fresh(df: pd.DataFrame, timeframe: str) -> bool:
    if df is None or df.empty:
        return False
    last_ts = df.index.max()
    if not isinstance(last_ts, pd.Timestamp):
        return False
    if last_ts.tz is None:
        last_ts = last_ts.tz_localize(timezone.utc)
    now = datetime.now(timezone.utc)
    thr = _FRESHNESS.get(timeframe.lower(), timedelta(hours=1))
    return (now - last_ts) < thr
