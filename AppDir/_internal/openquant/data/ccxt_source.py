"""ccxt data loader for crypto exchanges (e.g., binance).
Requires: pip install ccxt
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, Dict
import time
import pandas as pd

import ccxt  # type: ignore
from ..utils.rate_limit import get_rate_limiter


_CCXT_TIMEFRAMES = {"1d": "1d", "4h": "4h", "1h": "1h", "30m": "30m", "15m": "15m"}


_EX_CACHE: Dict[str, ccxt.Exchange] = {}

def _exchange(name: str) -> ccxt.Exchange:  # type: ignore
    """Return a cached ccxt exchange instance in rate-limit-friendly mode.
    Note: ccxt instances are not guaranteed to be thread-safe for concurrent calls.
    We use a process-wide rate limiter to serialize requests at the desired rate.
    """
    name = name.lower()
    if not hasattr(ccxt, name):
        raise ValueError(f"Unknown ccxt exchange: {name}")
    ex = _EX_CACHE.get(name)
    if ex is None:
        ex = getattr(ccxt, name)({"enableRateLimit": True})
        _EX_CACHE[name] = ex
    return ex


def fetch_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str = "1h",
    since: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch OHLCV from ccxt and return standardized DataFrame."""
    if timeframe not in _CCXT_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe for ccxt: {timeframe}")
    ex = _exchange(exchange)
    tf = _CCXT_TIMEFRAMES[timeframe]

    since_ms = int(since.timestamp() * 1000) if since else None

    # Rate limiting: bound request rate per exchange
    limiter = get_rate_limiter(exchange, rate_per_sec=8.0, capacity=8)
    limiter.acquire()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, since=since_ms, limit=limit)
    if not ohlcv:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"]).astype(float)

    df = pd.DataFrame(ohlcv, columns=["timestamp","Open","High","Low","Close","Volume"]).set_index("timestamp")
    # Basic data validation / normalization
    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[["Open","High","Low","Close","Volume"]].astype(float)
    # Drop rows with impossible values
    df = df[(df[["Open","High","Low","Close","Volume"]] >= 0).all(axis=1)]
    return df

