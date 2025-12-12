"""Unified data loader for multiple sources with integrated caching."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd

from .cache import get_cache


@dataclass
class DataLoader:
    """Data loader that routes to different providers by name with caching support."""
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache = get_cache() if use_cache else None

    def get_ohlcv(
        self,
        source: str,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from a given source with caching.
        
        Args:
            source: 'yfinance' or 'ccxt:<exchange>' e.g., 'ccxt:binance'.
            symbol: Ticker/symbol (e.g., 'AAPL' for yfinance, 'BTC/USDT' for ccxt).
            timeframe: One of {'1d','4h','1h','30m','15m'}.
            start, end: Optional datetime bounds.
            limit: Optional max rows for ccxt.
            
        Returns:
            DataFrame with columns ['Open','High','Low','Close','Volume'] indexed by UTC datetime.
        """
        # Try cache first
        if self.use_cache and self._cache:
            cached = self._cache.get_ohlcv(source, symbol, timeframe, start, end)
            if cached is not None and not cached.empty:
                return cached
        
        # Fetch from source
        df = self._fetch_from_source(source, symbol, timeframe, start, end, limit)
        
        # Cache the result
        if self.use_cache and self._cache and df is not None and not df.empty:
            self._cache.set_ohlcv(df, source, symbol, timeframe, start, end)
        
        return df
    
    def _fetch_from_source(
        self,
        source: str,
        symbol: str,
        timeframe: str,
        start: Optional[datetime],
        end: Optional[datetime],
        limit: Optional[int],
    ) -> pd.DataFrame:
        """Fetch data from the actual source."""
        if source == "yfinance":
            from .yfinance_source import fetch_ohlcv as yf_fetch
            return yf_fetch(symbol, timeframe=timeframe, start=start, end=end)
        
        if source.startswith("ccxt:"):
            try:
                from .ccxt_source import fetch_ohlcv as ccxt_fetch
            except Exception as e:
                raise RuntimeError("ccxt not available. Install 'ccxt' to use crypto sources.") from e
            
            exchange = source.split(":", 1)[1]
            start_dt = start
            if isinstance(start_dt, str):
                start_dt = pd.to_datetime(start_dt, utc=True).to_pydatetime()
            return ccxt_fetch(exchange, symbol, timeframe=timeframe, since=start_dt, limit=limit)
        
        raise ValueError(f"Unknown data source: {source}")
