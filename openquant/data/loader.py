"""Unified data loader for multiple sources."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd

# Lazy imports inside methods to avoid optional dependency failures at import time


@dataclass
class DataLoader:
    """Data loader that routes to different providers by name."""

    def get_ohlcv(
        self,
        source: str,
        symbol: str,
        timeframe: str = "1d",
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from a given source.
        Args:
            source: 'yfinance' or 'ccxt:<exchange>' e.g., 'ccxt:binance'.
            symbol: Ticker/symbol (e.g., 'AAPL' for yfinance, 'BTC/USDT' for ccxt).
            timeframe: One of {'1d','4h','1h','30m','15m'}.
            start, end: Optional datetime bounds.
            limit: Optional max rows for ccxt.
        Returns:
            DataFrame with columns ['Open','High','Low','Close','Volume'] indexed by UTC datetime.
        """
        if source == "yfinance":
            # Import on demand to avoid requiring yfinance for offline tests
            from .yfinance_source import fetch_ohlcv as yf_fetch  # local import
            return yf_fetch(symbol, timeframe=timeframe, start=start, end=end)
        if source.startswith("ccxt:"):
            # Import on demand and fail gracefully if ccxt is missing
            try:
                from .ccxt_source import fetch_ohlcv as ccxt_fetch  # type: ignore
            except Exception as e:  # pragma: no cover - optional dependency
                raise RuntimeError("ccxt not available. Install 'ccxt' to use crypto sources.") from e
            exchange = source.split(":", 1)[1]
            # Normalize start to datetime for ccxt; ignore end (not supported in fetch_ohlcv signature)
            start_dt = start
            if isinstance(start_dt, str):
                # Use pandas to parse and ensure UTC
                start_dt = pd.to_datetime(start_dt, utc=True).to_pydatetime()
            return ccxt_fetch(exchange, symbol, timeframe=timeframe, since=start_dt, limit=limit)
        raise ValueError(f"Unknown data source: {source}")

