"""Local caching utilities for market data.
- Prefer Parquet; fallback to CSV if pyarrow/fastparquet missing.
- Simple freshness heuristic by timeframe.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime, timezone, timedelta

_CACHE_ROOT = Path("data_cache")


def _sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "-")


def _sanitize_source(source: str) -> str:
    # Replace characters invalid on Windows paths (e.g., ':')
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
        # Fallback to CSV to avoid optional dependency installs
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

