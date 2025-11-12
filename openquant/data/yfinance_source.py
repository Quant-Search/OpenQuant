"""yfinance data loader for equities/ETFs.
Note: yfinance allows timeframes like '1d','1h'. Intraday history may be limited.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
import pandas as pd
import yfinance as yf

_TIMEFRAME_MAP = {
    "1d": "1d",
    "4h": "60m",  # approximate via hourly resample after download
    "1h": "60m",
    "30m": "30m",
    "15m": "15m",
}


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Fetch OHLCV from yfinance.
    Returns UTC-indexed OHLCV DataFrame.
    """
    if timeframe not in _TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe for yfinance: {timeframe}")

    interval = _TIMEFRAME_MAP[timeframe]
    df = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"]).astype(float)

    # Normalize columns and index
    df = df.rename(columns={"Adj Close": "AdjClose"})
    df.index = pd.to_datetime(df.index, utc=True)

    # Resample 4h from 60m if requested
    if timeframe == "4h" and interval == "60m":
        o = df["Open"].resample("4h").first()
        h = df["High"].resample("4h").max()
        l = df["Low"].resample("4h").min()
        c = df["Close"].resample("4h").last()
        v = df["Volume"].resample("4h").sum()
        df = pd.concat({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, axis=1).dropna()

    return df[["Open","High","Low","Close","Volume"]].dropna().astype(float)

