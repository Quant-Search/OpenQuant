"""Technical analysis features (minimal set)."""
from __future__ import annotations
import pandas as pd
import numpy as np


def sma(df: pd.DataFrame, length: int, price_col: str = "Close") -> pd.Series:
    """Simple moving average of a price column."""
    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not in DataFrame")
    return df[price_col].rolling(length, min_periods=length).mean()




def ema(df: pd.DataFrame, length: int, price_col: str = "Close") -> pd.Series:
    """Exponential moving average of a price column."""
    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not in DataFrame")
    # Use ewm with span=length and adjust=False (standard EMA)
    return df[price_col].ewm(span=length, adjust=False, min_periods=length).mean()



def rsi(df: pd.DataFrame, length: int = 14, price_col: str = "Close") -> pd.Series:
    """Relative Strength Index (Wilder's smoothing).
    Returns RSI in [0,100], NaN until warmup (length).
    """
    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not in DataFrame")
    px = df[price_col].astype(float)
    delta = px.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val.astype(float)


def macd_features(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
                   price_col: str = "Close") -> pd.DataFrame:
    """MACD features: returns DataFrame with columns ['macd','signal','hist'].
    macd = EMA_fast - EMA_slow; signal = EMA(macd, signal); hist = macd - signal
    """
    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not in DataFrame")
    ema_fast = df[price_col].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - sig_line
    out = pd.DataFrame({"macd": macd_line, "signal": sig_line, "hist": hist})
    out.index = df.index
    return out


def bollinger_bands(df: pd.DataFrame, length: int = 20, k: float = 2.0,
                    price_col: str = "Close") -> pd.DataFrame:
    """Bollinger Bands: returns DataFrame with ['mid','upper','lower','std'].
    mid = SMA(length); upper = mid + k*std; lower = mid - k*std
    """
    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not in DataFrame")
    mid = sma(df, length, price_col=price_col)
    std = df[price_col].rolling(length, min_periods=length).std()
    upper = mid + k * std
    lower = mid - k * std
    out = pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "std": std})
    out.index = df.index
    return out
