from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd


def compute_regime_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute simple regime features:
    - trend_score: normalized EMA spread (EMA(20)-EMA(50)) divided by return std
    - volatility: std of returns
    Returns dict with finite floats; falls back to 0.0 when insufficient data.
    """
    out = {"trend_score": 0.0, "volatility": 0.0}
    if df is None or df.empty:
        return out
    close = df["Close"].astype(float)
    if close.size < 60:
        # not enough data for 50/20 spans; fallback to short spans
        span_fast, span_slow = 10, 20
    else:
        span_fast, span_slow = 20, 50
    try:
        ema_fast = close.ewm(span=span_fast, adjust=False).mean()
        ema_slow = close.ewm(span=span_slow, adjust=False).mean()
        spread = (ema_fast - ema_slow)
        rets = close.pct_change().dropna()
        vol = float(np.std(rets.values)) if rets.size else 0.0
        # Normalize spread by price and volatility to be comparable
        denom = (float(np.mean(close)) if float(np.mean(close)) != 0 else 1.0)
        norm_spread = float(np.mean(spread)) / denom if denom else 0.0
        trend_score = (norm_spread / vol) if vol > 1e-12 else 0.0
        out["trend_score"] = float(np.clip(trend_score, -10.0, 10.0))
        out["volatility"] = float(vol)
    except Exception:
        pass
    return out

