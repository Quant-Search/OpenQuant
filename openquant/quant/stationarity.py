"""Stationarity and Regime Classification Tests.

Implements rigorous statistical tests to classify time series:
- Augmented Dickey-Fuller (ADF): Test for unit root (non-stationarity).
- KPSS: Test for stationarity (null hypothesis is stationary).
- Hurst Exponent: Measure of long-term memory (Trending vs Mean Reverting).
"""
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

def adf_test(series: pd.Series | np.ndarray, maxlag: int | None = None) -> dict[str, Any]:
    """
    Perform Augmented Dickey-Fuller test.
    Null Hypothesis (H0): Series has a unit root (is non-stationary).
    Alternate Hypothesis (H1): Series is stationary.

    Returns dict with p-value, test statistic, and boolean 'is_stationary' (at 5% level).
    """
    if isinstance(series, pd.Series):
        series = series.values

    series = series[~np.isnan(series)]

    try:
        result = adfuller(series, maxlag=maxlag, autolag='AIC')
        p_value: float = result[1]
        test_stat: float = result[0]
        is_stationary: bool = p_value < 0.05

        return {
            "test": "ADF",
            "p_value": float(p_value),
            "statistic": float(test_stat),
            "is_stationary": is_stationary,
            "lags": int(result[2])
        }
    except Exception as e:
        return {"test": "ADF", "error": str(e), "is_stationary": False}

def kpss_test(series: pd.Series | np.ndarray, regression: str = 'c') -> dict[str, Any]:
    """
    Perform KPSS test for stationarity.
    Null Hypothesis (H0): Series is stationary.
    Alternate Hypothesis (H1): Series has a unit root (is non-stationary).

    Note: This is the inverse of ADF.
    """
    if isinstance(series, pd.Series):
        series = series.values
    series = series[~np.isnan(series)]

    try:
        stat, p_value, lags, crit_vals = kpss(series, regression=regression, nlags='auto')
        is_stationary: bool = p_value > 0.05

        return {
            "test": "KPSS",
            "p_value": float(p_value),
            "statistic": float(stat),
            "is_stationary": is_stationary
        }
    except Exception as e:
        return {"test": "KPSS", "error": str(e), "is_stationary": False}

from hurst import compute_Hc


def hurst_exponent(series: pd.Series | np.ndarray, max_lag: int = 100) -> float:
    """
    Calculate the Hurst Exponent to determine regime.
    H < 0.5: Mean Reverting
    H = 0.5: Random Walk (Geometric Brownian Motion)
    H > 0.5: Trending

    Uses the 'hurst' library (R/S analysis).
    """
    if isinstance(series, pd.Series):
        series = series.values
    series = series[~np.isnan(series)]

    try:
        H, c, data = compute_Hc(series, kind='random_walk', simplified=False)
        return float(H)
    except Exception:
        return 0.5

def classify_regime(series: pd.Series, symbol: str = "unknown", timeframe: str = "1d", use_cache: bool = True, config=None) -> Dict[str, Any]:
    """
    Run battery of tests to classify regime with optional caching.
    
    Args:
        series: Price or returns series
        symbol: Symbol identifier for caching
        timeframe: Timeframe for caching
        use_cache: Whether to use cache
        config: Configuration object (optional)
    """
    # Get cache if enabled
    if use_cache:
        from openquant.data import get_cache
        cache = get_cache()
        
        # Try cache first
        if cache:
            cached = cache.get_indicator("regime_classification", symbol, timeframe, max_lag=100)
            if cached is not None:
                return cached
    else:
        cache = None
    
    # Get config if not provided
    if config is None:
        from openquant.config.manager import get_config
        config = get_config()
    
    stationarity_config = config.get_section("stationarity")
    
    # 1. Stationarity (Mean Reversion strength)
    adf = adf_test(series)
    kpss_res = kpss_test(series)

    h = hurst_exponent(series)
    
    regime = "Random Walk"
    confidence = "Low"
    
    hurst_mr = stationarity_config.hurst_mean_reverting
    hurst_tr = stationarity_config.hurst_trending
    hurst_hc = stationarity_config.hurst_high_confidence
    
    if h < hurst_mr:
        regime = "Mean Reverting"
        confidence = "High" if h < (hurst_mr - hurst_hc) else "Medium"
    elif h > hurst_tr:
        regime = "Trending"
        confidence = "High" if h > (hurst_tr + hurst_hc) else "Medium"
        
    # Conflict check: If ADF says Stationary (Mean Reverting) but Hurst says Trending?
    # ADF checks for unit root (random walk). If p < 0.05, it's NOT a random walk.
    # Usually stationary implies H < 0.5.
    
    result = {
        "regime": regime,
        "hurst": float(h),
        "adf_p": adf.get("p_value"),
        "is_stationary": adf.get("is_stationary"),
        "kpss_stationary": kpss_res.get("is_stationary")
    }
    
    # Cache result
    if cache:
        cache.set_indicator(result, "regime_classification", symbol, timeframe, max_lag=100)
    
    return result
