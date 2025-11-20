"""Stationarity and Regime Classification Tests.

Implements rigorous statistical tests to classify time series:
- Augmented Dickey-Fuller (ADF): Test for unit root (non-stationarity).
- KPSS: Test for stationarity (null hypothesis is stationary).
- Hurst Exponent: Measure of long-term memory (Trending vs Mean Reverting).
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Union
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

# Suppress warnings from statsmodels/arch if needed
warnings.filterwarnings("ignore")

def adf_test(series: Union[pd.Series, np.ndarray], maxlag: int = None) -> Dict[str, Any]:
    """
    Perform Augmented Dickey-Fuller test.
    Null Hypothesis (H0): Series has a unit root (is non-stationary).
    Alternate Hypothesis (H1): Series is stationary.
    
    Returns dict with p-value, test statistic, and boolean 'is_stationary' (at 5% level).
    """
    if isinstance(series, pd.Series):
        series = series.values
        
    # Handle NaNs
    series = series[~np.isnan(series)]
    
    try:
        result = adfuller(series, maxlag=maxlag, autolag='AIC')
        p_value = result[1]
        test_stat = result[0]
        is_stationary = p_value < 0.05
        
        return {
            "test": "ADF",
            "p_value": p_value,
            "statistic": test_stat,
            "is_stationary": is_stationary,
            "lags": result[2]
        }
    except Exception as e:
        return {"test": "ADF", "error": str(e), "is_stationary": False}

def kpss_test(series: Union[pd.Series, np.ndarray], regression: str = 'c') -> Dict[str, Any]:
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
        # 'c' = constant (stationary around mean), 'ct' = constant + trend (stationary around trend)
        stat, p_value, lags, crit_vals = kpss(series, regression=regression, nlags='auto')
        is_stationary = p_value > 0.05 # High p-value means we cannot reject H0 (Stationary)
        
        return {
            "test": "KPSS",
            "p_value": p_value,
            "statistic": stat,
            "is_stationary": is_stationary
        }
    except Exception as e:
        return {"test": "KPSS", "error": str(e), "is_stationary": False}

from hurst import compute_Hc

def hurst_exponent(series: Union[pd.Series, np.ndarray], max_lag: int = 100) -> float:
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
        # compute_Hc returns H, c, data
        # kind='random_walk' means the series represents values (prices/levels), not returns
        H, c, data = compute_Hc(series, kind='random_walk', simplified=False)
        return H
    except Exception:
        return 0.5

def classify_regime(series: pd.Series) -> Dict[str, Any]:
    """
    Run battery of tests to classify regime.
    """
    # 1. Stationarity (Mean Reversion strength)
    adf = adf_test(series)
    kpss_res = kpss_test(series)
    
    # 2. Memory (Trend strength)
    h = hurst_exponent(series)
    
    regime = "Random Walk"
    confidence = "Low"
    
    if h < 0.45:
        regime = "Mean Reverting"
        confidence = "High" if h < 0.4 else "Medium"
    elif h > 0.55:
        regime = "Trending"
        confidence = "High" if h > 0.6 else "Medium"
        
    # Conflict check: If ADF says Stationary (Mean Reverting) but Hurst says Trending?
    # ADF checks for unit root (random walk). If p < 0.05, it's NOT a random walk.
    # Usually stationary implies H < 0.5.
    
    return {
        "regime": regime,
        "hurst": h,
        "adf_p": adf.get("p_value"),
        "is_stationary": adf.get("is_stationary"),
        "kpss_stationary": kpss_res.get("is_stationary")
    }
