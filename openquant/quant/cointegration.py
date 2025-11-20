"""Cointegration Tests for Pairs Trading.

Implements tests to find pairs of assets that move together in the long run.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Union, Optional
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

def engle_granger_test(series_y: pd.Series, series_x: pd.Series) -> Dict[str, Any]:
    """
    Perform Engle-Granger two-step cointegration test.
    Null Hypothesis (H0): No cointegration.
    
    Returns p-value and score. p < 0.05 implies cointegration.
    """
    # Align series
    df = pd.concat([series_y, series_x], axis=1).dropna()
    if len(df) < 30:
        return {"is_cointegrated": False, "error": "Insufficient data"}
        
    y = df.iloc[:, 0]
    x = df.iloc[:, 1]
    
    try:
        score, p_value, _ = coint(y, x)
        is_coint = p_value < 0.05
        
        # Calculate Hedge Ratio (Beta) via OLS
        # y = beta * x + c + epsilon
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        beta = model.params.iloc[1]
        const = model.params.iloc[0]
        spread = y - (beta * x + const)
        
        # Calculate Half-Life of Mean Reversion for the spread
        # d(Spread) = -lambda * (Spread - mean) * dt
        # Regress spread_lagged on spread_diff
        spread_lag = spread.shift(1)
        spread_diff = spread.diff()
        spread_df = pd.concat([spread_lag, spread_diff], axis=1).dropna()
        spread_df.columns = ['lag', 'diff']
        
        # diff = -lambda * (lag - mean)
        # diff = -lambda * lag + lambda * mean
        # diff = slope * lag + intercept
        # slope = -lambda -> lambda = -slope
        # half_life = log(2) / lambda
        
        model_hl = sm.OLS(spread_df['diff'], sm.add_constant(spread_df['lag'])).fit()
        slope = model_hl.params['lag']
        
        half_life = -np.log(2) / slope if slope < 0 else np.inf
        
        return {
            "test": "Engle-Granger",
            "p_value": p_value,
            "score": score,
            "is_cointegrated": is_coint,
            "hedge_ratio": beta,
            "half_life": half_life,
            "spread_std": spread.std()
        }
    except Exception as e:
        return {"is_cointegrated": False, "error": str(e)}

def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling Z-Score."""
    r = series.rolling(window=window)
    m = r.mean()
    s = r.std()
    return (series - m) / s
