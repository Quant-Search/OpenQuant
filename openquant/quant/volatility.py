"""GARCH Volatility Modeling.

Uses the ARCH library to model and forecast volatility using GARCH(1,1).
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from arch import arch_model

def fit_garch(returns: Union[pd.Series, np.ndarray], p: int = 1, q: int = 1) -> Dict[str, Any]:
    """
    Fit GARCH(p,q) model to returns.
    Returns model summary and next-day volatility forecast.
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Scale returns to percentage for numerical stability (GARCH likes values ~ 1-100)
    scale = 100.0
    scaled_returns = returns * scale
    
    try:
        # GARCH(1,1) is standard
        model = arch_model(scaled_returns, vol='Garch', p=p, q=q, dist='Normal')
        res = model.fit(disp='off', show_warning=False)
        
        # Forecast next step
        forecast = res.forecast(horizon=1)
        next_vol_scaled = np.sqrt(forecast.variance.values[-1, :])[0]
        next_vol = next_vol_scaled / scale
        
        # Annualized vol
        ann_vol = next_vol * np.sqrt(252) # Assuming daily data
        
        return {
            "model": "GARCH(1,1)",
            "params": res.params.to_dict(),
            "next_vol_daily": next_vol,
            "next_vol_annual": ann_vol,
            "aic": res.aic,
            "bic": res.bic
        }
    except Exception as e:
        return {"error": str(e), "next_vol_daily": np.std(returns)} # Fallback to simple std

def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson volatility estimator based on High-Low range.
    More efficient than Close-to-Close volatility.
    """
    const = 1.0 / (4.0 * np.log(2.0))
    rs = np.log(high / low) ** 2
    return np.sqrt(const * rs.rolling(window=window).mean())

def garman_klass_volatility(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    Garman-Klass volatility estimator.
    Uses OHLC to be more efficient than Parkinson (HL) or Close-Close.
    
    Sigma^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    
    rs = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    
    return np.sqrt(rs.rolling(window=window).mean())
