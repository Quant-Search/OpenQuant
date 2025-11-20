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
