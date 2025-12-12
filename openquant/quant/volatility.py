"""GARCH Volatility Modeling.

Uses the ARCH library to model and forecast volatility using GARCH(1,1).
"""
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(returns: pd.Series | np.ndarray, p: int = 1, q: int = 1) -> dict[str, Any]:
    """
    Fit GARCH(p,q) model to returns.
    Returns model summary and next-day volatility forecast.
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    scale: float = 100.0
    scaled_returns = returns * scale

    try:
        model = arch_model(scaled_returns, vol='Garch', p=p, q=q, dist='Normal')
        res = model.fit(disp='off', show_warning=False)

        forecast = res.forecast(horizon=1)
        next_vol_scaled = np.sqrt(forecast.variance.values[-1, :])[0]
        next_vol = next_vol_scaled / scale

        ann_vol = next_vol * np.sqrt(252)

        return {
            "model": "GARCH(1,1)",
            "params": res.params.to_dict(),
            "next_vol_daily": float(next_vol),
            "next_vol_annual": float(ann_vol),
            "aic": float(res.aic),
            "bic": float(res.bic)
        }
    except Exception as e:
        return {"error": str(e), "next_vol_daily": float(np.std(returns))}

def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson volatility estimator based on High-Low range.
    More efficient than Close-to-Close volatility.
    """
    const: float = 1.0 / (4.0 * np.log(2.0))
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
