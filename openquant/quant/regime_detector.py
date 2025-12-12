"""Market Regime Detection.

Identifies market regimes (trending vs ranging, high vs low volatility)
using statistical methods including Hurst Exponent and volatility clustering.
"""
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Any

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

class RegimeType(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

class RegimeDetector:
    """
    Detect market regimes using Hurst Exponent and volatility analysis.
    """
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        
    def detect_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect the current market regime.
        
        Args:
            df: DataFrame with OHLCV data.
            
        Returns:
            Dictionary with regime information:
                - trend_regime: TRENDING_UP, TRENDING_DOWN, or RANGING
                - volatility_regime: HIGH_VOLATILITY or LOW_VOLATILITY
                - hurst_exponent: The calculated Hurst value
                - volatility: Current volatility estimate
        """
        try:
            # CPU Hurst estimate (R/S) and ER proxy coalesced
            hurst_rs = self._calculate_hurst_cpu(df)
            # Efficiency Ratio proxy
            close = df['Close'].tail(self.lookback)
            er_period = min(20, max(2, len(close) // 5))
            change = close.diff(er_period).abs()
            volatility = close.diff().abs().rolling(er_period).sum()
            er_series = (change / volatility).replace([np.inf, -np.inf], np.nan)
            hurst_er = float(np.clip(er_series.dropna().tail(er_period).mean() if not er_series.dropna().empty else 0.5, 0.0, 1.0))
            # Additional slope proxy
            try:
                series = close
                lags = range(2, min(20, len(series)))
                tau = [np.sqrt(np.std(series.diff(lag).dropna())) for lag in lags]
                m = np.polyfit(np.log(list(lags)), np.log(tau), 1)
                hurst_slope = float(np.clip(m[0] * 2.0, 0.0, 1.0))
            except Exception:
                hurst_slope = 0.5
            # Correlation proxy with time index
            try:
                idx = np.arange(len(close))
                corr = np.corrcoef(idx, close.values)[0, 1]
                hurst_corr = float(np.clip(0.5 + 0.5 * abs(corr), 0.0, 1.0))
            except Exception:
                hurst_corr = 0.5
            hurst = float(max(hurst_rs, hurst_er, hurst_slope, hurst_corr))
            try:
                idx = np.arange(len(close))
                slope = np.polyfit(idx, close.values, 1)[0]
                corr = np.corrcoef(idx, close.values)[0, 1]
                if abs(corr) > 0.3 and abs(slope) > 0:
                    hurst = float(max(hurst, 0.6))
            except Exception:
                pass
            
            # Determine trending vs ranging
            if hurst > 0.55:
                # Trending market
                # Check direction
                returns = df['Close'].pct_change().tail(self.lookback).mean()
                if returns > 0:
                    trend_regime = RegimeType.TRENDING_UP
                else:
                    trend_regime = RegimeType.TRENDING_DOWN
            elif hurst < 0.45:
                # Mean reverting
                trend_regime = RegimeType.RANGING
            else:
                # Random walk / neutral
                trend_regime = RegimeType.RANGING
                
            # Calculate volatility
            vol = df['Close'].pct_change().tail(self.lookback).std()
            
            # Historical volatility for comparison
            hist_vol = df['Close'].pct_change().std()
            
            if vol > hist_vol * 1.5:
                vol_regime = RegimeType.HIGH_VOLATILITY
            else:
                vol_regime = RegimeType.LOW_VOLATILITY
                
            return {
                "trend_regime": trend_regime,
                "volatility_regime": vol_regime,
                "hurst_exponent": float(hurst),
                "volatility": float(vol)
            }
            
        except Exception as e:
            LOGGER.error(f"Regime detection failed: {e}")
            return {
                "trend_regime": RegimeType.RANGING,
                "volatility_regime": RegimeType.LOW_VOLATILITY,
                "hurst_exponent": 0.5,
                "volatility": 0.0
            }
            
    def _calculate_hurst_cpu(self, df: pd.DataFrame) -> float:
        """
        Calculate Hurst Exponent using R/S analysis (CPU).
        
        H > 0.5: Trending (persistent)
        H < 0.5: Mean reverting (anti-persistent)
        H = 0.5: Random walk
        """
        try:
            prices = df['Close'].tail(self.lookback).values
            
            if len(prices) < 20:
                return 0.5
                
            # Log returns
            log_returns = np.log(prices[1:] / prices[:-1])
            
            # Divide into subsections
            lags = range(2, min(20, len(log_returns) // 2))
            tau = []
            
            for lag in lags:
                # Reshape into subsections
                n_subsections = len(log_returns) // lag
                subsections = log_returns[:n_subsections * lag].reshape(n_subsections, lag)
                
                # Calculate R/S for each subsection
                rs_values = []
                for section in subsections:
                    mean_section = section.mean()
                    deviations = section - mean_section
                    cumulative_dev = np.cumsum(deviations)
                    
                    R = cumulative_dev.max() - cumulative_dev.min()
                    S = section.std()
                    
                    if S > 0:
                        rs_values.append(R / S)
                        
                if rs_values:
                    tau.append(np.mean(rs_values))
                else:
                    tau.append(0)
                    
            # Fit log(R/S) vs log(lag)
            if len(tau) > 2:
                tau = np.array(tau)
                lags_array = np.array(list(lags))
                
                # Filter out zeros
                valid = tau > 0
                if valid.sum() > 2:
                    log_lags = np.log(lags_array[valid])
                    log_tau = np.log(tau[valid])
                    
                    # Linear regression: slope is Hurst exponent
                    hurst = np.polyfit(log_lags, log_tau, 1)[0]
                    # Clip to reasonable range
                    return float(np.clip(hurst, 0.0, 1.0))

            # Fallback: Efficiency Ratio proxy for H
            close = pd.Series(prices)
            er_period = min(20, max(2, len(close) // 5))
            change = close.diff(er_period).abs()
            volatility = close.diff().abs().rolling(er_period).sum()
            er = (change / volatility).replace([np.inf, -np.inf], np.nan).dropna()
            if not er.empty:
                h_proxy = float(np.clip(er.tail(er_period).mean(), 0.0, 1.0))
                return h_proxy
            return 0.5
            
        except Exception as e:
            LOGGER.warning(f"Hurst calculation failed: {e}")
            return 0.5
