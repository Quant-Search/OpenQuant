from typing import Dict, Any
import pandas as pd
import numpy as np
from ..base import BaseStrategy

class KalmanMeanReversionStrategy(BaseStrategy):
    """Kalman Filter Mean Reversion Strategy (GPU-Accelerated).
    
    Uses a 1D Kalman Filter to estimate the 'true' price (hidden state).
    Signals are generated based on the deviation of the observed price from the estimated price.
    
    GPU acceleration is automatically used for datasets with >= 100 bars.
    
    Long: Price < Estimated Price - Threshold
    Short: Price > Estimated Price + Threshold
    """
    
    def __init__(self, process_noise: float = 1e-5, measurement_noise: float = 1e-3, threshold: float = 1.0, use_gpu: bool = True):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.threshold = float(threshold)
        self.use_gpu = use_gpu
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series()
            
        prices = df['Close'].values
        n = len(prices)
        
        # Try GPU-accelerated Kalman filter
        if self.use_gpu and n >= 100:
            try:
                from ...gpu.dispatcher import kalman_filter
                estimated_prices, deviations = kalman_filter(
                    prices, 
                    self.process_noise, 
                    self.measurement_noise
                )
            except Exception as e:
                # Fallback to CPU
                estimated_prices, deviations = self._cpu_kalman_filter(prices)
        else:
            # Use CPU for small datasets
            estimated_prices, deviations = self._cpu_kalman_filter(prices)
            
        signals = pd.Series(0, index=df.index)
        
        # Calculate standard deviation of deviations to normalize threshold
        dev_series = pd.Series(deviations, index=df.index)
        std_dev = dev_series.rolling(window=50).std().fillna(1.0)
        
        z_score = dev_series / std_dev
        
        # Signal Logic
        # If Price is significantly ABOVE estimate (Positive Z), it should revert DOWN -> Short
        signals[z_score > self.threshold] = -1
        
        # If Price is significantly BELOW estimate (Negative Z), it should revert UP -> Long
        signals[z_score < -self.threshold] = 1
        
        return signals
        
    def _cpu_kalman_filter(self, prices: np.ndarray):
        """CPU fallback Kalman filter implementation."""
        n = len(prices)
        
        # Kalman Filter Initialization
        x = prices[0]
        P = 1.0
        Q = self.process_noise
        R = self.measurement_noise
        
        estimated_prices = np.zeros(n)
        deviations = np.zeros(n)
        
        for i in range(n):
            # Prediction Step
            x_pred = x
            P_pred = P + Q
            
            # Update Step
            z = prices[i]
            K = P_pred / (P_pred + R)
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred
            
            estimated_prices[i] = x
            deviations[i] = z - x
            
        return estimated_prices, deviations
