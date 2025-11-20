from typing import Dict, Any
import pandas as pd
import numpy as np
from ..base import BaseStrategy

class KalmanMeanReversionStrategy(BaseStrategy):
    """Kalman Filter Mean Reversion Strategy.
    
    Uses a 1D Kalman Filter to estimate the 'true' price (hidden state).
    Signals are generated based on the deviation of the observed price from the estimated price.
    
    Long: Price < Estimated Price - Threshold
    Short: Price > Estimated Price + Threshold
    """
    
    def __init__(self, process_noise: float = 1e-5, measurement_noise: float = 1e-3, threshold: float = 1.0):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.threshold = float(threshold)
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series()
            
        prices = df['Close'].values
        n = len(prices)
        
        # Kalman Filter Initialization
        # State: x (estimated price)
        # Covariance: P
        
        x = prices[0]
        P = 1.0
        
        Q = self.process_noise      # Process noise variance
        R = self.measurement_noise  # Measurement noise variance
        
        estimated_prices = np.zeros(n)
        deviations = np.zeros(n)
        
        for i in range(n):
            # Prediction Step
            x_pred = x
            P_pred = P + Q
            
            # Update Step
            z = prices[i] # Observation
            K = P_pred / (P_pred + R) # Kalman Gain
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred
            
            estimated_prices[i] = x
            deviations[i] = z - x
            
        signals = pd.Series(0, index=df.index)
        
        # Calculate standard deviation of deviations to normalize threshold?
        # Or use fixed threshold? Let's use rolling std of deviation for dynamic threshold
        
        dev_series = pd.Series(deviations, index=df.index)
        std_dev = dev_series.rolling(window=50).std().fillna(1.0)
        
        z_score = dev_series / std_dev
        
        # Signal Logic
        # If Price is significantly ABOVE estimate (Positive Z), it should revert DOWN -> Short
        signals[z_score > self.threshold] = -1
        
        # If Price is significantly BELOW estimate (Negative Z), it should revert UP -> Long
        signals[z_score < -self.threshold] = 1
        
        return signals
