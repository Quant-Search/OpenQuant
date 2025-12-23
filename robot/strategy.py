"""
Strategy Interface and Implementations

Open/Closed Principle: Easy to add new strategies without modifying existing code.
Liskov Substitution: All strategies implement the same interface.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from OHLCV data.
        
        Args:
            df: DataFrame with 'Close' column
            
        Returns:
            Series of signals: +1 (long), 0 (flat), -1 (short)
        """
        raise NotImplementedError


class KalmanStrategy(BaseStrategy):
    """
    Kalman Filter Mean Reversion Strategy.
    
    Mathematical Model:
    ------------------
    State equation:     x(t+1) = x(t) + w(t),  w(t) ~ N(0, Q)
    Observation:        z(t) = x(t) + v(t),    v(t) ~ N(0, R)
    
    Trading Logic:
    - deviation = observed_price - kalman_estimate
    - z_score = deviation / rolling_std(deviation, 50 periods)
    - LONG when z_score < -threshold (price below true value)
    - SHORT when z_score > threshold (price above true value)
    """
    
    def __init__(
        self,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-3,
        threshold: float = 1.5
    ):
        """
        Initialize the Kalman Strategy.
        
        Args:
            process_noise: Q - how much the true price is expected to vary
            measurement_noise: R - how noisy the observed prices are
            threshold: Z-score threshold for generating signals
        """
        self.Q = process_noise      # Process noise variance
        self.R = measurement_noise  # Measurement noise variance
        self.threshold = threshold  # Signal generation threshold
        
    def _kalman_filter(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply 1D Kalman Filter to price series.
        
        Args:
            prices: Array of observed prices
            
        Returns:
            Tuple of (estimated_prices, deviations)
        """
        n = len(prices)
        
        # Initialize state with high uncertainty to allow non-zero early deviations
        # Use mean of first 10 prices as initial estimate (warm-up)
        warmup = min(10, n)
        x = np.mean(prices[:warmup])  # Initial estimate = mean of first 10 bars
        P = np.var(prices[:warmup]) if warmup > 1 else 1.0  # Initial uncertainty = variance
        P = max(P, 1e-6)  # Ensure non-zero uncertainty
        
        # Output arrays
        estimates = np.zeros(n)
        deviations = np.zeros(n)
        
        for i in range(n):
            # === PREDICTION STEP ===
            x_pred = x           # State prediction (random walk model)
            P_pred = P + self.Q  # Uncertainty grows by process noise
            
            # === UPDATE STEP ===
            z = prices[i]                    # Current observation
            K = P_pred / (P_pred + self.R)   # Kalman gain
            x = x_pred + K * (z - x_pred)    # Updated estimate
            P = (1 - K) * P_pred             # Updated uncertainty
            
            # Store results
            estimates[i] = x
            deviations[i] = z - x_pred  # Innovation = observation - prediction (before update)
            
        return estimates, deviations
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from OHLCV data.
        
        Args:
            df: DataFrame with 'Close' column
            
        Returns:
            Series of signals: +1 (long), 0 (flat), -1 (short)
        """
        if df.empty or len(df) < 50:
            return pd.Series(0, index=df.index, dtype=int)
        
        # Get close prices
        prices = df['Close'].values
        
        # Apply Kalman filter
        estimates, deviations = self._kalman_filter(prices)
        
        # Calculate z-score of deviations
        dev_series = pd.Series(deviations, index=df.index)
        rolling_std = dev_series.rolling(window=50).std()
        # Replace NaN and zero values to avoid division errors
        # Use 1.0 as fallback (results in z_score = deviation, conservative)
        rolling_std = rolling_std.replace(0, np.nan).fillna(1.0)
        z_score = dev_series / rolling_std
        
        # Generate signals
        signals = pd.Series(0, index=df.index, dtype=int)
        signals[z_score > self.threshold] = -1   # Price above estimate -> SHORT
        signals[z_score < -self.threshold] = 1   # Price below estimate -> LONG
        
        return signals


