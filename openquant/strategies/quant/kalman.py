
import numpy as np
import pandas as pd

from ..base import BaseStrategy
from openquant.utils.validation import (
    validate_params,
    validate_positive_param
)


class KalmanMeanReversionStrategy(BaseStrategy):
    """Kalman Filter Mean Reversion Strategy (GPU-Accelerated).

    Uses a 1D Kalman Filter to estimate the 'true' price (hidden state).
    Signals are generated based on the deviation of the observed price from the estimated price.

    GPU acceleration is automatically used for datasets with >= 100 bars.

    Long: Price < Estimated Price - Threshold
    Short: Price > Estimated Price + Threshold
    """
    
    @validate_params(
        process_noise=validate_positive_param('process_noise'),
        measurement_noise=validate_positive_param('measurement_noise'),
        threshold=validate_positive_param('threshold')
    )
    def __init__(self, process_noise: float = 1e-5, measurement_noise: float = 1e-3, threshold: float = 1.0, use_gpu: bool = True) -> None:
        self.process_noise: float = float(process_noise)
        self.measurement_noise: float = float(measurement_noise)
        self.threshold: float = float(threshold)
        self.use_gpu: bool = use_gpu

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=int)

        prices = df['Close'].values
        n = len(prices)

        if self.use_gpu and n >= 100:
            try:
                from ...gpu.dispatcher import kalman_filter
                estimated_prices, deviations = kalman_filter(
                    prices,
                    self.process_noise,
                    self.measurement_noise
                )
            except Exception:
                estimated_prices, deviations = self._cpu_kalman_filter(prices)
        else:
            estimated_prices, deviations = self._cpu_kalman_filter(prices)

        signals = pd.Series(0, index=df.index, dtype=int)

        dev_series = pd.Series(deviations, index=df.index)
        std_dev = dev_series.rolling(window=50).std().fillna(1.0)

        z_score = dev_series / std_dev

        signals[z_score > self.threshold] = -1

        signals[z_score < -self.threshold] = 1

        return signals

    def _cpu_kalman_filter(self, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """CPU fallback Kalman filter implementation."""
        n = len(prices)

        x: float = prices[0]
        P: float = 1.0
        Q: float = self.process_noise
        R: float = self.measurement_noise

        estimated_prices = np.zeros(n)
        deviations = np.zeros(n)

        for i in range(n):
            x_pred = x
            P_pred = P + Q

            z = prices[i]
            K = P_pred / (P_pred + R)
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred

            estimated_prices[i] = x
            deviations[i] = z - x

        return estimated_prices, deviations
