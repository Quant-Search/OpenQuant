
import numpy as np
import pandas as pd

from ..base import BaseStrategy
from openquant.utils.validation import (
    validate_params,
    validate_positive_param
)


class KalmanMeanReversionStrategy(BaseStrategy):
    """
    Kalman Filter-Based Mean Reversion Strategy with GPU Acceleration.
    
    This strategy uses a 1D Kalman Filter to estimate the underlying "true price" 
    (hidden state) from noisy market observations. Trading signals are generated 
    based on deviations of the observed price from the Kalman-filtered estimate.
    
    Mathematical Model
    ------------------
    The Kalman Filter models the price process as:
    
    **State Equation (Process Model):**
    
    .. math::
        x_{t+1} = x_t + w_t, \\quad w_t \\sim \\mathcal{N}(0, Q)
    
    **Measurement Equation (Observation Model):**
    
    .. math::
        z_t = x_t + v_t, \\quad v_t \\sim \\mathcal{N}(0, R)
    
    where:
    - :math:`x_t` is the true (hidden) price at time t
    - :math:`z_t` is the observed market price at time t
    - :math:`w_t` is the process noise with variance Q (process_noise)
    - :math:`v_t` is the measurement noise with variance R (measurement_noise)
    
    Kalman Filter Equations
    ------------------------
    **Prediction Step:**
    
    .. math::
        \\hat{x}_{t|t-1} = \\hat{x}_{t-1|t-1}
    
    .. math::
        P_{t|t-1} = P_{t-1|t-1} + Q
    
    **Update Step:**
    
    .. math::
        K_t = \\frac{P_{t|t-1}}{P_{t|t-1} + R}
    
    .. math::
        \\hat{x}_{t|t} = \\hat{x}_{t|t-1} + K_t(z_t - \\hat{x}_{t|t-1})
    
    .. math::
        P_{t|t} = (1 - K_t)P_{t|t-1}
    
    where:
    - :math:`K_t` is the Kalman gain
    - :math:`P_t` is the estimation error covariance
    
    Trading Logic
    -------------
    The deviation (innovation) is calculated as:
    
    .. math::
        d_t = z_t - \\hat{x}_{t|t}
    
    The normalized deviation (z-score) is:
    
    .. math::
        Z_t = \\frac{d_t}{\\sigma_{d,50}}
    
    where :math:`\\sigma_{d,50}` is the rolling standard deviation of deviations 
    over 50 periods.
    
    **Trading Signals:**
    - Long (1): :math:`Z_t < -\\text{threshold}` (price below true value)
    - Short (-1): :math:`Z_t > \\text{threshold}` (price above true value)
    - Flat (0): :math:`|Z_t| \\leq \\text{threshold}`
    
    GPU Acceleration
    ----------------
    The strategy automatically uses GPU acceleration for datasets with 100 or more 
    bars when use_gpu=True. This provides significant speedup for long time series. 
    Falls back to CPU implementation if GPU is unavailable.
    
    Parameters
    ----------
    process_noise : float, optional
        Process noise variance (Q). Controls how much the true price is expected 
        to vary between observations. Lower values produce smoother estimates.
        Default is 1e-5.
    measurement_noise : float, optional
        Measurement noise variance (R). Represents the noise level in observed prices. 
        Higher values trust observations less.
        Default is 1e-3.
    threshold : float, optional
        Z-score threshold for signal generation. Positions are entered when 
        |z-score| exceeds this value.
        Default is 1.0.
    use_gpu : bool, optional
        Whether to use GPU acceleration when available and dataset size >= 100.
        Default is True.
    
    Attributes
    ----------
    process_noise : float
        Process noise variance (Q).
    measurement_noise : float
        Measurement noise variance (R).
    threshold : float
        Signal generation threshold.
    use_gpu : bool
        GPU acceleration flag.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample mean-reverting data
    >>> dates = pd.date_range('2020-01-01', periods=500)
    >>> true_price = 100
    >>> noise = np.random.randn(500) * 2
    >>> prices = true_price + np.cumsum(noise * 0.1)
    >>> df = pd.DataFrame({'Close': prices}, index=dates)
    >>> 
    >>> # Initialize strategy
    >>> strategy = KalmanMeanReversionStrategy(
    ...     process_noise=1e-5,
    ...     measurement_noise=1e-3,
    ...     threshold=1.5,
    ...     use_gpu=True
    ... )
    >>> 
    >>> # Generate signals
    >>> signals = strategy.generate_signals(df)
    >>> 
    >>> # Analyze results
    >>> print(f"Long signals: {(signals == 1).sum()}")
    >>> print(f"Short signals: {(signals == -1).sum()}")
    >>> print(f"Flat periods: {(signals == 0).sum()}")
    
    Notes
    -----
    - The ratio Q/R determines filter responsiveness. Low Q/R produces smooth, 
      slowly-adapting estimates. High Q/R makes the filter more responsive to 
      price changes.
    - Optimal parameter values depend on the specific market and timeframe. 
      Backtesting is recommended for parameter tuning.
    - GPU acceleration requires appropriate hardware and drivers.
    - The 50-period rolling standard deviation for normalization may need adjustment 
      for different timeframes or volatility regimes.
    
    References
    ----------
    .. [1] Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction 
           Problems". Journal of Basic Engineering, 82(1): 35-45.
    .. [2] Welch, G., & Bishop, G. (2006). "An Introduction to the Kalman Filter". 
           University of North Carolina at Chapel Hill.
    .. [3] Akansu, A. N., et al. (2017). "Quant Trading with Kalman Filter". 
           Handbook of Financial Data and Risk Information.
    """
    
    @validate_params(
        process_noise=validate_positive_param('process_noise'),
        measurement_noise=validate_positive_param('measurement_noise'),
        threshold=validate_positive_param('threshold')
    )
    def __init__(self, process_noise: float = 1e-5, measurement_noise: float = 1e-3, threshold: float = 1.0, use_gpu: bool = True) -> None:
        """
        Initialize the Kalman Mean Reversion Strategy.
        
        Parameters
        ----------
        process_noise : float, optional
            Process noise variance (Q). Controls smoothness of the filter.
        measurement_noise : float, optional
            Measurement noise variance (R). Represents observation noise.
        threshold : float, optional
            Z-score threshold for signal generation.
        use_gpu : bool, optional
            Enable GPU acceleration for large datasets.
        """
        self.process_noise: float = float(process_noise)
        self.measurement_noise: float = float(measurement_noise)
        self.threshold: float = float(threshold)
        self.use_gpu: bool = use_gpu
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate mean reversion signals using Kalman Filter estimation.
        
        The method filters the observed prices to estimate the true underlying price, 
        then generates signals based on deviations from this estimate.
        
        Parameters
        ----------
        df : pd.DataFrame
            Price data with at least a 'Close' column. Index should be datetime.
            For GPU acceleration, at least 100 bars are recommended.
        
        Returns
        -------
        signals : pd.Series
            Trading signals with the same index as df:
            - 1: Long position (price below estimated true value)
            - 0: No position (price near estimated true value)
            - -1: Short position (price above estimated true value)
        
        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Create noisy price series around mean
        >>> np.random.seed(42)
        >>> dates = pd.date_range('2020-01-01', periods=200)
        >>> mean_price = 100
        >>> prices = mean_price + np.cumsum(np.random.randn(200) * 0.5)
        >>> df = pd.DataFrame({'Close': prices}, index=dates)
        >>> 
        >>> # Apply strategy
        >>> strategy = KalmanMeanReversionStrategy(threshold=1.5)
        >>> signals = strategy.generate_signals(df)
        >>> 
        >>> # Check signal transitions
        >>> transitions = signals.diff().fillna(0)
        >>> print(f"Number of signal changes: {(transitions != 0).sum()}")
        
        Notes
        -----
        - Returns empty Series if input DataFrame is empty.
        - GPU implementation is attempted first for datasets >= 100 bars when use_gpu=True.
        - Falls back to CPU implementation if GPU acceleration fails or is unavailable.
        - The 50-period rolling window for z-score calculation may produce NaN values 
          for the first 49 bars.
        """
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
        """
        CPU-based Kalman Filter implementation.
        
        This is the fallback implementation when GPU is not available or not requested.
        
        Parameters
        ----------
        prices : np.ndarray
            Array of observed prices (measurements).
        
        Returns
        -------
        estimated_prices : np.ndarray
            Kalman-filtered price estimates (posterior state estimates).
        deviations : np.ndarray
            Innovations (observed price - estimated price) at each time step.
        
        Notes
        -----
        The implementation uses standard Kalman Filter equations for a 1D state-space 
        model with constant dynamics matrix (identity).
        """
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
