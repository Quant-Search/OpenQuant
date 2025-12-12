import numpy as np
import pandas as pd

from ..base import BaseStrategy
from openquant.utils.validation import (
    validate_params,
    validate_positive_int_param,
    validate_range_param
)


class HurstExponentStrategy(BaseStrategy):
    """
    Adaptive Strategy Based on Hurst Exponent (Fractal Dimension).
    
    This strategy calculates the Hurst exponent to determine the market regime and 
    adaptively switches between trend-following and mean-reversion strategies based 
    on the detected regime.
    
    Mathematical Background
    -----------------------
    The Hurst exponent (H) characterizes the long-term memory of a time series:
    
    .. math::
        \\text{Var}(\\tau) \\sim \\tau^{2H}
    
    where :math:`\\tau` is the time lag and :math:`\\text{Var}(\\tau)` is the 
    variance of the differenced series at lag :math:`\\tau`.
    
    Taking logarithms:
    
    .. math::
        \\log(\\text{Var}(\\tau)) \\sim 2H \\cdot \\log(\\tau)
    
    The slope of the log-log plot gives :math:`2H`, from which H is derived.
    
    Market Regimes
    --------------
    - **H < 0.5**: Mean-reverting (anti-persistent) regime
        - Price tends to reverse after moves
        - Best for mean-reversion strategies (e.g., RSI-based)
    - **H = 0.5**: Random walk (efficient market)
        - No predictable structure
    - **H > 0.5**: Trending (persistent) regime
        - Price tends to continue in the same direction
        - Best for trend-following strategies (e.g., moving average crossover)
    
    Implementation Note
    -------------------
    This implementation uses the Efficiency Ratio (ER) as a computationally efficient 
    proxy for the Hurst exponent:
    
    .. math::
        ER = \\frac{|P_t - P_{t-n}|}{\\sum_{i=1}^{n} |P_{t-i+1} - P_{t-i}|}
    
    where:
    - Numerator: Net price change over n periods
    - Denominator: Sum of absolute price changes (total path length)
    
    ER ranges from 0 to 1, with high values indicating trending behavior (analogous 
    to H > 0.5) and low values indicating mean-reverting behavior (analogous to H < 0.5).
    
    Trading Logic
    -------------
    **Trend Regime** (ER > 0.3):
    - Long (1): Price > SMA(20)
    - Short (-1): Price < SMA(20)
    
    **Mean Reversion Regime** (ER < 0.2):
    - Long (1): RSI < 30 (oversold)
    - Short (-1): RSI > 70 (overbought)
    
    where RSI (Relative Strength Index) is calculated as:
    
    .. math::
        RSI = 100 - \\frac{100}{1 + RS}
    
    .. math::
        RS = \\frac{\\text{Average Gain}_{14}}{\\text{Average Loss}_{14}}
    
    Parameters
    ----------
    lookback : int, optional
        Window size for Efficiency Ratio calculation and regime detection.
        Default is 100.
    trend_threshold : float, optional
        Hurst exponent threshold for trending regime. Currently unused in favor 
        of ER-based thresholds.
        Default is 0.55.
    mr_threshold : float, optional
        Hurst exponent threshold for mean-reverting regime. Currently unused in 
        favor of ER-based thresholds.
        Default is 0.45.
    
    Attributes
    ----------
    lookback : int
        Window size for calculations.
    trend_threshold : float
        Threshold for trend detection.
    mr_threshold : float
        Threshold for mean reversion detection.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample trending data
    >>> dates = pd.date_range('2020-01-01', periods=500)
    >>> trend_prices = 100 + np.cumsum(np.random.randn(500) * 0.3 + 0.05)
    >>> df_trend = pd.DataFrame({'Close': trend_prices}, index=dates)
    >>> 
    >>> # Initialize strategy
    >>> strategy = HurstExponentStrategy(lookback=100)
    >>> signals = strategy.generate_signals(df_trend)
    >>> 
    >>> # Count signal distribution
    >>> print(signals.value_counts())
    
    Notes
    -----
    - The Efficiency Ratio proxy is computationally more efficient than calculating 
      the true Hurst exponent via R/S analysis on a rolling basis.
    - The strategy automatically adapts to market conditions without parameter tuning.
    - Both SMA period (20) and RSI period (14) are standard technical analysis values.
    
    References
    ----------
    .. [1] Hurst, H. E. (1951). "Long-term storage capacity of reservoirs". 
           Transactions of the American Society of Civil Engineers.
    .. [2] Kaufman, P. J. (2013). "Trading Systems and Methods". Wiley.
    .. [3] Wilder, J. W. (1978). "New Concepts in Technical Trading Systems".
    """
    
    @validate_params(
        lookback=validate_positive_int_param('lookback'),
        trend_threshold=validate_range_param('trend_threshold', min_val=0.0, max_val=1.0),
        mr_threshold=validate_range_param('mr_threshold', min_val=0.0, max_val=1.0)
    )
    def __init__(self, lookback: int = 100, trend_threshold: float = 0.55, mr_threshold: float = 0.45) -> None:
        """
        Initialize the Hurst Exponent Strategy.
        
        Parameters
        ----------
        lookback : int, optional
            Window size for Efficiency Ratio calculation and regime detection.
        trend_threshold : float, optional
            Hurst exponent threshold for trending regime (currently unused).
        mr_threshold : float, optional
            Hurst exponent threshold for mean-reverting regime (currently unused).
        """
        self.lookback: int = int(lookback)
        self.trend_threshold: float = float(trend_threshold)
        self.mr_threshold: float = float(mr_threshold)
        
    def _calculate_hurst(self, series: pd.Series) -> float:
        """
        Calculate the Hurst exponent using log-log variance ratio method.
        
        This method uses a simplified R/S analysis approach based on the variance 
        of differenced series at multiple time lags.
        
        Parameters
        ----------
        series : pd.Series
            Price series for Hurst calculation.
        
        Returns
        -------
        hurst : float
            Estimated Hurst exponent. Returns 0.5 if calculation fails.
            - H < 0.5: Mean-reverting
            - H = 0.5: Random walk
            - H > 0.5: Trending
        
        Notes
        -----
        The calculation uses lags from 2 to 19 to estimate the scaling relationship.
        This is a fast approximation rather than full R/S analysis.
        """
        lags = range(2, 20)
        tau = [np.sqrt(np.std(series.diff(lag).dropna())) for lag in lags]
        try:
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = m[0] * 2.0
            return float(hurst)
        except:
            return 0.5

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate adaptive trading signals based on market regime detection.
        
        The strategy first calculates the Efficiency Ratio (ER) as a proxy for the 
        Hurst exponent, then applies either trend-following or mean-reversion logic 
        based on the detected regime.
        
        Parameters
        ----------
        df : pd.DataFrame
            Price data with at least a 'Close' column. Index should be datetime.
            Requires sufficient history (>= lookback + 20 bars) for proper calculations.
        
        Returns
        -------
        signals : pd.Series
            Trading signals with the same index as df:
            - 1: Long position
            - 0: No position (flat)
            - -1: Short position
        
        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Create sample data with different regimes
        >>> dates = pd.date_range('2020-01-01', periods=300)
        >>> 
        >>> # First half: trending
        >>> trend = np.cumsum(np.random.randn(150) * 0.5 + 0.1)
        >>> # Second half: mean-reverting
        >>> mr = np.cumsum(np.random.randn(150) * 0.5)
        >>> prices = np.concatenate([trend, mr]) + 100
        >>> 
        >>> df = pd.DataFrame({'Close': prices}, index=dates)
        >>> 
        >>> strategy = HurstExponentStrategy(lookback=50)
        >>> signals = strategy.generate_signals(df)
        >>> 
        >>> # Analyze regime switching
        >>> print(f"Total signals generated: {(signals != 0).sum()}")
        
        Notes
        -----
        - Returns empty Series if input DataFrame is empty.
        - The ER thresholds (0.3 for trend, 0.2 for mean reversion) are heuristic 
          and may need adjustment based on the specific market.
        - RSI and SMA calculations require sufficient warm-up period.
        - Signals may overlap between regimes at threshold boundaries.
        """
        if df.empty:
            return pd.Series(dtype=int)

        close = df['Close']
        change = close.diff(self.lookback).abs()
        volatility = close.diff().abs().rolling(window=self.lookback).sum()

        er = change / volatility
        
        sma = close.rolling(window=20).mean()
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=df.index, dtype=int)
        
        trend_cond = er > 0.3
        signals[trend_cond & (close > sma)] = 1
        signals[trend_cond & (close < sma)] = -1
        
        mr_cond = er < 0.2
        signals[mr_cond & (rsi < 30)] = 1
        signals[mr_cond & (rsi > 70)] = -1

        return signals
