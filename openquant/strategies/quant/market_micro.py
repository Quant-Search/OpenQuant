"""
Liquidity Provision Strategy (Market Microstructure).
Adapts trading style based on Flow Toxicity (VPIN).
"""
import numpy as np
import pandas as pd

from openquant.quant.microstructure import vpin
from openquant.strategies.base import BaseStrategy
from openquant.utils.validation import (
    validate_params,
    validate_positive_int_param,
    validate_range_param
)

class LiquidityProvisionStrategy(BaseStrategy):
    """
    Adaptive Liquidity Provision Strategy Based on Flow Toxicity (VPIN).
    
    This strategy uses market microstructure metrics to detect informed trading 
    (toxic order flow) and adaptively switches between market-making (mean reversion) 
    and momentum strategies based on the level of flow toxicity.
    
    Mathematical Background
    -----------------------
    **Volume-Synchronized Probability of Informed Trading (VPIN):**
    
    VPIN measures the imbalance between buying and selling volumes as a proxy for 
    the presence of informed traders:
    
    .. math::
        \\text{VPIN}_t = \\frac{\\sum_{i=t-n+1}^{t} |V_{\\text{buy},i} - V_{\\text{sell},i}|}{\\sum_{i=t-n+1}^{t} V_i}
    
    where:
    - :math:`V_{\\text{buy},i}` is buy volume in bucket i
    - :math:`V_{\\text{sell},i}` is sell volume in bucket i
    - :math:`V_i` is total volume in bucket i
    - :math:`n` is the number of buckets in the rolling window
    
    VPIN ranges from 0 to 1:
    - **Low VPIN (< 0.25)**: Uninformed flow, benign liquidity
        - Market noise dominates
        - Mean reversion likely profitable
    - **High VPIN (> 0.25)**: Informed flow, toxic liquidity
        - Informed traders present
        - Price moves likely to persist
    
    Trading Regimes
    ---------------
    The strategy operates in two distinct regimes:
    
    **1. Low Toxicity Regime** (VPIN ≤ threshold):
    
    Market Making / Mean Reversion Logic:
    
    .. math::
        \\text{Signal} = \\begin{cases}
        +1 & \\text{if } P_t < \\text{MA}_t \\text{ (buy low)} \\\\
        -1 & \\text{if } P_t > \\text{MA}_t \\text{ (sell high)} \\\\
        0 & \\text{otherwise}
        \\end{cases}
    
    Rationale: Price deviations from fair value (MA) are likely temporary noise.
    
    **2. High Toxicity Regime** (VPIN > threshold):
    
    Momentum / Trend Following Logic:
    
    .. math::
        \\text{Signal} = \\begin{cases}
        +1 & \\text{if } P_t > \\text{MA}_t \\text{ (follow trend up)} \\\\
        -1 & \\text{if } P_t < \\text{MA}_t \\text{ (follow trend down)} \\\\
        0 & \\text{otherwise}
        \\end{cases}
    
    Rationale: Informed traders have private information; price moves likely to continue.
    
    Parameters
    ----------
    vpin_window : int, optional
        Number of volume buckets to use in the rolling VPIN calculation. 
        Larger windows produce smoother VPIN estimates.
        Default is 50.
    vpin_threshold : float, optional
        VPIN level above which to switch from mean reversion to momentum. 
        Typical values range from 0.2 to 0.3 for liquid markets, higher for 
        less liquid markets.
        Default is 0.25.
    lookback : int, optional
        Window size for moving average calculation used in both regimes.
        Default is 20.
    
    Attributes
    ----------
    vpin_window : int
        Number of buckets for VPIN calculation.
    vpin_threshold : float
        Toxicity threshold for regime switching.
    lookback : int
        Moving average lookback period.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Create sample data with volume
    >>> dates = pd.date_range('2020-01-01', periods=500, freq='1min')
    >>> df = pd.DataFrame({
    ...     'Close': 100 + np.cumsum(np.random.randn(500) * 0.1),
    ...     'Volume': np.random.randint(1000, 10000, 500),
    ...     'High': 100 + np.cumsum(np.random.randn(500) * 0.1) + 0.5,
    ...     'Low': 100 + np.cumsum(np.random.randn(500) * 0.1) - 0.5
    ... }, index=dates)
    >>> 
    >>> # Initialize strategy
    >>> strategy = LiquidityProvisionStrategy(
    ...     vpin_window=50,
    ...     vpin_threshold=0.25,
    ...     lookback=20
    ... )
    >>> 
    >>> # Generate signals
    >>> signals = strategy.generate_signals(df)
    >>> 
    >>> # Analyze regime distribution
    >>> print(f"Total positions: {(signals != 0).sum()}")
    
    Notes
    -----
    - Requires sufficient data (>= 2 * lookback bars) for proper initialization.
    - VPIN calculation requires volume data and uses heuristic bucket sizing.
    - The strategy is particularly effective in markets with heterogeneous participants 
      (mix of informed and uninformed traders).
    - Bucket volume is dynamically estimated from historical average; may require 
      adjustment for different markets or timeframes.
    - Returns all zeros if insufficient data is available.
    
    References
    ----------
    .. [1] Easley, D., López de Prado, M. M., & O'Hara, M. (2012). "Flow Toxicity 
           and Liquidity in a High-frequency World". Review of Financial Studies, 
           25(5): 1457-1493.
    .. [2] Easley, D., López de Prado, M. M., & O'Hara, M. (2011). "The Microstructure 
           of the 'Flash Crash': Flow Toxicity, Liquidity Crashes and the Probability 
           of Informed Trading". Journal of Portfolio Management, 37(2): 118-128.
    .. [3] Hasbrouck, J. (2007). "Empirical Market Microstructure: The Institutions, 
           Economics, and Econometrics of Securities Trading". Oxford University Press.
    """
    
    @validate_params(
        vpin_window=validate_positive_int_param('vpin_window'),
        vpin_threshold=validate_range_param('vpin_threshold', min_val=0.0, max_val=1.0),
        lookback=validate_positive_int_param('lookback')
    )
    def __init__(self, vpin_window: int = 50, vpin_threshold: float = 0.25, lookback: int = 20) -> None:
        """
        Initialize the Liquidity Provision Strategy.
        
        Parameters
        ----------
        vpin_window : int, optional
            Number of volume buckets for VPIN calculation.
        vpin_threshold : float, optional
            VPIN threshold for regime switching (0 to 1).
        lookback : int, optional
            Moving average lookback period for signal generation.
        """
        self.vpin_window: int = vpin_window
        self.vpin_threshold: float = vpin_threshold
        self.lookback: int = lookback
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate adaptive trading signals based on flow toxicity.
        
        The method first calculates VPIN to measure flow toxicity, then applies 
        either mean-reversion or momentum logic depending on the detected regime.
        
        Parameters
        ----------
        df : pd.DataFrame
            Price and volume data. Required columns:
            - 'Close': Closing prices
            - 'Volume': Trade volumes
            - 'High': High prices (for VPIN calculation)
            - 'Low': Low prices (for VPIN calculation)
            Index should be datetime.
            Minimum required length: 2 * lookback bars.
        
        Returns
        -------
        signals : pd.Series
            Trading signals with the same index as df:
            - 1: Long position
            - 0: No position (flat)
            - -1: Short position
            
            In low toxicity regime (VPIN ≤ threshold):
            - Long when price below MA (buy the dip)
            - Short when price above MA (sell the rally)
            
            In high toxicity regime (VPIN > threshold):
            - Long when price above MA (follow uptrend)
            - Short when price below MA (follow downtrend)
        
        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Create realistic intraday data
        >>> dates = pd.date_range('2020-01-01', periods=1000, freq='5min')
        >>> 
        >>> # Simulate price with regime changes
        >>> regime1 = np.cumsum(np.random.randn(500) * 0.05)  # noisy
        >>> regime2 = np.cumsum(np.random.randn(500) * 0.1 + 0.02)  # trending
        >>> prices = np.concatenate([regime1, regime2]) + 100
        >>> 
        >>> df = pd.DataFrame({
        ...     'Close': prices,
        ...     'High': prices + np.abs(np.random.randn(1000) * 0.1),
        ...     'Low': prices - np.abs(np.random.randn(1000) * 0.1),
        ...     'Volume': np.random.randint(5000, 20000, 1000)
        ... }, index=dates)
        >>> 
        >>> strategy = LiquidityProvisionStrategy(
        ...     vpin_window=50,
        ...     vpin_threshold=0.3,
        ...     lookback=20
        ... )
        >>> signals = strategy.generate_signals(df)
        >>> 
        >>> # Analyze strategy behavior
        >>> vpin_calc = vpin(df, bucket_volume=df['Volume'].mean()*10, window_buckets=50)
        >>> high_toxicity = vpin_calc > 0.3
        >>> print(f"High toxicity periods: {high_toxicity.sum()}")
        >>> print(f"Mean reversion trades: {(~high_toxicity & (signals != 0)).sum()}")
        >>> print(f"Momentum trades: {(high_toxicity & (signals != 0)).sum()}")
        
        Notes
        -----
        - Returns Series of zeros if insufficient data (< 2 * lookback).
        - Bucket volume for VPIN is dynamically estimated as mean(volume) * 10.
        - If bucket volume calculation produces NaN or zero, uses simple mean volume.
        - VPIN values typically range from 0.1-0.4 in normal markets, spike to 
          0.6-0.9 during informed trading or stress events.
        - The strategy naturally reduces position-taking during high-toxicity periods 
          when bid-ask spreads widen and adverse selection risk increases.
        """
        if len(df) < self.lookback * 2:
            return pd.Series(0, index=df.index, dtype=int)
            
        bucket_vol = df['Volume'].rolling(100).mean().iloc[-1] * 10
        if np.isnan(bucket_vol) or bucket_vol == 0:
            bucket_vol = df['Volume'].mean() * 10

        vpin_series = vpin(df, bucket_volume=bucket_vol, window_buckets=self.vpin_window)
        
        signals = pd.Series(0, index=df.index, dtype=int)
        
        ret = df['Close'].pct_change()
        
        ma = df['Close'].rolling(self.lookback).mean()
        mr_signal = np.where(df['Close'] > ma, -1, 1)
        
        mom_signal = np.where(df['Close'] > ma, 1, -1)
        
        is_toxic = vpin_series > self.vpin_threshold
        
        signals_arr = np.where(is_toxic, mom_signal, mr_signal)
        
        return pd.Series(signals_arr, index=df.index, dtype=int)
