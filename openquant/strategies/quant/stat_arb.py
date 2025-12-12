"""Statistical Arbitrage Strategy (Pairs Trading).

Uses Kalman Filter to dynamically estimate the hedge ratio and trade the spread.
"""

import numpy as np
import pandas as pd

from openquant.quant.filtering import KalmanRegression
from openquant.strategies.base import BaseStrategy
from openquant.utils.validation import (
    validate_params,
    validate_positive_param,
    validate_positive_int_param,
    validate_range_param
)


class StatArbStrategy(BaseStrategy):
    """
    Kalman Filter-based Pairs Trading and Mean Reversion Strategy.
    
    This strategy implements two modes:
    1. **Pairs Trading Mode**: Uses Kalman Filter to dynamically estimate the hedge ratio 
       between two cointegrated assets and trades the spread.
    2. **Single Asset Mean Reversion Mode**: Trades based on z-score deviations from 
       a moving average.
    
    Mathematical Model
    ------------------
    For pairs trading, the linear relationship is modeled as:
    
    .. math::
        y_t = \\alpha_t + \\beta_t x_t + \\epsilon_t
    
    where:
    - :math:`y_t` is the dependent asset price at time t
    - :math:`x_t` is the independent asset price at time t
    - :math:`\\alpha_t, \\beta_t` are time-varying parameters estimated by Kalman Filter
    - :math:`\\epsilon_t` is the spread (residual)
    
    The spread is calculated as:
    
    .. math::
        S_t = y_t - (\\alpha_t + \\beta_t x_t)
    
    The z-score of the spread is:
    
    .. math::
        Z_t = \\frac{S_t}{\\sigma_t}
    
    where :math:`\\sigma_t` is the standard deviation estimated from the Kalman Filter's 
    covariance matrix.
    
    For single asset mean reversion:
    
    .. math::
        Z_t = \\frac{P_t - \\mu_t}{\\sigma_t}
    
    where:
    - :math:`P_t` is the price at time t
    - :math:`\\mu_t` is the moving average over the lookback window
    - :math:`\\sigma_t` is the rolling standard deviation
    
    Trading Signals
    ---------------
    Entry Signals:
    - Long (1): :math:`Z_t < -\\text{entry\\_z}` (spread is undervalued)
    - Short (-1): :math:`Z_t > \\text{entry\\_z}` (spread is overvalued)
    
    Exit Signals:
    - Exit Long: :math:`Z_t \\geq \\text{exit\\_z}`
    - Exit Short: :math:`Z_t \\leq -\\text{exit\\_z}`
    
    Parameters
    ----------
    pair_symbol : str, optional
        The symbol of the independent variable (X) for pairs trading mode.
        If None, operates in single asset mean reversion mode.
        Default is None.
    entry_z : float, optional
        Z-score threshold for entering positions. Entry occurs when 
        |z-score| exceeds this value.
        Default is 2.0.
    exit_z : float, optional
        Z-score threshold for exiting positions. Exit occurs when 
        z-score reverts to this value.
        Default is 0.0.
    lookback : int, optional
        Window size for calculating rolling statistics (mean and standard deviation) 
        in single asset mode.
        Default is 100.
    
    Attributes
    ----------
    pair_symbol : str or None
        Symbol of the paired asset (X).
    entry_z : float
        Entry z-score threshold.
    exit_z : float
        Exit z-score threshold.
    kf : KalmanRegression
        Kalman Filter instance for dynamic hedge ratio estimation.
    spread_std_window : int
        Lookback window for spread statistics.
    spread_history : list
        Historical spread values.
    
    Examples
    --------
    >>> # Pairs trading mode
    >>> strategy = StatArbStrategy(pair_symbol='SPY', entry_z=2.5, exit_z=0.5)
    >>> signals = strategy.generate_signals(df_asset_y, pair_df=df_asset_x)
    >>> 
    >>> # Single asset mean reversion mode
    >>> strategy = StatArbStrategy(entry_z=2.0, exit_z=0.0, lookback=100)
    >>> signals = strategy.generate_signals(df)
    
    Notes
    -----
    - The Kalman Filter provides adaptive hedge ratio estimation that responds to 
      regime changes in the market.
    - The process noise parameter (delta) in KalmanRegression controls the filter's 
      responsiveness to changes in the hedge ratio.
    - In pairs trading, both legs should be executed simultaneously to maintain 
      the hedge ratio :math:`\\beta_t`.
    
    References
    ----------
    .. [1] Pole, A. (2007). Statistical Arbitrage: Algorithmic Trading Insights 
           and Techniques. Wiley.
    .. [2] Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction 
           Problems". Journal of Basic Engineering.
    """
    @validate_params(
        entry_z=validate_positive_param('entry_z'),
        exit_z=validate_range_param('exit_z', min_val=0.0),
        lookback=validate_positive_int_param('lookback')
    )
    def __init__(self, pair_symbol: str | None = None, entry_z: float = 2.0, exit_z: float = 0.0, lookback: int = 100) -> None:
        """
        Initialize the Statistical Arbitrage Strategy.
        
        Parameters
        ----------
        pair_symbol : str, optional
            The symbol of the independent variable (X) for pairs trading mode.
            If None, operates in single asset mean reversion mode.
        entry_z : float, optional
            Z-score threshold for entering positions.
        exit_z : float, optional
            Z-score threshold for exiting positions.
        lookback : int, optional
            Window size for rolling statistics in single asset mode.
        """
        super().__init__()
        self.pair_symbol: str | None = pair_symbol
        self.entry_z: float = entry_z
        self.exit_z: float = exit_z
        self.kf: KalmanRegression = KalmanRegression(delta=1e-4)
        self.spread_std_window: int = lookback
        self.spread_history: list = []

    def generate_signals(self, df: pd.DataFrame, pair_df: pd.DataFrame | None = None) -> pd.Series:
        """
        Generate trading signals based on spread z-scores.
        
        This method operates in two modes:
        1. If pair_df is provided: Pairs trading using Kalman Filter
        2. If pair_df is None: Single asset mean reversion using z-scores
        
        Parameters
        ----------
        df : pd.DataFrame
            Price data for the primary asset (Y). Must contain a 'Close' column.
            Index should be datetime.
        pair_df : pd.DataFrame, optional
            Price data for the paired asset (X) in pairs trading mode.
            Must contain a 'Close' column with matching datetime index.
            If None, uses single asset mean reversion mode.
            Default is None.
        
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
        >>> # Create sample data
        >>> dates = pd.date_range('2020-01-01', periods=200)
        >>> df_y = pd.DataFrame({
        ...     'Close': 100 + np.cumsum(np.random.randn(200) * 0.5)
        ... }, index=dates)
        >>> df_x = pd.DataFrame({
        ...     'Close': 100 + np.cumsum(np.random.randn(200) * 0.5)
        ... }, index=dates)
        >>> 
        >>> # Pairs trading mode
        >>> strategy = StatArbStrategy(pair_symbol='X', entry_z=2.0)
        >>> signals = strategy.generate_signals(df_y, pair_df=df_x)
        >>> 
        >>> # Single asset mode
        >>> strategy_single = StatArbStrategy(entry_z=2.0, lookback=50)
        >>> signals_single = strategy_single.generate_signals(df_y)
        
        Notes
        -----
        The function maintains state across calls via curr_pos to ensure proper 
        position management. Signals are generated sequentially to respect entry 
        and exit conditions.
        """
        signals = pd.Series(0, index=df.index, dtype=int)
        z_scores: list = []

        if pair_df is not None:
            data = pd.concat([df['Close'], pair_df['Close']], axis=1).dropna()
            data.columns = ['Y', 'X']

            kf = KalmanRegression(delta=1e-4)

            for i in range(len(data)):
                y = data.iloc[i]['Y']
                x = data.iloc[i]['X']

                H = np.array([[1.0, x]])
                S = H @ kf.kf.P @ H.T + kf.kf.R
                std_dev = np.sqrt(S[0, 0])

                err = kf.get_prediction_error(x, y)

                z: float = err / std_dev if std_dev > 0 else 0
                z_scores.append(z)

                kf.update(x, y)

            z_series = pd.Series(z_scores, index=data.index)

        else:
            close = df['Close']
            ma = close.rolling(window=self.spread_std_window).mean()
            std = close.rolling(window=self.spread_std_window).std()

            z_series = (close - ma) / std
            z_series = z_series.fillna(0)

        curr_pos: int = 0

        signals = pd.Series(0, index=df.index, dtype=int)

        for idx, z in z_series.items():
            sig: int = 0
            if curr_pos == 0:
                if z < -self.entry_z:
                    sig = 1
                    curr_pos = 1
                elif z > self.entry_z:
                    sig = -1
                    curr_pos = -1
            elif curr_pos == 1:
                if z >= self.exit_z:
                    sig = 0
                    curr_pos = 0
                else:
                    sig = 1
            elif curr_pos == -1:
                if z <= -self.exit_z:
                    sig = 0
                    curr_pos = 0
                else:
                    sig = -1

            signals.loc[idx] = sig

        return signals

    def get_hedge_ratio(self) -> float:
        """
        Return the current hedge ratio (beta) from the Kalman Filter.
        
        The hedge ratio represents the number of units of the independent asset (X) 
        required to hedge one unit of the dependent asset (Y) in pairs trading.
        
        Returns
        -------
        beta : float
            Current hedge ratio :math:`\\beta_t` from the Kalman Filter state.
            Only valid when operating in pairs trading mode.
        
        Examples
        --------
        >>> strategy = StatArbStrategy(pair_symbol='SPY')
        >>> signals = strategy.generate_signals(df_y, pair_df=df_x)
        >>> beta = strategy.get_hedge_ratio()
        >>> print(f"Current hedge ratio: {beta:.4f}")
        
        Notes
        -----
        This method should only be called after generate_signals() has been executed 
        at least once in pairs trading mode. The hedge ratio is time-varying and 
        updates with each new observation.
        """
        return float(self.kf.kf.x[1, 0])
