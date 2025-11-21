"""Statistical Arbitrage Strategy (Pairs Trading).

Uses Kalman Filter to dynamically estimate the hedge ratio and trade the spread.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from openquant.strategies.base import BaseStrategy
from openquant.quant.filtering import KalmanRegression

class StatArbStrategy(BaseStrategy):
    """
    Kalman Filter based Pairs Trading.
    
    Logic:
    1. Estimate y = alpha + beta * x dynamically.
    2. Calculate Spread = y - (alpha + beta * x).
    3. Calculate Z-Score of Spread.
    4. Long Spread (Long Y, Short X) if Z < -Threshold.
    5. Short Spread (Short Y, Long X) if Z > Threshold.
    6. Exit when Z reverts to 0 (or Stop Loss).
    """
    def __init__(self, pair_symbol: str = None, entry_z: float = 2.0, exit_z: float = 0.0, lookback: int = 100):
        """
        pair_symbol: The 'X' symbol (independent variable). The main symbol of the strategy is 'Y'.
        """
        super().__init__()
        self.pair_symbol = pair_symbol
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.kf = KalmanRegression(delta=1e-4)
        self.spread_std_window = lookback
        self.spread_history = []
        
    def generate_signals(self, df: pd.DataFrame, pair_df: pd.DataFrame = None) -> pd.Series:
        """
        Requires 'pair_df' to be passed (the data for the second asset).
        This signature differs from standard single-asset strategies.
        """
        if pair_df is None:
            # Cannot run without pair data
            return pd.Series(0, index=df.index)
            
        # Align data
        data = pd.concat([df['Close'], pair_df['Close']], axis=1).dropna()
        data.columns = ['Y', 'X']
        
        signals = pd.Series(0, index=data.index)
        
        # Running Kalman Filter loop
        # Ideally this should be vectorized or optimized, but KF is iterative.
        
        kf = KalmanRegression(delta=1e-4)
        z_scores = []
        
        # We need a rolling window for spread variance to normalize Z-score
        # Or use the KF's internal variance estimate (sqrt_Q)
        
        for i in range(len(data)):
            y = data.iloc[i]['Y']
            x = data.iloc[i]['X']
            
            # Predict
            # S = H P H' + R
            H = np.array([[1.0, x]])
            S = H @ kf.kf.P @ H.T + kf.kf.R
            std_dev = np.sqrt(S[0, 0])
            
            # Error (Spread)
            err = kf.get_prediction_error(x, y)
            
            # Z-Score = Error / Predicted_Std_Dev
            z = err / std_dev if std_dev > 0 else 0
            z_scores.append(z)
            
            # Update
            kf.update(x, y)
            
        z_series = pd.Series(z_scores, index=data.index)
        
        # Generate Signals
        curr_pos = 0 # 1 = Long Spread, -1 = Short Spread
        
        for i in range(1, len(z_series)):
            z = z_series.iloc[i]
            idx = z_series.index[i]
            
            sig = 0
            if curr_pos == 0:
                if z < -self.entry_z:
                    sig = 1 # Long Spread (Long Y, Short X)
                    curr_pos = 1
                elif z > self.entry_z:
                    sig = -1 # Short Spread (Short Y, Long X)
                    curr_pos = -1
            elif curr_pos == 1:
                if z >= self.exit_z:
                    sig = 0 # Exit
                    curr_pos = 0
                else:
                    sig = 1 # Hold
            elif curr_pos == -1:
                if z <= -self.exit_z:
                    sig = 0 # Exit
                    curr_pos = 0
                else:
                    sig = -1 # Hold
            
            signals.loc[idx] = sig
            
        return signals

    def get_hedge_ratio(self) -> float:
        """Return current beta."""
        return self.kf.kf.x[1, 0]
