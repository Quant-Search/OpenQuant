"""Statistical Arbitrage Strategy (Pairs Trading).

Uses Kalman Filter to dynamically estimate the hedge ratio and trade the spread.
"""

import numpy as np
import pandas as pd

from openquant.quant.filtering import KalmanRegression
from openquant.strategies.base import BaseStrategy


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
    def __init__(self, pair_symbol: str | None = None, entry_z: float = 2.0, exit_z: float = 0.0, lookback: int = 100) -> None:
        """
        pair_symbol: The 'X' symbol (independent variable). The main symbol of the strategy is 'Y'.
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
        Generates signals.
        If pair_df is provided, runs Pairs Trading (Cointegration/Kalman).
        If pair_df is None, runs Statistical Mean Reversion (Z-Score against MA).
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
        """Return current beta (only valid for pairs mode)."""
        return float(self.kf.kf.x[1, 0])
