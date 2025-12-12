"""Kalman Filter for Dynamic Hedge Ratio Estimation.

Uses a Kalman Filter to estimate the time-varying slope (beta) and intercept (alpha)
of the relationship between two assets: Y = alpha + beta * X + noise.
"""

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from openquant.utils.validation import validate_params, validate_positive_param


class KalmanRegression:
    """
    Online Kalman Filter for linear regression: y = alpha + beta * x
    State vector x = [alpha, beta]
    """
    @validate_params(
        delta=validate_positive_param('delta'),
        r=validate_positive_param('r')
    )
    def __init__(self, delta: float = 1e-5, r: float = 1e-3) -> None:
        """
        delta: Process noise covariance (allows parameters to drift).
        r: Measurement noise covariance.
        """
        self.kf: KalmanFilter = KalmanFilter(dim_x=2, dim_z=1)

        self.kf.F = np.eye(2)

        self.kf.P *= 1.0

        self.kf.Q = np.eye(2) * delta

        self.kf.R = np.eye(1) * r

        self.kf.x = np.zeros((2, 1))

    def update(self, x_t: float, y_t: float) -> tuple[float, float]:
        """
        Update filter with new observation (x_t, y_t).
        Returns estimated (alpha, beta).
        """
        self.kf.H = np.array([[1.0, x_t]])

        self.kf.predict()

        self.kf.update(np.array([[y_t]]))

        alpha: float = self.kf.x[0, 0]
        beta: float = self.kf.x[1, 0]
        return alpha, beta

    def get_prediction_error(self, x_t: float, y_t: float) -> float:
        """Calculate prediction error (innovation) before update."""
        alpha: float = self.kf.x[0, 0]
        beta: float = self.kf.x[1, 0]
        y_pred = alpha + beta * x_t
        return y_t - y_pred

def run_kalman_strategy(series_y: pd.Series, series_x: pd.Series, delta: float = 1e-5) -> pd.DataFrame:
    """
    Run Kalman Filter over full history to generate spread and Z-score.
    """
    df = pd.concat([series_y, series_x], axis=1).dropna()
    y_vals = df.iloc[:, 0].values
    x_vals = df.iloc[:, 1].values

    kf = KalmanRegression(delta=delta)

    alphas: list = []
    betas: list = []
    errors: list = []

    sqrt_Q: list = []

    for i in range(len(y_vals)):
        y = y_vals[i]
        x = x_vals[i]

        H = np.array([[1.0, x]])
        S = H @ kf.kf.P @ H.T + kf.kf.R
        std_dev = np.sqrt(S[0, 0])
        sqrt_Q.append(std_dev)

        err = kf.get_prediction_error(x, y)
        errors.append(err)

        a, b = kf.update(x, y)
        alphas.append(a)
        betas.append(b)

    res = pd.DataFrame(index=df.index)
    res['alpha'] = alphas
    res['beta'] = betas
    res['spread'] = errors
    res['spread_std'] = sqrt_Q
    res['z_score'] = res['spread'] / res['spread_std']

    return res

def hp_filter(series: pd.Series, lamb: float = 1600) -> tuple[pd.Series, pd.Series]:
    """
    Hodrick-Prescott Filter to separate Trend and Cycle.
    y_t = tau_t + c_t

    Args:
        series: Time series data.
        lamb: Smoothing parameter (1600 for quarterly, 6.25 for annual, 129600 for monthly?).
              For daily crypto data, usually much higher (e.g. 100*252^2?).
              Standard defaults: 1600 (Quarterly), 14400 (Monthly), 100 (Annual).
              For Daily, literature suggests 1600 * (freq/4)^4? No.
              Ravn and Uhlig: lambda = 1600 * (freq/4)^4.
              Daily: 1600 * (365/4)^4 approx 1e8.
              Common practice: try 1600 or adjust visually.

    Returns:
        (trend, cycle) as pd.Series
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    n = len(series)
    if n < 4:
        return series, pd.Series(0, index=series.index)

    data = [1, -2, 1]
    diags = [0, 1, 2]
    D = sp.spdiags([data] * (n-2), diags, n-2, n)

    I = sp.eye(n)
    F = I + lamb * D.T @ D

    tau = spsolve(F, series.values)

    trend = pd.Series(tau, index=series.index)
    cycle = series - trend
    return trend, cycle
