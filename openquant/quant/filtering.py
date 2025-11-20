"""Kalman Filter for Dynamic Hedge Ratio Estimation.

Uses a Kalman Filter to estimate the time-varying slope (beta) and intercept (alpha)
of the relationship between two assets: Y = alpha + beta * X + noise.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from filterpy.kalman import KalmanFilter

class KalmanRegression:
    """
    Online Kalman Filter for linear regression: y = alpha + beta * x
    State vector x = [alpha, beta]
    """
    def __init__(self, delta: float = 1e-5, r: float = 1e-3):
        """
        delta: Process noise covariance (allows parameters to drift).
        r: Measurement noise covariance.
        """
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # State transition matrix (Identity: parameters are random walks)
        self.kf.F = np.eye(2)
        
        # Measurement function H will be updated at each step: H = [1, x_t]
        
        # Initial state covariance
        self.kf.P *= 1.0
        
        # Process noise covariance Q
        # Small drift allowed
        self.kf.Q = np.eye(2) * delta
        
        # Measurement noise covariance R
        self.kf.R = np.eye(1) * r
        
        # Initial state
        self.kf.x = np.zeros((2, 1))

    def update(self, x_t: float, y_t: float) -> Tuple[float, float]:
        """
        Update filter with new observation (x_t, y_t).
        Returns estimated (alpha, beta).
        """
        # Measurement matrix H = [1, x_t]
        self.kf.H = np.array([[1.0, x_t]])
        
        # Predict step
        self.kf.predict()
        
        # Update step
        self.kf.update(np.array([[y_t]]))
        
        alpha = self.kf.x[0, 0]
        beta = self.kf.x[1, 0]
        return alpha, beta

    def get_prediction_error(self, x_t: float, y_t: float) -> float:
        """Calculate prediction error (innovation) before update."""
        # y_pred = alpha + beta * x
        alpha = self.kf.x[0, 0]
        beta = self.kf.x[1, 0]
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
    
    alphas = []
    betas = []
    errors = [] # The "spread"
    
    # We also need the variance of the prediction error to calculate Z-score properly
    # Q_t = H P H' + R
    sqrt_Q = []
    
    for i in range(len(y_vals)):
        y = y_vals[i]
        x = x_vals[i]
        
        # Get error stats before update (one-step ahead prediction)
        # H = [1, x]
        H = np.array([[1.0, x]])
        # S = H P H' + R (System uncertainty)
        S = H @ kf.kf.P @ H.T + kf.kf.R
        std_dev = np.sqrt(S[0, 0])
        sqrt_Q.append(std_dev)
        
        # Prediction error
        err = kf.get_prediction_error(x, y)
        errors.append(err)
        
        # Update
        a, b = kf.update(x, y)
        alphas.append(a)
        betas.append(b)
        
    res = pd.DataFrame(index=df.index)
    res['alpha'] = alphas
    res['beta'] = betas
    res['spread'] = errors # Innovation
    res['spread_std'] = sqrt_Q
    res['z_score'] = res['spread'] / res['spread_std']
    
    return res
