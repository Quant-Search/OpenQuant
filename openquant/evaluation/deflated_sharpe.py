"""Deflated Sharpe Ratio (DSR) utilities without external deps.

Multiple-testing adjustment via Šidák correction.
We approximate the normal CDF and its inverse to avoid SciPy dependency.
"""
from __future__ import annotations
import math
import numpy as np

# Normal CDF using error function
_def_sqrt2 = math.sqrt(2.0)

def _ncdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _def_sqrt2))

# Inverse normal CDF using Acklam's approximation (good to ~1e-9)
# Coefficients from Peter J. Acklam, 2003
_a = [ -3.969683028665376e+01,  2.209460984245205e+02,
      -2.759285104469687e+02,  1.383577518672690e+02,
      -3.066479806614716e+01,  2.506628277459239e+00 ]
_b = [ -5.447609879822406e+01,  1.615858368580409e+02,
      -1.556989798598866e+02,  6.680131188771972e+01,
      -1.328068155288572e+01 ]
_c = [ -7.784894002430293e-03, -3.223964580411365e-01,
      -2.400758277161838e+00, -2.549732539343734e+00,
       4.374664141464968e+00,  2.938163982698783e+00 ]
_d = [ 7.784695709041462e-03,  3.224671290700398e-01,
       2.445134137142996e+00,  3.754408661907416e+00 ]

def _nppf(p: float) -> float:
    if not (0.0 < p < 1.0):
        if p <= 0.0:
            return -float("inf")
        if p >= 1.0:
            return float("inf")
    # Break-points
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2.0*math.log(p))
        return (((((_c[0]*q+_c[1])*q+_c[2])*q+_c[3])*q+_c[4])*q+_c[5]) / \
               (((( _d[0]*q+_d[1])*q+_d[2])*q+_d[3])*q+1.0)
    if phigh < p:
        q = math.sqrt(-2.0*math.log(1.0-p))
        return -(((((_c[0]*q+_c[1])*q+_c[2])*q+_c[3])*q+_c[4])*q+_c[5]) / \
                 (((( _d[0]*q+_d[1])*q+_d[2])*q+_d[3])*q+1.0)
    q = p - 0.5
    r = q*q
    return (((((_a[0]*r+_a[1])*r+_a[2])*r+_a[3])*r+_a[4])*r+_a[5])*q / \
           (((((_b[0]*r+_b[1])*r+_b[2])*r+_b[3])*r+_b[4])*r+1.0)


def deflated_sharpe_ratio(sharpe: float, T: int, trials: int = 1) -> float:
    """Compute the Deflated Sharpe Ratio.
    
    Adjusts the Sharpe ratio to account for:
    - Multiple testing (trials)
    - Sample size (T)
    
    Args:
        sharpe: Observed Sharpe ratio
        T: Number of observations (bars)
        trials: Number of trials/strategies tested
        
    Returns:
        Deflated Sharpe Ratio (can be negative)
    """
    if T <= 1:
        return sharpe
    # Ensure trials >= 1
    trials = max(1, int(trials))
    
    # Variance of Sharpe estimate
    var_sr = (1 + 0.5 * sharpe**2) / T
    std_sr = var_sr**0.5
    
    # Expected max Sharpe under null (no skill)
    # Using Euler-Mascheroni constant γ ≈ 0.5772
    gamma = 0.5772156649
    E_max_sr = (2 * math.log(trials))**0.5 - gamma / (2 * (2 * math.log(trials))**0.5)
    
    # Deflated SR
    dsr = (sharpe - E_max_sr) / std_sr if std_sr > 0 else sharpe
    return float(dsr)

def deflated_sharpe_ratio_with_complexity(
    sharpe: float, 
    T: int, 
    trials: int = 1, 
    num_params: int = 1,
    complexity_penalty: float = 0.1
) -> float:
    """DSR with complexity penalty.
    
    Additional Args:
        num_params: Number of tunable parameters in the strategy
        complexity_penalty: Penalty factor per parameter (e.g., 0.1)
    """
    base_dsr = deflated_sharpe_ratio(sharpe, T, trials)
    # Penalize for each parameter beyond 1
    penalty = complexity_penalty * max(0, num_params - 1)
    return base_dsr - penalty
