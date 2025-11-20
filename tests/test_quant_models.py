"""Verification script for Advanced Quant Models."""
import numpy as np
import pandas as pd
import pytest
from openquant.quant.stationarity import adf_test, hurst_exponent, classify_regime
from openquant.quant.cointegration import engle_granger_test
from openquant.quant.filtering import run_kalman_strategy

def test_stationarity():
    print("\n--- Testing Stationarity ---")
    # 1. Generate Mean Reverting Series (OU Process)
    np.random.seed(42)
    n = 2000
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = ou[i-1] - 0.1 * ou[i-1] + np.random.normal()
    
    res_ou = adf_test(ou)
    hurst_ou = hurst_exponent(ou)
    print(f"Mean Reverting (OU): ADF p={res_ou['p_value']:.4f}, Hurst={hurst_ou:.2f}")
    assert res_ou['is_stationary'] == True
    assert hurst_ou < 0.5
    
    # 2. Generate Random Walk
    rw = np.cumsum(np.random.normal(size=n))
    res_rw = adf_test(rw)
    hurst_rw = hurst_exponent(rw)
    print(f"Random Walk: ADF p={res_rw['p_value']:.4f}, Hurst={hurst_rw:.2f}")
    assert res_rw['is_stationary'] == False
    assert 0.4 < hurst_rw < 0.6

def test_cointegration():
    print("\n--- Testing Cointegration ---")
    np.random.seed(42)
    n = 500
    x = np.cumsum(np.random.normal(size=n))
    # y = 2x + noise (Cointegrated)
    y = 2 * x + np.random.normal(size=n) * 5
    
    res = engle_granger_test(pd.Series(y), pd.Series(x))
    print(f"Cointegrated Pair: p={res['p_value']:.4f}, Beta={res['hedge_ratio']:.2f}")
    assert res['is_cointegrated'] == True
    assert 1.8 < res['hedge_ratio'] < 2.2

def test_kalman():
    print("\n--- Testing Kalman Filter ---")
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 100, n)
    # Dynamic Beta: starts at 1.0, moves to 2.0
    true_beta = np.linspace(1.0, 2.0, n)
    y = true_beta * x + np.random.normal(size=n) * 2
    
    res = run_kalman_strategy(pd.Series(y), pd.Series(x))
    final_beta = res['beta'].iloc[-1]
    print(f"Kalman Estimated Beta: {final_beta:.2f} (True: 2.0)")
    assert 1.8 < final_beta < 2.2

if __name__ == "__main__":
    test_stationarity()
    test_cointegration()
    test_kalman()
    print("\n✅ All Quant Tests Passed!")
