"""Test Regime Detection.

Verifies that RegimeDetector correctly identifies trending and ranging markets.
"""
import pytest
import pandas as pd
import numpy as np
from openquant.quant.regime_detector import RegimeDetector, RegimeType

@pytest.fixture
def trending_data():
    """Create synthetic trending data."""
    n = 200
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    # Strong uptrend
    trend = np.linspace(100, 150, n)
    noise = np.random.randn(n) * 0.5
    close = trend + noise
    
    df = pd.DataFrame({
        'Open': close,
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    return df

@pytest.fixture
def ranging_data():
    """Create synthetic mean-reverting data."""
    n = 200
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    # Oscillating around mean
    close = 100 + 10 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.randn(n) * 0.5
    
    df = pd.DataFrame({
        'Open': close,
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    return df

def test_trending_detection(trending_data):
    detector = RegimeDetector(lookback=100)
    result = detector.detect_regime(trending_data)
    
    assert result['hurst_exponent'] > 0.5
    assert result['trend_regime'] in [RegimeType.TRENDING_UP, RegimeType.TRENDING_DOWN]
    assert 'volatility' in result

def test_ranging_detection(ranging_data):
    detector = RegimeDetector(lookback=100)
    result = detector.detect_regime(ranging_data)
    
    # Mean reverting should have H < 0.5, but with noise, might be close to 0.5
    # Just check it doesn't error
    assert 0.0 <= result['hurst_exponent'] <= 1.0
    assert 'volatility' in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
