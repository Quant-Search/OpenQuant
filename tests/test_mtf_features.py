"""Test multi-timeframe features in MLStrategy.

Verifies that MLStrategy generates 4h and 1d features.
"""
import pytest
import pandas as pd
import numpy as np
from openquant.strategies.ml_strategy import MLStrategy

@pytest.fixture
def sample_data():
    # 10 days of hourly data
    n = 24 * 10
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    close = 100 * (1 + np.random.randn(n) * 0.01).cumprod()
    df = pd.DataFrame({
        'Open': close,
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    return df

def test_mtf_features(sample_data):
    strategy = MLStrategy()
    features = strategy._generate_features(sample_data)
    
    assert not features.empty
    
    # Check for new columns
    assert 'trend_4h' in features.columns
    assert 'rsi_4h' in features.columns
    assert 'trend_1d' in features.columns
    assert 'rsi_1d' in features.columns
    
    # Check that they are not all NaN
    # Note: first few rows will be NaN due to rolling windows on higher TFs
    # But eventually they should be filled
    assert features['trend_4h'].notna().sum() > 0
    assert features['trend_1d'].notna().sum() > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
