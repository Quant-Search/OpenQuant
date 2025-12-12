"""Test GPU features integration in MLStrategy.

Verifies that MLStrategy can generate features using GPU implementation.
"""
import pytest
import pandas as pd
import numpy as np
from openquant.strategies.ml_strategy import MLStrategy
from openquant.features.gpu_features import is_gpu_features_available

@pytest.fixture
def sample_data():
    n = 200
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

def test_feature_generation(sample_data):
    """Test feature generation works (GPU or CPU)."""
    strategy = MLStrategy()
    features = strategy._generate_features(sample_data)
    
    assert not features.empty
    assert 'rsi' in features.columns
    assert 'vol_20' in features.columns
    
    # Check for NaNs (should be dropped)
    assert not features.isnull().values.any()

def test_gpu_features_specific(sample_data):
    """Test specifically GPU path if available."""
    if not is_gpu_features_available():
        pytest.skip("GPU not available")
        
    strategy = MLStrategy()
    # Force GPU usage is implicit if available
    features = strategy._generate_features(sample_data)
    
    assert not features.empty
    print("GPU features generated successfully")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
