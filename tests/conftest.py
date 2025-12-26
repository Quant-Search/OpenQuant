"""
Pytest Configuration and Fixtures
==================================
Shared fixtures and configuration for all tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 500
    
    # Generate random walk prices
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='h')
    returns = np.random.normal(0, 0.001, n_bars)
    close = 1.1000 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n_bars)))
    open_prices = np.roll(close, 1)
    open_prices[0] = 1.1000
    volume = np.random.randint(100, 10000, n_bars).astype(float)
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    return df


@pytest.fixture
def small_ohlcv_data() -> pd.DataFrame:
    """Generate small OHLCV data (insufficient for strategy)."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=20, freq='h')
    
    return pd.DataFrame({
        'Open': np.linspace(1.10, 1.12, 20),
        'High': np.linspace(1.105, 1.125, 20),
        'Low': np.linspace(1.095, 1.115, 20),
        'Close': np.linspace(1.10, 1.12, 20),
        'Volume': np.ones(20) * 1000
    }, index=dates)


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Return empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def trending_up_data() -> pd.DataFrame:
    """Generate strongly trending up data."""
    np.random.seed(42)
    n_bars = 200
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='h')
    
    # Strong uptrend
    trend = np.linspace(1.10, 1.20, n_bars)
    noise = np.random.normal(0, 0.001, n_bars)
    close = trend + noise
    
    return pd.DataFrame({
        'Open': np.roll(close, 1),
        'High': close * 1.002,
        'Low': close * 0.998,
        'Close': close,
        'Volume': np.random.randint(100, 10000, n_bars).astype(float)
    }, index=dates)


@pytest.fixture
def trending_down_data() -> pd.DataFrame:
    """Generate strongly trending down data."""
    np.random.seed(42)
    n_bars = 200
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='h')
    
    # Strong downtrend
    trend = np.linspace(1.20, 1.10, n_bars)
    noise = np.random.normal(0, 0.001, n_bars)
    close = trend + noise
    
    return pd.DataFrame({
        'Open': np.roll(close, 1),
        'High': close * 1.002,
        'Low': close * 0.998,
        'Close': close,
        'Volume': np.random.randint(100, 10000, n_bars).astype(float)
    }, index=dates)


@pytest.fixture
def mean_reverting_data() -> pd.DataFrame:
    """Generate mean-reverting data (ideal for Kalman strategy)."""
    np.random.seed(42)
    n_bars = 300
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='h')
    
    # Mean-reverting around 1.10
    mean = 1.10
    close = mean + 0.01 * np.sin(np.linspace(0, 10*np.pi, n_bars))
    close += np.random.normal(0, 0.001, n_bars)
    
    return pd.DataFrame({
        'Open': np.roll(close, 1),
        'High': close * 1.002,
        'Low': close * 0.998,
        'Close': close,
        'Volume': np.random.randint(100, 10000, n_bars).astype(float)
    }, index=dates)


@pytest.fixture
def data_with_nan() -> pd.DataFrame:
    """Generate data with NaN values for edge case testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    close = np.linspace(1.10, 1.12, 100)
    close[10] = np.nan
    close[50] = np.nan
    
    return pd.DataFrame({
        'Open': close,
        'High': close * 1.002,
        'Low': close * 0.998,
        'Close': close,
        'Volume': np.ones(100) * 1000
    }, index=dates)

