"""Tests for Regime Adaptive Strategy."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openquant.strategies.regime_adaptive import RegimeAdaptiveStrategy
from openquant.strategies.quant.hurst import HurstExponentStrategy
from openquant.strategies.quant.stat_arb import StatArbStrategy
from openquant.backtest.engine import backtest_signals
from openquant.backtest.metrics import sharpe


@pytest.fixture
def sample_df():
    """Generate sample OHLCV data."""
    dates = pd.date_range(start='2020-01-01', periods=500, freq='1H')
    np.random.seed(42)
    
    price = 100.0
    prices = []
    for _ in range(500):
        price *= (1 + 0.0005 + 0.01 * np.random.randn())
        prices.append(price)
    
    close_prices = np.array(prices)
    high_prices = close_prices * (1 + np.abs(np.random.randn(500) * 0.005))
    low_prices = close_prices * (1 - np.abs(np.random.randn(500) * 0.005))
    open_prices = close_prices * (1 + np.random.randn(500) * 0.003)
    volume = np.random.uniform(1000, 10000, 500)
    
    return pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)


def test_regime_adaptive_initialization():
    """Test RegimeAdaptiveStrategy initialization."""
    strategy = RegimeAdaptiveStrategy(
        lookback=100,
        hurst_threshold_trend=0.55,
        hurst_threshold_mr=0.45,
        vol_reduce_factor=0.5,
        enable_vol_scaling=True
    )
    
    assert strategy.lookback == 100
    assert strategy.hurst_threshold_trend == 0.55
    assert strategy.hurst_threshold_mr == 0.45
    assert strategy.vol_reduce_factor == 0.5
    assert strategy.enable_vol_scaling is True
    assert isinstance(strategy.hurst_strategy, HurstExponentStrategy)
    assert isinstance(strategy.stat_arb_strategy, StatArbStrategy)
    assert strategy.regime_history == []


def test_regime_adaptive_generate_signals(sample_df):
    """Test signal generation."""
    strategy = RegimeAdaptiveStrategy(lookback=50)
    signals = strategy.generate_signals(sample_df)
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(sample_df)
    assert signals.isin([-1, 0, 1]).all()


def test_regime_adaptive_with_short_data():
    """Test with data shorter than lookback."""
    dates = pd.date_range(start='2020-01-01', periods=50, freq='1H')
    df = pd.DataFrame({
        'Open': np.random.randn(50) + 100,
        'High': np.random.randn(50) + 101,
        'Low': np.random.randn(50) + 99,
        'Close': np.random.randn(50) + 100,
        'Volume': np.random.uniform(1000, 10000, 50)
    }, index=dates)
    
    strategy = RegimeAdaptiveStrategy(lookback=100)
    signals = strategy.generate_signals(df)
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(df)
    assert (signals == 0).all()


def test_regime_history_tracking(sample_df):
    """Test regime history tracking."""
    strategy = RegimeAdaptiveStrategy(lookback=50)
    signals = strategy.generate_signals(sample_df)
    
    regime_history = strategy.get_regime_history()
    
    assert isinstance(regime_history, pd.DataFrame)
    assert not regime_history.empty
    assert 'trend_regime' in regime_history.columns
    assert 'volatility_regime' in regime_history.columns
    assert 'hurst_exponent' in regime_history.columns
    assert 'volatility' in regime_history.columns


def test_regime_stats(sample_df):
    """Test regime statistics calculation."""
    strategy = RegimeAdaptiveStrategy(lookback=50)
    signals = strategy.generate_signals(sample_df)
    
    stats = strategy.get_regime_stats()
    
    assert isinstance(stats, dict)
    assert 'trend_regime_distribution' in stats
    assert 'volatility_regime_distribution' in stats
    assert 'mean_hurst_exponent' in stats
    assert 'std_hurst_exponent' in stats
    assert 'mean_volatility' in stats
    assert 'std_volatility' in stats


def test_backtest_integration(sample_df):
    """Test integration with backtest engine."""
    strategy = RegimeAdaptiveStrategy(lookback=50)
    signals = strategy.generate_signals(sample_df)
    
    result = backtest_signals(sample_df, signals, fee_bps=2.0, weight=1.0)
    
    assert result.equity_curve is not None
    assert result.returns is not None
    assert result.positions is not None
    assert len(result.equity_curve) == len(sample_df)


def test_vol_scaling_enabled(sample_df):
    """Test with volatility scaling enabled."""
    strategy_with_scaling = RegimeAdaptiveStrategy(
        lookback=50,
        enable_vol_scaling=True,
        vol_reduce_factor=0.5
    )
    
    signals = strategy_with_scaling.generate_signals(sample_df)
    assert isinstance(signals, pd.Series)


def test_vol_scaling_disabled(sample_df):
    """Test with volatility scaling disabled."""
    strategy_no_scaling = RegimeAdaptiveStrategy(
        lookback=50,
        enable_vol_scaling=False
    )
    
    signals = strategy_no_scaling.generate_signals(sample_df)
    assert isinstance(signals, pd.Series)


def test_custom_hurst_params(sample_df):
    """Test with custom Hurst strategy parameters."""
    hurst_params = {
        'lookback': 80,
        'trend_threshold': 0.6,
        'mr_threshold': 0.4
    }
    
    strategy = RegimeAdaptiveStrategy(
        lookback=50,
        hurst_params=hurst_params
    )
    
    signals = strategy.generate_signals(sample_df)
    assert isinstance(signals, pd.Series)


def test_custom_stat_arb_params(sample_df):
    """Test with custom StatArb strategy parameters."""
    stat_arb_params = {
        'lookback': 80,
        'entry_z': 2.5,
        'exit_z': 0.5
    }
    
    strategy = RegimeAdaptiveStrategy(
        lookback=50,
        stat_arb_params=stat_arb_params
    )
    
    signals = strategy.generate_signals(sample_df)
    assert isinstance(signals, pd.Series)


def test_different_thresholds(sample_df):
    """Test with different regime thresholds."""
    strategy = RegimeAdaptiveStrategy(
        lookback=50,
        hurst_threshold_trend=0.6,
        hurst_threshold_mr=0.4
    )
    
    signals = strategy.generate_signals(sample_df)
    assert isinstance(signals, pd.Series)


def test_empty_dataframe():
    """Test with empty DataFrame."""
    df = pd.DataFrame()
    strategy = RegimeAdaptiveStrategy(lookback=50)
    signals = strategy.generate_signals(df)
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == 0


def test_regime_history_empty():
    """Test regime history when no signals generated."""
    strategy = RegimeAdaptiveStrategy(lookback=50)
    history = strategy.get_regime_history()
    
    assert isinstance(history, pd.DataFrame)
    assert history.empty


def test_regime_stats_empty():
    """Test regime stats when no signals generated."""
    strategy = RegimeAdaptiveStrategy(lookback=50)
    stats = strategy.get_regime_stats()
    
    assert isinstance(stats, dict)
    assert stats == {}
