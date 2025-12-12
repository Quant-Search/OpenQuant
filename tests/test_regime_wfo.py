"""Tests for regime-specific WFO functions."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openquant.evaluation.wfo import (
    walk_forward_evaluate_regime_specific,
    compare_strategies_by_regime,
    WFOSpec
)
from openquant.strategies.regime_adaptive import RegimeAdaptiveStrategy
from openquant.strategies.quant.hurst import HurstExponentStrategy
from openquant.strategies.quant.stat_arb import StatArbStrategy


@pytest.fixture
def large_sample_df():
    """Generate larger sample OHLCV data for WFO testing."""
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    price = 100.0
    prices = []
    for i in range(1000):
        if i < 333:
            drift = 0.001
            vol = 0.003
        elif i < 666:
            drift = 0.0
            vol = 0.01
        else:
            drift = -0.0005
            vol = 0.005
        
        price *= (1 + drift + vol * np.random.randn())
        prices.append(price)
    
    close_prices = np.array(prices)
    high_prices = close_prices * (1 + np.abs(np.random.randn(1000) * 0.005))
    low_prices = close_prices * (1 - np.abs(np.random.randn(1000) * 0.005))
    open_prices = close_prices * (1 + np.random.randn(1000) * 0.003)
    volume = np.random.uniform(1000, 10000, 1000)
    
    return pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)


def test_regime_specific_wfo_basic(large_sample_df):
    """Test basic regime-specific WFO."""
    def strategy_factory(lookback=100):
        return RegimeAdaptiveStrategy(lookback=lookback)
    
    param_grid = {'lookback': [80, 100]}
    wfo_spec = WFOSpec(n_splits=2, train_frac=0.7)
    
    results = walk_forward_evaluate_regime_specific(
        df=large_sample_df,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        fee_bps=2.0,
        weight=1.0,
        wfo=wfo_spec
    )
    
    assert 'test_sharpes' in results
    assert 'mean_test_sharpe' in results
    assert 'best_params_per_split' in results
    assert 'regime_performance' in results
    assert 'regime_distribution' in results
    
    assert isinstance(results['test_sharpes'], list)
    assert isinstance(results['mean_test_sharpe'], float)
    assert isinstance(results['regime_performance'], dict)
    assert isinstance(results['regime_distribution'], dict)


def test_regime_performance_keys(large_sample_df):
    """Test that regime performance has expected keys."""
    def strategy_factory(lookback=100):
        return RegimeAdaptiveStrategy(lookback=lookback)
    
    param_grid = {'lookback': [100]}
    wfo_spec = WFOSpec(n_splits=2, train_frac=0.7)
    
    results = walk_forward_evaluate_regime_specific(
        df=large_sample_df,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        fee_bps=2.0,
        weight=1.0,
        wfo=wfo_spec
    )
    
    for regime in ['trending', 'mean_reverting', 'volatile', 'neutral']:
        assert regime in results['regime_performance']
        perf = results['regime_performance'][regime]
        assert 'sharpe' in perf
        assert 'mean_return' in perf
        assert 'std_return' in perf
        assert 'num_periods' in perf


def test_regime_distribution_keys(large_sample_df):
    """Test that regime distribution has expected keys."""
    def strategy_factory(lookback=100):
        return RegimeAdaptiveStrategy(lookback=lookback)
    
    param_grid = {'lookback': [100]}
    wfo_spec = WFOSpec(n_splits=2, train_frac=0.7)
    
    results = walk_forward_evaluate_regime_specific(
        df=large_sample_df,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        fee_bps=2.0,
        weight=1.0,
        wfo=wfo_spec
    )
    
    for regime in ['trending', 'mean_reverting', 'volatile', 'neutral']:
        assert regime in results['regime_distribution']
        dist = results['regime_distribution'][regime]
        assert 'count' in dist
        assert 'percentage' in dist


def test_custom_regime_classifier(large_sample_df):
    """Test WFO with custom regime classifier."""
    def strategy_factory(lookback=100):
        return RegimeAdaptiveStrategy(lookback=lookback)
    
    def custom_classifier(df_window: pd.DataFrame) -> str:
        returns = df_window['Close'].pct_change().tail(50)
        vol = returns.std()
        
        if vol > 0.015:
            return 'volatile'
        else:
            return 'normal'
    
    param_grid = {'lookback': [100]}
    wfo_spec = WFOSpec(n_splits=2, train_frac=0.7)
    
    results = walk_forward_evaluate_regime_specific(
        df=large_sample_df,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        fee_bps=2.0,
        weight=1.0,
        wfo=wfo_spec,
        regime_classifier=custom_classifier
    )
    
    assert 'regime_performance' in results
    assert 'regime_distribution' in results


def test_compare_strategies_basic(large_sample_df):
    """Test basic strategy comparison by regime."""
    strategies = {
        'RegimeAdaptive': RegimeAdaptiveStrategy(lookback=100),
        'HurstOnly': HurstExponentStrategy(lookback=100),
        'StatArbOnly': StatArbStrategy(lookback=100)
    }
    
    comparison_df = compare_strategies_by_regime(
        df=large_sample_df,
        strategies=strategies,
        fee_bps=2.0,
        weight=1.0
    )
    
    assert isinstance(comparison_df, pd.DataFrame)
    assert not comparison_df.empty
    assert 'strategy' in comparison_df.columns
    assert 'regime' in comparison_df.columns
    assert 'sharpe' in comparison_df.columns
    assert 'mean_return' in comparison_df.columns
    assert 'std_return' in comparison_df.columns
    assert 'num_periods' in comparison_df.columns


def test_compare_strategies_has_overall(large_sample_df):
    """Test that comparison includes overall metrics."""
    strategies = {
        'RegimeAdaptive': RegimeAdaptiveStrategy(lookback=100),
    }
    
    comparison_df = compare_strategies_by_regime(
        df=large_sample_df,
        strategies=strategies,
        fee_bps=2.0,
        weight=1.0
    )
    
    overall_results = comparison_df[comparison_df['regime'] == 'overall']
    assert not overall_results.empty
    assert len(overall_results) == len(strategies)


def test_compare_strategies_custom_classifier(large_sample_df):
    """Test strategy comparison with custom classifier."""
    strategies = {
        'RegimeAdaptive': RegimeAdaptiveStrategy(lookback=100),
    }
    
    def custom_classifier(df_window: pd.DataFrame) -> str:
        vol = df_window['Close'].pct_change().tail(50).std()
        return 'high_vol' if vol > 0.01 else 'low_vol'
    
    comparison_df = compare_strategies_by_regime(
        df=large_sample_df,
        strategies=strategies,
        fee_bps=2.0,
        weight=1.0,
        regime_classifier=custom_classifier
    )
    
    assert isinstance(comparison_df, pd.DataFrame)
    assert not comparison_df.empty


def test_wfo_with_multiple_params(large_sample_df):
    """Test WFO with multiple parameter combinations."""
    def strategy_factory(lookback=100, vol_reduce_factor=0.5):
        return RegimeAdaptiveStrategy(
            lookback=lookback,
            vol_reduce_factor=vol_reduce_factor
        )
    
    param_grid = {
        'lookback': [80, 100, 120],
        'vol_reduce_factor': [0.3, 0.5, 0.7]
    }
    
    wfo_spec = WFOSpec(n_splits=2, train_frac=0.7)
    
    results = walk_forward_evaluate_regime_specific(
        df=large_sample_df,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        fee_bps=2.0,
        weight=1.0,
        wfo=wfo_spec
    )
    
    assert len(results['test_sharpes']) > 0
    assert len(results['best_params_per_split']) > 0


def test_empty_strategies_dict(large_sample_df):
    """Test comparison with empty strategies dict."""
    strategies = {}
    
    comparison_df = compare_strategies_by_regime(
        df=large_sample_df,
        strategies=strategies,
        fee_bps=2.0,
        weight=1.0
    )
    
    assert isinstance(comparison_df, pd.DataFrame)
    assert comparison_df.empty


def test_wfo_spec_parameters():
    """Test WFOSpec initialization."""
    wfo = WFOSpec(n_splits=5, train_frac=0.8)
    
    assert wfo.n_splits == 5
    assert wfo.train_frac == 0.8
    assert wfo.use_cpcv is False


def test_wfo_with_different_fees(large_sample_df):
    """Test WFO with different fee settings."""
    def strategy_factory(lookback=100):
        return RegimeAdaptiveStrategy(lookback=lookback)
    
    param_grid = {'lookback': [100]}
    wfo_spec = WFOSpec(n_splits=2, train_frac=0.7)
    
    results_low_fee = walk_forward_evaluate_regime_specific(
        df=large_sample_df,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        fee_bps=1.0,
        weight=1.0,
        wfo=wfo_spec
    )
    
    results_high_fee = walk_forward_evaluate_regime_specific(
        df=large_sample_df,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        fee_bps=5.0,
        weight=1.0,
        wfo=wfo_spec
    )
    
    assert 'mean_test_sharpe' in results_low_fee
    assert 'mean_test_sharpe' in results_high_fee
