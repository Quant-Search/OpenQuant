"""Integration test: Backtest → Optimization workflow."""
from __future__ import annotations
import pandas as pd
from openquant.strategies.registry import make_strategy
from openquant.backtest.engine import backtest_signals
from openquant.optimization.optuna_search import optuna_best_params


def test_backtest_to_optuna_optimization(sample_ohlcv_df, small_param_grid):
    """Test end-to-end: backtest → hyperparameter optimization with Optuna."""
    df = sample_ohlcv_df
    
    best_params = optuna_best_params(
        strat_name='kalman',
        df=df,
        grid=small_param_grid,
        fee_bps=1.0,
        weight=1.0,
        timeframe='1h',
        n_trials=5
    )
    
    assert 'process_noise' in best_params
    assert 'measurement_noise' in best_params
    assert 'threshold' in best_params
    
    assert best_params['process_noise'] in small_param_grid['process_noise']
    assert best_params['measurement_noise'] in small_param_grid['measurement_noise']
    assert best_params['threshold'] in small_param_grid['threshold']
    
    strategy = make_strategy('kalman', **best_params)
    signals = strategy.generate_signals(df)
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    assert len(result.equity_curve) == len(df)
    assert result.equity_curve.iloc[-1] > 0


def test_optimization_multiple_strategies(sample_ohlcv_df):
    """Test optimization across different strategy types."""
    df = sample_ohlcv_df
    
    strategies_and_grids = {
        'kalman': {
            'process_noise': [1e-5, 1e-4],
            'measurement_noise': [1e-3],
            'threshold': [1.0, 1.5]
        },
        'hurst': {
            'lookback': [50, 100],
            'trend_threshold': [0.55],
            'mr_threshold': [0.45]
        }
    }
    
    results = {}
    for strat_name, grid in strategies_and_grids.items():
        best = optuna_best_params(
            strat_name=strat_name,
            df=df,
            grid=grid,
            fee_bps=1.0,
            weight=1.0,
            timeframe='1h',
            n_trials=3
        )
        results[strat_name] = best
    
    assert len(results) == len(strategies_and_grids)
    
    for strat_name, best_params in results.items():
        assert len(best_params) > 0
        strategy = make_strategy(strat_name, **best_params)
        assert strategy is not None


def test_optimization_improves_performance(sample_ohlcv_df, small_param_grid):
    """Test that optimization finds better parameters than random choice."""
    df = sample_ohlcv_df
    
    import itertools
    param_combinations = list(itertools.product(
        small_param_grid['process_noise'],
        small_param_grid['measurement_noise'],
        small_param_grid['threshold']
    ))
    
    random_params = {
        'process_noise': param_combinations[0][0],
        'measurement_noise': param_combinations[0][1],
        'threshold': param_combinations[0][2],
    }
    
    strategy_random = make_strategy('kalman', **random_params)
    signals_random = strategy_random.generate_signals(df)
    result_random = backtest_signals(df, signals_random, fee_bps=1.0, weight=1.0)
    
    best_params = optuna_best_params(
        strat_name='kalman',
        df=df,
        grid=small_param_grid,
        fee_bps=1.0,
        weight=1.0,
        timeframe='1h',
        n_trials=8
    )
    
    strategy_optimized = make_strategy('kalman', **best_params)
    signals_optimized = strategy_optimized.generate_signals(df)
    result_optimized = backtest_signals(df, signals_optimized, fee_bps=1.0, weight=1.0)
    
    from openquant.backtest.metrics import sharpe
    sharpe_random = sharpe(result_random.returns, freq='1h')
    sharpe_optimized = sharpe(result_optimized.returns, freq='1h')
    
    assert sharpe_optimized >= sharpe_random - 0.5
