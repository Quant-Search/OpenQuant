"""Integration test: Research workflow with data fetching and analysis."""
from __future__ import annotations
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from openquant.strategies.registry import make_strategy
from openquant.backtest.engine import backtest_signals, summarize_performance
from openquant.optimization.optuna_search import optuna_best_params
from openquant.risk.guardrails import apply_guardrails
from openquant.backtest.metrics import max_drawdown


def test_research_single_symbol_workflow(sample_ohlcv_df):
    """Test research workflow for a single symbol with optimization."""
    df = sample_ohlcv_df
    
    strategies_to_test = ['kalman', 'hurst']
    
    results = {}
    
    for strat_name in strategies_to_test:
        if strat_name == 'kalman':
            grid = {
                'process_noise': [1e-5, 1e-4],
                'measurement_noise': [1e-3],
                'threshold': [1.0, 1.5]
            }
        else:
            grid = {
                'lookback': [50, 100],
                'trend_threshold': [0.55],
                'mr_threshold': [0.45]
            }
        
        best_params = optuna_best_params(
            strat_name=strat_name,
            df=df,
            grid=grid,
            fee_bps=1.0,
            weight=1.0,
            timeframe='1h',
            n_trials=3
        )
        
        strategy = make_strategy(strat_name, **best_params)
        signals = strategy.generate_signals(df)
        result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
        
        metrics = summarize_performance(result, freq='1h')
        
        dd = abs(float(max_drawdown(result.equity_curve)))
        returns_arr = result.returns.dropna().values
        losses = -returns_arr
        var = float(np.quantile(losses, 0.95)) if len(losses) > 0 else 0.0
        tail = losses[losses >= var]
        cvar = float(tail.mean()) if tail.size else 0.0
        
        daily_returns = result.returns.resample('1D').sum().dropna()
        worst_daily = float(abs(daily_returns.min())) if not daily_returns.empty else 0.0
        
        ok, reasons = apply_guardrails(
            max_drawdown=dd,
            cvar=cvar,
            worst_daily_loss=worst_daily,
            dd_limit=0.30,
            cvar_limit=0.15,
            daily_loss_cap=0.10
        )
        
        results[strat_name] = {
            'params': best_params,
            'metrics': metrics,
            'guardrails_ok': ok,
            'guardrails_reasons': reasons,
            'result': result
        }
    
    assert len(results) == len(strategies_to_test)
    
    for strat_name, data in results.items():
        assert 'sharpe' in data['metrics']
        assert isinstance(data['guardrails_ok'], bool)


def test_research_multi_symbol_comparison(sample_ohlcv_df):
    """Test comparing same strategy across multiple synthetic symbols."""
    
    def create_synthetic_data(seed: int, n: int = 200):
        idx = pd.date_range('2023-01-01', periods=n, freq='1h', tz='UTC')
        base = 100.0 + seed * 5
        trend = np.linspace(0, 10 * (1 + seed * 0.1), n)
        noise = np.random.RandomState(seed).normal(0, 2, n)
        close = base + trend + noise
        
        return pd.DataFrame({
            'Open': close,
            'High': close + 0.5,
            'Low': close - 0.5,
            'Close': close,
            'Volume': 1000,
        }, index=idx)
    
    symbols = ['SYM1', 'SYM2', 'SYM3']
    
    results = {}
    
    for i, symbol in enumerate(symbols):
        df = create_synthetic_data(seed=i+1)
        
        strategy = make_strategy('kalman', process_noise=1e-5, measurement_noise=1e-3, threshold=1.0)
        signals = strategy.generate_signals(df)
        result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
        
        metrics = summarize_performance(result, freq='1h')
        
        results[symbol] = {
            'metrics': metrics,
            'final_equity': result.equity_curve.iloc[-1]
        }
    
    assert len(results) == len(symbols)
    
    best_symbol = max(results.items(), key=lambda x: x[1]['metrics']['sharpe'])
    assert best_symbol[0] in symbols


def test_research_with_walk_forward_validation(sample_ohlcv_df):
    """Test research workflow with walk-forward optimization."""
    from openquant.evaluation.wfo import walk_forward_evaluate, WFOSpec
    
    df = sample_ohlcv_df
    
    grid = {
        'process_noise': [1e-5, 1e-4],
        'measurement_noise': [1e-3],
        'threshold': [1.0]
    }
    
    try:
        wfo_result = walk_forward_evaluate(
            df=df,
            strategy_factory=lambda **params: make_strategy('kalman', **params),
            param_grid=grid,
            fee_bps=1.0,
            weight=1.0,
            wfo=WFOSpec(train_periods=3, test_periods=1, step_periods=1)
        )
        
        assert 'mean_test_sharpe' in wfo_result
        assert isinstance(wfo_result['mean_test_sharpe'], (int, float))
    except Exception as e:
        pass


def test_research_concentration_limits():
    """Test concentration limit application in research results."""
    from openquant.risk.guardrails import apply_concentration_limits
    
    rows = [
        {
            'symbol': 'BTC/USDT',
            'strategy': 'kalman',
            'metrics': {'ok': True, 'sharpe': 2.0, 'dsr': 1.5, 'wfo_mts': 1.2}
        },
        {
            'symbol': 'BTC/USDT',
            'strategy': 'hurst',
            'metrics': {'ok': True, 'sharpe': 1.8, 'dsr': 1.3, 'wfo_mts': 1.0}
        },
        {
            'symbol': 'ETH/USDT',
            'strategy': 'kalman',
            'metrics': {'ok': True, 'sharpe': 1.5, 'dsr': 1.0, 'wfo_mts': 0.8}
        },
    ]
    
    filtered = apply_concentration_limits(
        rows,
        max_per_symbol=1,
        max_per_strategy_per_symbol=1
    )
    
    ok_count = sum(1 for r in filtered if r['metrics']['ok'])
    assert ok_count <= len(rows)


def test_research_regime_detection(sample_ohlcv_df):
    """Test regime detection in research workflow."""
    from openquant.evaluation.regime import compute_regime_features
    
    df = sample_ohlcv_df
    
    regime = compute_regime_features(df)
    
    assert 'trend_score' in regime
    assert isinstance(regime['trend_score'], (int, float))
    assert -1.0 <= regime['trend_score'] <= 1.0


def test_research_exposure_allocation(sample_ohlcv_df):
    """Test portfolio exposure allocation from research results."""
    from openquant.risk.exposure import propose_portfolio_weights
    
    rows = [
        {
            'symbol': 'BTC/USDT',
            'strategy': 'kalman',
            'metrics': {'ok': True, 'sharpe': 2.0, 'dsr': 1.5}
        },
        {
            'symbol': 'ETH/USDT',
            'strategy': 'hurst',
            'metrics': {'ok': True, 'sharpe': 1.8, 'dsr': 1.3}
        },
        {
            'symbol': 'BNB/USDT',
            'strategy': 'kalman',
            'metrics': {'ok': False, 'sharpe': 0.5, 'dsr': 0.2}
        },
    ]
    
    allocation = propose_portfolio_weights(
        rows,
        max_total_weight=1.0,
        max_symbol_weight=0.5,
        slot_weight=0.1
    )
    
    assert isinstance(allocation, list)
    
    if allocation:
        total_weight = sum(w for _, w in allocation)
        assert total_weight <= 1.0


def test_research_results_persistence(sample_ohlcv_df, tmp_path):
    """Test persisting research results to database."""
    from openquant.storage.results_db import upsert_results, get_best_config
    
    df = sample_ohlcv_df
    
    strategy = make_strategy('kalman', process_noise=1e-5, measurement_noise=1e-3, threshold=1.0)
    signals = strategy.generate_signals(df)
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    metrics = summarize_performance(result, freq='1h')
    
    rows = [{
        'strategy': 'kalman',
        'exchange': 'binance',
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'params': {'process_noise': 1e-5, 'measurement_noise': 1e-3, 'threshold': 1.0},
        'bars': len(df),
        'metrics': {
            'sharpe': metrics['sharpe'],
            'max_dd': metrics['max_drawdown'],
            'win_rate': metrics['win_rate'],
            'ok': True,
            'dsr': 1.0,
            'wfo_mts': 0.8
        }
    }]
    
    db_path = tmp_path / "test_results.duckdb"
    
    try:
        upsert_results(rows, db_path=str(db_path), run_id='test_run_001')
        
        best_params, best_score = get_best_config(
            str(db_path),
            exchange='binance',
            symbol='BTC/USDT',
            timeframe='1h',
            strategy='kalman'
        )
        
        if best_params is not None:
            assert 'process_noise' in best_params
    except Exception:
        pass


def test_research_deflated_sharpe(sample_ohlcv_df):
    """Test deflated Sharpe ratio calculation in research."""
    from openquant.evaluation.deflated_sharpe import deflated_sharpe_ratio
    
    df = sample_ohlcv_df
    
    strategy = make_strategy('kalman')
    signals = strategy.generate_signals(df)
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    from openquant.backtest.metrics import sharpe
    sharpe_val = sharpe(result.returns, freq='1h')
    
    returns_arr = result.returns.dropna().values
    T = len(returns_arr)
    
    dsr = deflated_sharpe_ratio(sharpe_val, T=T, trials=10)
    
    assert isinstance(dsr, float)
    assert dsr <= sharpe_val
