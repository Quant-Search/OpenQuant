"""Integration test: Data Fetch → Backtest workflow."""
from __future__ import annotations
import pandas as pd
from openquant.data.loader import DataLoader
from openquant.strategies.registry import make_strategy
from openquant.backtest.engine import backtest_signals, summarize_performance


def test_data_fetch_to_backtest(sample_ohlcv_df):
    """Test end-to-end: fetch data → generate signals → backtest."""
    df = sample_ohlcv_df
    
    assert not df.empty
    assert 'Close' in df.columns
    assert len(df) > 50
    
    strategy = make_strategy('kalman', process_noise=1e-5, measurement_noise=1e-3, threshold=1.0)
    
    signals = strategy.generate_signals(df)
    
    assert len(signals) == len(df)
    assert signals.isin([-1, 0, 1]).all()
    
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    assert len(result.equity_curve) == len(df)
    assert result.equity_curve.iloc[0] == 1.0
    assert result.equity_curve.iloc[-1] > 0
    
    metrics = summarize_performance(result, freq='1h')
    
    assert 'sharpe' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics
    assert isinstance(metrics['sharpe'], float)


def test_multiple_strategies_backtest(sample_ohlcv_df):
    """Test backtesting multiple strategies on same data."""
    df = sample_ohlcv_df
    strategies = ['kalman', 'hurst', 'stat_arb']
    
    results = {}
    for strat_name in strategies:
        strategy = make_strategy(strat_name)
        signals = strategy.generate_signals(df)
        result = backtest_signals(df, signals, fee_bps=1.0, weight=0.5)
        results[strat_name] = result
    
    assert len(results) == len(strategies)
    
    for strat_name, result in results.items():
        assert len(result.equity_curve) == len(df)
        assert result.equity_curve.iloc[0] == 1.0


def test_backtest_with_costs(sample_ohlcv_df):
    """Test backtest with realistic transaction costs."""
    df = sample_ohlcv_df
    
    strategy = make_strategy('kalman', process_noise=1e-5, measurement_noise=1e-3, threshold=1.0)
    signals = strategy.generate_signals(df)
    
    no_cost = backtest_signals(df, signals, fee_bps=0.0, slippage_bps=0.0, weight=1.0)
    with_cost = backtest_signals(df, signals, fee_bps=5.0, slippage_bps=2.0, weight=1.0)
    
    assert with_cost.equity_curve.iloc[-1] < no_cost.equity_curve.iloc[-1]
    
    assert no_cost.trades.sum() == with_cost.trades.sum()


def test_backtest_with_stop_loss(sample_ohlcv_df):
    """Test backtest with stop-loss and take-profit."""
    df = sample_ohlcv_df
    
    strategy = make_strategy('kalman')
    signals = strategy.generate_signals(df)
    
    result_no_sl = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    result_with_sl = backtest_signals(
        df, signals, 
        fee_bps=1.0, 
        weight=1.0,
        stop_loss_atr=2.0,
        take_profit_atr=3.0
    )
    
    assert len(result_no_sl.equity_curve) == len(result_with_sl.equity_curve)
    
    assert result_with_sl.trades.sum() >= result_no_sl.trades.sum()
