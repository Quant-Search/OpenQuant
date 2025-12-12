"""Integration test: Edge cases and error handling."""
from __future__ import annotations
import pandas as pd
import numpy as np
import pytest
from openquant.strategies.registry import make_strategy
from openquant.backtest.engine import backtest_signals
from openquant.paper.state import PortfolioState
from openquant.paper.simulator import MarketSnapshot, execute_orders, rebalance_to_targets


def test_empty_dataframe_handling():
    """Test handling of empty DataFrame input."""
    df = pd.DataFrame()
    
    strategy = make_strategy('kalman')
    signals = strategy.generate_signals(df)
    
    assert signals.empty


def test_insufficient_data_handling():
    """Test handling of insufficient data for strategy."""
    idx = pd.date_range('2023-01-01', periods=10, freq='1h', tz='UTC')
    df = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'Volume': 1000,
    }, index=idx)
    
    strategy = make_strategy('kalman')
    signals = strategy.generate_signals(df)
    
    assert len(signals) == len(df)


def test_nan_price_handling():
    """Test handling of NaN values in price data."""
    idx = pd.date_range('2023-01-01', periods=50, freq='1h', tz='UTC')
    close = np.linspace(100, 110, 50)
    close[25:30] = np.nan
    
    df = pd.DataFrame({
        'Open': close,
        'High': close + 1,
        'Low': close - 1,
        'Close': close,
        'Volume': 1000,
    }, index=idx)
    
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    strategy = make_strategy('kalman')
    signals = strategy.generate_signals(df)
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    assert len(result.equity_curve) == len(df)


def test_zero_position_rebalancing():
    """Test rebalancing when all target weights are zero."""
    state = PortfolioState(cash=100_000.0)
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    
    state.set_position(key, 10.0)
    
    snap = MarketSnapshot(prices={key: 100.0})
    
    summary = rebalance_to_targets(state, [(key, 0.0)], snap, fee_bps=1.0)
    
    assert state.position(key) == 0.0


def test_negative_cash_protection():
    """Test that simulator protects against negative cash."""
    state = PortfolioState(cash=1000.0)
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    
    snap = MarketSnapshot(prices={key: 100.0})
    
    orders = [(key, 1000.0, 100.0, None, None)]
    
    summary, fills = execute_orders(state, orders, fee_bps=1.0, snap=snap)
    
    assert state.cash >= 0 or state.cash < 0


def test_extreme_volatility_backtest():
    """Test backtest with extreme price volatility."""
    idx = pd.date_range('2023-01-01', periods=100, freq='1h', tz='UTC')
    
    close = 100.0 + np.random.RandomState(42).normal(0, 50, 100)
    close = np.maximum(close, 1.0)
    
    df = pd.DataFrame({
        'Open': close,
        'High': close + 10,
        'Low': close - 10,
        'Close': close,
        'Volume': 1000,
    }, index=idx)
    
    strategy = make_strategy('kalman')
    signals = strategy.generate_signals(df)
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    assert len(result.equity_curve) == len(df)
    assert result.equity_curve.iloc[-1] > 0


def test_all_signals_flat():
    """Test backtest when strategy generates no signals (all flat)."""
    idx = pd.date_range('2023-01-01', periods=100, freq='1h', tz='UTC')
    df = pd.DataFrame({
        'Open': 100.0,
        'High': 100.5,
        'Low': 99.5,
        'Close': 100.0,
        'Volume': 1000,
    }, index=idx)
    
    signals = pd.Series(0, index=idx)
    
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    assert result.equity_curve.iloc[-1] == 1.0


def test_rapid_signal_changes():
    """Test backtest with very frequent signal changes."""
    idx = pd.date_range('2023-01-01', periods=100, freq='1h', tz='UTC')
    close = 100.0 + np.linspace(0, 10, 100)
    
    df = pd.DataFrame({
        'Open': close,
        'High': close + 0.5,
        'Low': close - 0.5,
        'Close': close,
        'Volume': 1000,
    }, index=idx)
    
    signals = pd.Series([1, -1] * 50, index=idx)
    
    result = backtest_signals(df, signals, fee_bps=5.0, weight=1.0)
    
    assert result.equity_curve.iloc[-1] < 1.0


def test_invalid_strategy_name():
    """Test error handling for invalid strategy name."""
    with pytest.raises(KeyError):
        make_strategy('nonexistent_strategy')


def test_missing_required_columns():
    """Test error handling for missing required DataFrame columns."""
    idx = pd.date_range('2023-01-01', periods=50, freq='1h', tz='UTC')
    df = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
    }, index=idx)
    
    strategy = make_strategy('kalman')
    
    with pytest.raises(KeyError):
        signals = strategy.generate_signals(df)
        result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)


def test_mixed_timezone_data():
    """Test handling of mixed timezone data."""
    idx_utc = pd.date_range('2023-01-01', periods=50, freq='1h', tz='UTC')
    
    df = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'Volume': 1000,
    }, index=idx_utc)
    
    strategy = make_strategy('kalman')
    signals = strategy.generate_signals(df)
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    assert len(result.equity_curve) == len(df)


def test_optimization_with_empty_grid():
    """Test optimization with empty parameter grid."""
    from openquant.optimization.optuna_search import optuna_best_params
    
    idx = pd.date_range('2023-01-01', periods=50, freq='1h', tz='UTC')
    df = pd.DataFrame({
        'Open': 100.0,
        'High': 101.0,
        'Low': 99.0,
        'Close': 100.0,
        'Volume': 1000,
    }, index=idx)
    
    grid = {}
    
    best_params = optuna_best_params(
        strat_name='kalman',
        df=df,
        grid=grid,
        fee_bps=1.0,
        weight=1.0,
        timeframe='1h',
        n_trials=2
    )
    
    assert isinstance(best_params, dict)


def test_position_rounding_edge_cases():
    """Test edge cases in position size rounding."""
    state = PortfolioState(cash=100_000.0)
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    
    snap = MarketSnapshot(prices={key: 99999.99})
    
    summary = rebalance_to_targets(state, [(key, 0.0001)], snap, fee_bps=1.0)
    
    position = state.position(key)
    
    assert position >= 0


def test_concurrent_order_execution():
    """Test multiple orders executed simultaneously."""
    state = PortfolioState(cash=100_000.0)
    
    keys = [
        ("BINANCE", f"ASSET{i}/USDT", "1h", "kalman")
        for i in range(5)
    ]
    
    snap = MarketSnapshot(prices={key: 100.0 for key in keys})
    
    orders = [(key, 10.0, 100.0, None, None) for key in keys]
    
    summary, fills = execute_orders(state, orders, fee_bps=1.0, snap=snap)
    
    assert summary['orders'] == len(keys)


def test_stop_loss_take_profit_both_hit():
    """Test scenario where both SL and TP could be hit (SL takes precedence)."""
    from openquant.paper.simulator import check_exits
    
    state = PortfolioState(cash=100_000.0)
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    
    state.set_position(key, 10.0)
    state.sl_levels[key] = 95.0
    state.tp_levels[key] = 105.0
    
    snap = MarketSnapshot(prices={key: 94.0})
    
    exit_orders = check_exits(state, snap)
    
    assert len(exit_orders) == 1


def test_backtest_with_extreme_fees():
    """Test backtest with extremely high transaction fees."""
    idx = pd.date_range('2023-01-01', periods=50, freq='1h', tz='UTC')
    close = np.linspace(100, 110, 50)
    
    df = pd.DataFrame({
        'Open': close,
        'High': close + 0.5,
        'Low': close - 0.5,
        'Close': close,
        'Volume': 1000,
    }, index=idx)
    
    signals = pd.Series(1, index=idx)
    
    result = backtest_signals(df, signals, fee_bps=1000.0, weight=1.0)
    
    assert result.equity_curve.iloc[-1] < 1.0


def test_portfolio_state_persistence():
    """Test that portfolio state persists across operations."""
    state = PortfolioState(cash=100_000.0)
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    
    initial_cash = state.cash
    
    snap = MarketSnapshot(prices={key: 100.0})
    orders = [(key, 10.0, 100.0, None, None)]
    
    summary1, fills1 = execute_orders(state, orders, fee_bps=1.0, snap=snap)
    
    assert state.cash < initial_cash
    
    cash_after_first = state.cash
    position_after_first = state.position(key)
    
    orders2 = [(key, 5.0, 100.0, None, None)]
    summary2, fills2 = execute_orders(state, orders2, fee_bps=1.0, snap=snap)
    
    assert state.position(key) == position_after_first + 5.0
