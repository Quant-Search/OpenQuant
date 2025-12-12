"""Integration test: Optimization → Paper Trading workflow."""
from __future__ import annotations
import pandas as pd
from openquant.strategies.registry import make_strategy
from openquant.optimization.optuna_search import optuna_best_params
from openquant.paper.state import PortfolioState, Key
from openquant.paper.simulator import MarketSnapshot, rebalance_to_targets


def test_optimization_to_paper_trading(sample_ohlcv_df, small_param_grid):
    """Test end-to-end: optimization → deploy to paper trading."""
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
    
    strategy = make_strategy('kalman', **best_params)
    signals = strategy.generate_signals(df)
    
    last_signal = signals.iloc[-1]
    last_price = df['Close'].iloc[-1]
    
    state = PortfolioState(cash=100_000.0)
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    
    target_weight = 0.0
    if last_signal == 1:
        target_weight = 0.2
    elif last_signal == -1:
        target_weight = 0.0
    
    snap = MarketSnapshot(prices={key: float(last_price)})
    
    summary = rebalance_to_targets(state, [(key, target_weight)], snap, fee_bps=1.0)
    
    assert 'orders' in summary
    assert 'turnover' in summary
    
    position = state.position(key)
    if target_weight > 0:
        assert position > 0
    else:
        assert position == 0


def test_multi_symbol_paper_trading(sample_ohlcv_df):
    """Test paper trading with multiple symbols."""
    df = sample_ohlcv_df
    
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    
    state = PortfolioState(cash=100_000.0)
    
    prices = {}
    targets = []
    for i, symbol in enumerate(symbols):
        key = ("BINANCE", symbol, "1h", "kalman")
        price = 100.0 + i * 10
        prices[key] = price
        targets.append((key, 0.1))
    
    snap = MarketSnapshot(prices=prices)
    
    summary = rebalance_to_targets(state, targets, snap, fee_bps=1.0)
    
    assert summary['orders'] > 0
    
    total_position_count = sum(1 for pos in state.holdings.values() if abs(pos) > 1e-9)
    assert total_position_count == len(symbols)
    
    equity = state.cash
    for key, units in state.holdings.items():
        equity += units * prices[key]
    
    assert equity > 0
    assert equity <= 100_000.0


def test_paper_trading_with_stop_loss(sample_ohlcv_df):
    """Test paper trading with stop-loss orders."""
    from openquant.paper.simulator import execute_orders, check_exits
    
    state = PortfolioState(cash=100_000.0)
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    
    entry_price = 100.0
    units = 10.0
    sl_price = 95.0
    tp_price = 110.0
    
    snap = MarketSnapshot(prices={key: entry_price})
    
    orders = [(key, units, entry_price, sl_price, tp_price)]
    summary, fills = execute_orders(state, orders, fee_bps=1.0, snap=snap)
    
    assert state.position(key) == units
    assert state.sl_levels.get(key) == sl_price
    assert state.tp_levels.get(key) == tp_price
    
    snap_sl_hit = MarketSnapshot(prices={key: 94.0})
    exit_orders = check_exits(state, snap_sl_hit)
    
    assert len(exit_orders) == 1
    assert exit_orders[0][0] == key
    assert exit_orders[0][1] == -units
