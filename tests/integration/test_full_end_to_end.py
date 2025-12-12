"""Integration test: Full end-to-end workflow combining all components."""
from __future__ import annotations
import pandas as pd
import numpy as np
from openquant.strategies.registry import make_strategy
from openquant.backtest.engine import backtest_signals, summarize_performance
from openquant.optimization.optuna_search import optuna_best_params
from openquant.paper.state import PortfolioState, Key
from openquant.paper.simulator import MarketSnapshot, rebalance_to_targets, check_exits
from openquant.risk.guardrails import apply_guardrails
from openquant.backtest.metrics import max_drawdown


def test_full_pipeline_single_strategy(sample_ohlcv_df, small_param_grid, clean_risk_state):
    """Test complete pipeline: data → optimize → backtest → paper trade → risk check."""
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
    
    strategy = make_strategy('kalman', **best_params)
    signals = strategy.generate_signals(df)
    
    assert len(signals) == len(df)
    
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    metrics = summarize_performance(result, freq='1h')
    assert 'sharpe' in metrics
    
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
    
    if ok:
        state = PortfolioState(cash=100_000.0)
        key = ("BINANCE", "BTC/USDT", "1h", "kalman")
        
        last_signal = signals.iloc[-1]
        last_price = df['Close'].iloc[-1]
        
        target_weight = 0.2 if last_signal == 1 else 0.0
        
        snap = MarketSnapshot(prices={key: float(last_price)})
        summary = rebalance_to_targets(state, [(key, target_weight)], snap, fee_bps=1.0)
        
        assert 'orders' in summary
        
        position = state.position(key)
        if target_weight > 0:
            assert position > 0
        else:
            assert position == 0


def test_multi_strategy_pipeline(sample_ohlcv_df, clean_risk_state):
    """Test pipeline with multiple strategies running in parallel."""
    df = sample_ohlcv_df
    
    strategies = {
        'kalman': {'process_noise': [1e-5], 'measurement_noise': [1e-3], 'threshold': [1.0]},
        'hurst': {'lookback': [50], 'trend_threshold': [0.55], 'mr_threshold': [0.45]}
    }
    
    state = PortfolioState(cash=100_000.0)
    prices = {}
    targets = []
    
    for i, (strat_name, grid) in enumerate(strategies.items()):
        best_params = optuna_best_params(
            strat_name=strat_name,
            df=df,
            grid=grid,
            fee_bps=1.0,
            weight=1.0,
            timeframe='1h',
            n_trials=2
        )
        
        strategy = make_strategy(strat_name, **best_params)
        signals = strategy.generate_signals(df)
        result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
        
        metrics = summarize_performance(result, freq='1h')
        
        if metrics['sharpe'] > 0:
            key = ("BINANCE", f"ASSET{i}/USDT", "1h", strat_name)
            prices[key] = 100.0 + i * 10
            
            last_signal = signals.iloc[-1]
            if last_signal != 0:
                targets.append((key, 0.1))
    
    if targets:
        snap = MarketSnapshot(prices=prices)
        summary = rebalance_to_targets(state, targets, snap, fee_bps=1.0)
        
        assert summary['orders'] >= 0


def test_live_update_simulation(sample_ohlcv_df, clean_risk_state):
    """Simulate live trading updates with rolling data windows."""
    df = sample_ohlcv_df
    
    lookback = 150
    update_every = 10
    
    state = PortfolioState(cash=100_000.0)
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    
    strategy = make_strategy('kalman', process_noise=1e-5, measurement_noise=1e-3, threshold=1.0)
    
    for i in range(lookback, len(df), update_every):
        window = df.iloc[i-lookback:i]
        
        signals = strategy.generate_signals(window)
        last_signal = signals.iloc[-1]
        last_price = window['Close'].iloc[-1]
        
        target_weight = 0.0
        if last_signal == 1:
            target_weight = 0.2
        elif last_signal == -1:
            target_weight = 0.0
        
        snap = MarketSnapshot(prices={key: float(last_price)})
        
        exit_orders = check_exits(state, snap)
        if exit_orders:
            from openquant.paper.simulator import execute_orders
            execute_orders(state, exit_orders, fee_bps=1.0, snap=snap)
        
        summary = rebalance_to_targets(state, [(key, target_weight)], snap, fee_bps=1.0)
    
    final_equity = state.cash
    for k, units in state.holdings.items():
        price = df['Close'].iloc[-1]
        final_equity += units * price
    
    assert final_equity > 0


def test_portfolio_rebalancing_workflow(sample_ohlcv_df, clean_risk_state):
    """Test portfolio rebalancing with multiple positions."""
    df = sample_ohlcv_df
    
    state = PortfolioState(cash=100_000.0)
    
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    keys = [("BINANCE", sym, "1h", "kalman") for sym in symbols]
    
    prices = {key: 100.0 + i * 10 for i, key in enumerate(keys)}
    
    initial_weights = [(keys[0], 0.3), (keys[1], 0.3), (keys[2], 0.3)]
    snap = MarketSnapshot(prices=prices)
    
    summary1 = rebalance_to_targets(state, initial_weights, snap, fee_bps=1.0)
    assert summary1['orders'] == len(symbols)
    
    new_prices = {key: price * 1.05 for key, price in prices.items()}
    new_snap = MarketSnapshot(prices=new_prices)
    
    new_weights = [(keys[0], 0.5), (keys[1], 0.25), (keys[2], 0.0)]
    summary2 = rebalance_to_targets(state, new_weights, new_snap, fee_bps=1.0)
    
    assert summary2['orders'] > 0
    
    position0 = state.position(keys[0])
    position1 = state.position(keys[1])
    position2 = state.position(keys[2])
    
    assert position0 > 0
    assert position1 > 0
    assert abs(position2) < 1e-9


def test_stress_test_high_volatility(clean_risk_state):
    """Test system behavior under high volatility conditions."""
    from openquant.risk.circuit_breaker import CIRCUIT_BREAKER
    
    n = 100
    idx = pd.date_range('2023-01-01', periods=n, freq='1h', tz='UTC')
    
    volatility = np.random.RandomState(42).normal(0, 10, n)
    close = 100.0 + np.cumsum(volatility)
    close = np.maximum(close, 50.0)
    
    df = pd.DataFrame({
        'Open': close,
        'High': close + 5,
        'Low': close - 5,
        'Close': close,
        'Volume': 1000,
    }, index=idx)
    
    strategy = make_strategy('kalman')
    signals = strategy.generate_signals(df)
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    CIRCUIT_BREAKER.reset()
    
    equity_curve = result.equity_curve.values * 100_000.0
    
    for i, equity in enumerate(equity_curve):
        tripped = CIRCUIT_BREAKER.update(current_equity=equity)
        if tripped:
            break
    
    status = CIRCUIT_BREAKER.get_status()
    assert 'is_tripped' in status
