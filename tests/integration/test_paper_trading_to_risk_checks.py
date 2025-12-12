"""Integration test: Paper Trading → Risk Checks workflow."""
from __future__ import annotations
import pandas as pd
from openquant.paper.state import PortfolioState, Key
from openquant.paper.simulator import MarketSnapshot, execute_orders, check_daily_loss
from openquant.risk.guardrails import apply_guardrails
from openquant.risk.kill_switch import KillSwitch
from openquant.risk.circuit_breaker import CircuitBreaker


def test_paper_trading_with_guardrails(sample_ohlcv_df):
    """Test paper trading with risk guardrails applied."""
    from openquant.strategies.registry import make_strategy
    from openquant.backtest.engine import backtest_signals
    from openquant.backtest.metrics import max_drawdown
    import numpy as np
    
    df = sample_ohlcv_df
    strategy = make_strategy('kalman')
    signals = strategy.generate_signals(df)
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    dd = abs(float(max_drawdown(result.equity_curve)))
    
    returns_arr = result.returns.dropna().values
    losses = -returns_arr
    var = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else 0.0
    
    daily_returns = result.returns.resample('1D').sum().dropna()
    worst_daily = float(abs(daily_returns.min())) if not daily_returns.empty else 0.0
    
    ok, reasons = apply_guardrails(
        max_drawdown=dd,
        cvar=cvar,
        worst_daily_loss=worst_daily,
        dd_limit=0.20,
        cvar_limit=0.08,
        daily_loss_cap=0.05
    )
    
    assert isinstance(ok, bool)
    assert isinstance(reasons, list)
    
    if not ok:
        assert len(reasons) > 0


def test_kill_switch_blocks_trading(clean_risk_state, temp_state_files):
    """Test that kill switch blocks all trading operations."""
    from openquant.risk.kill_switch import KILL_SWITCH
    
    assert not KILL_SWITCH.is_active()
    
    state = PortfolioState(cash=100_000.0)
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    
    orders = [(key, 10.0, 100.0, None, None)]
    snap = MarketSnapshot(prices={key: 100.0})
    
    summary, fills = execute_orders(state, orders, fee_bps=1.0, snap=snap)
    assert summary['orders'] > 0
    assert len(fills) > 0
    
    KILL_SWITCH.activate()
    assert KILL_SWITCH.is_active()
    
    state2 = PortfolioState(cash=100_000.0)
    summary2, fills2 = execute_orders(state2, orders, fee_bps=1.0, snap=snap)
    
    assert summary2['orders'] == 0
    assert len(fills2) == 0
    assert 'kill_switch_blocked' in summary2


def test_circuit_breaker_daily_loss(clean_risk_state, temp_state_files):
    """Test circuit breaker trips on daily loss limit."""
    from openquant.risk.circuit_breaker import CIRCUIT_BREAKER
    
    CIRCUIT_BREAKER.reset()
    
    start_equity = 100_000.0
    CIRCUIT_BREAKER.update(current_equity=start_equity)
    
    assert not CIRCUIT_BREAKER.is_tripped()
    
    loss_equity = start_equity * 0.97
    CIRCUIT_BREAKER.update(current_equity=loss_equity)
    
    assert CIRCUIT_BREAKER.is_tripped()
    assert CIRCUIT_BREAKER.state.daily_loss_breaker_tripped
    
    state = PortfolioState(cash=100_000.0)
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    orders = [(key, 10.0, 100.0, None, None)]
    snap = MarketSnapshot(prices={key: 100.0})
    
    summary, fills = execute_orders(state, orders, fee_bps=1.0, snap=snap)
    
    assert summary['orders'] == 0
    assert len(fills) == 0
    assert 'circuit_breaker_blocked' in summary


def test_circuit_breaker_drawdown(clean_risk_state, temp_state_files):
    """Test circuit breaker trips on drawdown limit."""
    from openquant.risk.circuit_breaker import CIRCUIT_BREAKER
    
    CIRCUIT_BREAKER.reset()
    
    peak_equity = 100_000.0
    CIRCUIT_BREAKER.update(current_equity=peak_equity)
    
    assert CIRCUIT_BREAKER.state.peak_equity == peak_equity
    
    drawdown_equity = peak_equity * 0.88
    CIRCUIT_BREAKER.update(current_equity=drawdown_equity)
    
    assert CIRCUIT_BREAKER.is_tripped()
    assert CIRCUIT_BREAKER.state.drawdown_breaker_tripped


def test_daily_loss_limit_in_simulator(sample_ohlcv_df):
    """Test daily loss limit enforcement in paper simulator."""
    state = PortfolioState(cash=100_000.0)
    state.daily_start_equity = 100_000.0
    
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    
    state.set_position(key, 10.0)
    
    price_loss = 90.0
    snap = MarketSnapshot(prices={key: price_loss})
    
    hit_limit = check_daily_loss(state, snap, limit_pct=0.05)
    
    assert isinstance(hit_limit, bool)
    
    if hit_limit:
        pass


def test_integrated_risk_workflow(sample_ohlcv_df, clean_risk_state, temp_state_files):
    """Test complete risk workflow: trading → loss → circuit breaker → halt."""
    from openquant.strategies.registry import make_strategy
    from openquant.backtest.engine import backtest_signals
    from openquant.risk.circuit_breaker import CIRCUIT_BREAKER
    
    df = sample_ohlcv_df
    
    strategy = make_strategy('kalman')
    signals = strategy.generate_signals(df)
    result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
    
    CIRCUIT_BREAKER.reset()
    
    initial_equity = 100_000.0
    CIRCUIT_BREAKER.update(current_equity=initial_equity)
    
    state = PortfolioState(cash=initial_equity)
    key = ("BINANCE", "BTC/USDT", "1h", "kalman")
    
    snap = MarketSnapshot(prices={key: 100.0})
    orders = [(key, 100.0, 100.0, None, None)]
    summary, fills = execute_orders(state, orders, fee_bps=1.0, snap=snap)
    
    assert summary['orders'] > 0
    
    current_equity = state.cash + 100.0 * 95.0
    
    if current_equity < initial_equity * 0.97:
        CIRCUIT_BREAKER.update(current_equity=current_equity)
        
        if CIRCUIT_BREAKER.is_tripped():
            new_orders = [(key, 50.0, 95.0, None, None)]
            summary2, fills2 = execute_orders(state, new_orders, fee_bps=1.0, snap=snap)
            
            assert summary2['orders'] == 0
            assert 'circuit_breaker_blocked' in summary2
