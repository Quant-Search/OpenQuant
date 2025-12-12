"""Tests for Kelly Criterion adaptive position sizing."""
from __future__ import annotations
import numpy as np
import pytest
from openquant.risk.kelly_criterion import (
    KellyCriterion,
    compute_rolling_volatility,
    estimate_win_rate_from_signals,
    TradeRecord,
    KellyStats,
)
from openquant.paper.state import PortfolioState
from openquant.paper.simulator import (
    MarketSnapshot,
    compute_target_units_with_kelly,
    rebalance_to_targets_with_kelly,
    record_closed_trades,
)


def test_kelly_criterion_init():
    """Test Kelly Criterion initialization."""
    kelly = KellyCriterion()
    assert kelly.kelly_fraction == 0.5
    assert kelly.min_trades == 20
    assert kelly.volatility_target == 0.20
    assert len(kelly.trade_history) == 0


def test_kelly_criterion_custom_params():
    """Test Kelly Criterion with custom parameters."""
    kelly = KellyCriterion(
        kelly_fraction=0.75,
        min_trades=30,
        volatility_target=0.15,
        max_drawdown_threshold=0.20,
    )
    assert kelly.kelly_fraction == 0.75
    assert kelly.min_trades == 30
    assert kelly.volatility_target == 0.15
    assert kelly.max_drawdown_threshold == 0.20


def test_record_trade():
    """Test recording a trade."""
    kelly = KellyCriterion()
    
    kelly.record_trade(
        pnl=100.0,
        entry_price=50000.0,
        exit_price=51000.0,
        size=0.1,
    )
    
    assert len(kelly.trade_history) == 1
    assert kelly.trade_history[0].pnl == 100.0
    assert kelly._stats_dirty is True


def test_kelly_stats_computation():
    """Test Kelly statistics computation."""
    kelly = KellyCriterion()
    
    # Add 6 winning trades
    for _ in range(6):
        kelly.record_trade(pnl=200.0, entry_price=50000, exit_price=51000, size=0.1)
    
    # Add 4 losing trades
    for _ in range(4):
        kelly.record_trade(pnl=-100.0, entry_price=50000, exit_price=49000, size=0.1)
    
    stats = kelly.get_stats()
    
    assert stats.total_trades == 10
    assert stats.wins == 6
    assert stats.losses == 4
    assert stats.win_rate == 0.6
    assert stats.avg_win == 200.0
    assert stats.avg_loss == 100.0
    assert stats.payoff_ratio == 2.0
    
    # Kelly formula: f* = p - q/b = 0.6 - 0.4/2.0 = 0.4
    # With kelly_fraction=0.5: 0.4 * 0.5 = 0.2
    assert abs(stats.kelly_fraction - 0.2) < 1e-9


def test_kelly_expectancy():
    """Test expectancy calculation."""
    kelly = KellyCriterion()
    
    # Win rate 60%, avg win 200, avg loss 100
    for _ in range(6):
        kelly.record_trade(pnl=200.0, entry_price=50000, exit_price=51000, size=0.1)
    for _ in range(4):
        kelly.record_trade(pnl=-100.0, entry_price=50000, exit_price=49000, size=0.1)
    
    stats = kelly.get_stats()
    expectancy = stats.expectancy()
    
    # Expected: 0.6 * 200 - 0.4 * 100 = 120 - 40 = 80
    assert abs(expectancy - 80.0) < 1e-9


def test_update_equity():
    """Test equity and drawdown tracking."""
    kelly = KellyCriterion()
    
    # Initial equity
    kelly.update_equity(100_000)
    assert kelly.peak_equity == 100_000
    assert kelly.current_equity == 100_000
    assert kelly.current_drawdown == 0.0
    
    # Equity increases - new peak
    kelly.update_equity(110_000)
    assert kelly.peak_equity == 110_000
    assert kelly.current_drawdown == 0.0
    
    # Equity drops - drawdown
    kelly.update_equity(95_000)
    assert kelly.peak_equity == 110_000
    assert kelly.current_drawdown == (110_000 - 95_000) / 110_000
    assert abs(kelly.current_drawdown - 0.136364) < 1e-5


def test_compute_base_kelly_size_insufficient_trades():
    """Test conservative sizing with insufficient trade history."""
    kelly = KellyCriterion(min_trades=20)
    
    # Add only 10 trades (50% of minimum)
    for _ in range(5):
        kelly.record_trade(pnl=100.0, entry_price=50000, exit_price=51000, size=0.1)
    for _ in range(5):
        kelly.record_trade(pnl=-100.0, entry_price=50000, exit_price=49000, size=0.1)
    
    size = kelly.compute_base_kelly_size()
    
    # Should be ramped: 10/20 = 0.5, so 0.1 * 0.5 = 0.05
    assert abs(size - 0.05) < 1e-9


def test_compute_base_kelly_size_sufficient_trades():
    """Test Kelly sizing with sufficient trade history."""
    kelly = KellyCriterion(min_trades=10, kelly_fraction=0.5)
    
    # Add 20 trades with 60% win rate, payoff ratio 2.0
    for _ in range(12):
        kelly.record_trade(pnl=200.0, entry_price=50000, exit_price=51000, size=0.1)
    for _ in range(8):
        kelly.record_trade(pnl=-100.0, entry_price=50000, exit_price=49000, size=0.1)
    
    size = kelly.compute_base_kelly_size()
    
    # Kelly = 0.6 - 0.4/2.0 = 0.4, fractional = 0.4 * 0.5 = 0.2
    assert abs(size - 0.2) < 1e-9


def test_apply_volatility_adjustment():
    """Test volatility-based position sizing adjustment."""
    kelly = KellyCriterion(volatility_target=0.20)
    
    base_size = 0.20
    
    # Low volatility (10%) -> scale up
    adjusted = kelly.apply_volatility_adjustment(base_size, 0.10)
    assert adjusted > base_size  # Should scale up
    assert abs(adjusted - 0.40) < 1e-9  # 0.20 * (0.20/0.10) = 0.40
    
    # High volatility (40%) -> scale down
    adjusted = kelly.apply_volatility_adjustment(base_size, 0.40)
    assert adjusted < base_size  # Should scale down
    assert abs(adjusted - 0.10) < 1e-9  # 0.20 * (0.20/0.40) = 0.10
    
    # Target volatility (20%) -> no change
    adjusted = kelly.apply_volatility_adjustment(base_size, 0.20)
    assert abs(adjusted - base_size) < 1e-9


def test_apply_drawdown_scaling_below_threshold():
    """Test drawdown scaling when below threshold."""
    kelly = KellyCriterion(max_drawdown_threshold=0.15)
    kelly.current_drawdown = 0.10  # Below threshold
    
    size = 0.20
    scaled = kelly.apply_drawdown_scaling(size)
    
    # Should not reduce
    assert scaled == size


def test_apply_drawdown_scaling_above_threshold():
    """Test drawdown scaling when above threshold."""
    kelly = KellyCriterion(
        max_drawdown_threshold=0.15,
        drawdown_scale_factor=2.0,
    )
    kelly.current_drawdown = 0.20  # Above threshold
    
    size = 0.20
    scaled = kelly.apply_drawdown_scaling(size)
    
    # Excess DD = 0.20 - 0.15 = 0.05
    # Reduction = (0.05 * 2.0)^2 = 0.01
    # Scale = 1 - 0.01 = 0.99
    # Scaled size = 0.20 * 0.99 = 0.198
    assert scaled < size
    assert abs(scaled - 0.198) < 1e-9


def test_compute_position_size_full_pipeline():
    """Test full position size computation pipeline."""
    kelly = KellyCriterion(
        kelly_fraction=0.5,
        min_trades=10,
        volatility_target=0.20,
        max_drawdown_threshold=0.15,
    )
    
    # Add trade history
    for _ in range(12):
        kelly.record_trade(pnl=200.0, entry_price=50000, exit_price=51000, size=0.1)
    for _ in range(8):
        kelly.record_trade(pnl=-100.0, entry_price=50000, exit_price=49000, size=0.1)
    
    # Set drawdown
    kelly.update_equity(110_000)
    kelly.update_equity(95_000)  # ~13.6% drawdown
    
    # Compute with volatility
    size = kelly.compute_position_size(volatility=0.30)
    
    # Base Kelly: 0.2
    # Vol adjustment: 0.2 * (0.20/0.30) = 0.133
    # DD scaling: minimal since DD < threshold
    # Result should be ~0.133
    assert 0.12 < size < 0.14


def test_compute_rolling_volatility():
    """Test rolling volatility calculation."""
    # Generate price series with known volatility
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    
    vol = compute_rolling_volatility(prices, window=20, annualization_factor=252)
    
    # Should be roughly 0.01 * sqrt(252) ~ 0.159
    assert 0.10 < vol < 0.25  # Reasonable range


def test_estimate_win_rate_from_signals():
    """Test win rate estimation from signals."""
    signals = np.array([1, 1, -1, 1, -1, 0, 1, -1, 1, 1])
    returns = np.array([0.01, -0.005, 0.008, 0.02, -0.01, 0.0, -0.015, 0.012, 0.018, 0.005])
    
    win_rate, avg_win, avg_loss = estimate_win_rate_from_signals(signals, returns)
    
    # Calculate expected values
    signal_returns = [0.01, -0.005, -0.008, 0.02, 0.01, -0.015, -0.012, 0.018, 0.005]
    wins = [r for r in signal_returns if r > 0]
    losses = [r for r in signal_returns if r < 0]
    
    expected_win_rate = len(wins) / len(signal_returns)
    
    assert abs(win_rate - expected_win_rate) < 1e-9
    assert avg_win > 0
    assert avg_loss > 0


def test_compute_target_units_with_kelly():
    """Test computing target units with Kelly sizing."""
    state = PortfolioState(cash=100_000.0)
    key = ("paper", "BTC/USD", "1h", "test")
    
    # Create price history
    prices = np.linspace(50000, 51000, 50)
    snap = MarketSnapshot(
        prices={key: 51000.0},
        price_history={key: prices},
    )
    
    # Initialize Kelly sizer with trade history
    kelly = KellyCriterion(min_trades=10)
    for _ in range(12):
        kelly.record_trade(pnl=200.0, entry_price=50000, exit_price=51000, size=0.1)
    for _ in range(8):
        kelly.record_trade(pnl=-100.0, entry_price=50000, exit_price=49000, size=0.1)
    
    kelly_sizers = {key: kelly}
    
    # Compute target units with signal weight 1.0
    targets = [(key, 1.0)]
    units = compute_target_units_with_kelly(state, targets, snap, kelly_sizers)
    
    # Should have units for the key
    assert key in units
    assert units[key] > 0
    
    # Units should be less than full equity (due to Kelly fraction)
    full_units = 100_000 / 51000  # ~1.96
    assert units[key] < full_units


def test_rebalance_with_kelly():
    """Test rebalancing with Kelly sizing."""
    state = PortfolioState(cash=100_000.0)
    key = ("paper", "BTC/USD", "1h", "test")
    
    # Create price history
    prices = np.linspace(50000, 51000, 50)
    snap = MarketSnapshot(
        prices={key: 51000.0},
        price_history={key: prices},
    )
    
    # Initialize Kelly sizer
    kelly = KellyCriterion(min_trades=10)
    for _ in range(15):
        kelly.record_trade(pnl=200.0, entry_price=50000, exit_price=51000, size=0.1)
    for _ in range(5):
        kelly.record_trade(pnl=-100.0, entry_price=50000, exit_price=49000, size=0.1)
    
    kelly_sizers = {key: kelly}
    
    # Rebalance with Kelly
    targets = [(key, 1.0)]
    summary, kelly_stats = rebalance_to_targets_with_kelly(
        state, targets, snap, kelly_sizers
    )
    
    # Should have executed orders
    assert summary["orders"] > 0
    
    # Should have position
    assert state.position(key) > 0
    
    # Should have Kelly stats
    assert key in kelly_stats
    assert "kelly_fraction" in kelly_stats[key]
    assert "win_rate" in kelly_stats[key]


def test_record_closed_trades():
    """Test recording closed trades for Kelly statistics."""
    state = PortfolioState(cash=100_000.0)
    key = ("paper", "BTC/USD", "1h", "test")
    
    # Set up initial position
    state.set_position(key, 1.0)
    state.avg_price[key] = 50000.0
    
    # Create fills for closing position
    fills = [
        (key, -1.0, 51000.0, 10.0),  # Close 1 unit at 51000, fee 10
    ]
    
    kelly_sizers = {key: KellyCriterion()}
    
    # Record trades
    record_closed_trades(state, fills, kelly_sizers)
    
    # Should have recorded one trade
    kelly = kelly_sizers[key]
    assert len(kelly.trade_history) == 1
    
    # PnL should be 1.0 * (51000 - 50000) - 10 = 990
    assert abs(kelly.trade_history[0].pnl - 990.0) < 1e-9


def test_kelly_criterion_zero_trades():
    """Test Kelly with no trade history."""
    kelly = KellyCriterion()
    stats = kelly.get_stats()
    
    assert stats.total_trades == 0
    assert stats.win_rate == 0.5  # Default
    assert stats.kelly_fraction == 0.0


def test_kelly_criterion_negative_kelly():
    """Test Kelly with negative edge (losing strategy)."""
    kelly = KellyCriterion(min_trades=10)
    
    # 40% win rate, equal payoff
    for _ in range(4):
        kelly.record_trade(pnl=100.0, entry_price=50000, exit_price=51000, size=0.1)
    for _ in range(6):
        kelly.record_trade(pnl=-100.0, entry_price=50000, exit_price=49000, size=0.1)
    
    size = kelly.compute_base_kelly_size()
    
    # Kelly formula would give negative, should be clamped to 0
    assert size >= 0.0


def test_kelly_summary():
    """Test getting Kelly summary."""
    kelly = KellyCriterion()
    
    for _ in range(6):
        kelly.record_trade(pnl=200.0, entry_price=50000, exit_price=51000, size=0.1)
    for _ in range(4):
        kelly.record_trade(pnl=-100.0, entry_price=50000, exit_price=49000, size=0.1)
    
    kelly.update_equity(100_000)
    
    summary = kelly.get_summary()
    
    assert "kelly_fraction" in summary
    assert "win_rate" in summary
    assert "payoff_ratio" in summary
    assert "expectancy" in summary
    assert "total_trades" in summary
    assert "current_drawdown" in summary
    assert summary["total_trades"] == 10
