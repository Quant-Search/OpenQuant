from __future__ import annotations
from openquant.paper.state import PortfolioState
from openquant.paper.simulator import MarketSnapshot, compute_target_units, rebalance_to_targets


def test_compute_target_units_basic():
    ps = PortfolioState(cash=100_000.0)
    key = ("BINANCE","BTC/USDT","1h","ema")
    snap = MarketSnapshot(prices={key: 100.0})
    units = compute_target_units(ps, [(key, 0.10)], snap)
    assert abs(units[key] - 100.0) < 1e-9  # 10% of 100k = 10k notional; 10k/100 = 100 units


def test_rebalance_to_targets_changes_holdings():
    ps = PortfolioState(cash=100_000.0)
    key = ("BINANCE","ETH/USDT","1h","ema")
    snap = MarketSnapshot(prices={key: 200.0})
    summary = rebalance_to_targets(ps, [(key, 0.05)], snap)
    assert summary["orders"] == 1.0
    # new units ~ (0.05*100000)/200 = 25
    assert abs(ps.position(key) - 25.0) < 1e-9

