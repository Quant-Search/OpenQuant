from __future__ import annotations
from openquant.paper.state import PortfolioState
from openquant.paper.simulator import MarketSnapshot, compute_rebalance_orders, execute_orders


def test_execute_orders_cash_and_units():
    key = ("BINANCE","ETH/USDT","1h","ema")
    state = PortfolioState(cash=1000.0)
    snap = MarketSnapshot(prices={key: 100.0})
    # target 10% weight -> equity=1000 -> notional=100 -> units=1 -> delta=+1
    orders = compute_rebalance_orders(state, [(key, 0.10)], snap)
    summary, fills = execute_orders(state, orders, fee_bps=100.0, slippage_bps=0.0)
    # buy 1 at 100, fee 1% -> cash decrease 100 + 1 = 101
    assert abs(state.cash - (1000.0 - 101.0)) < 1e-9
    assert abs(state.position(key) - 1.0) < 1e-9

