from __future__ import annotations
"""Simple paper order simulator using target weights from allocation JSON.

For now, we operate in "notional weights" per key and pretend we can
rebalance immediately at a given price snapshot (no slippage model yet).

Later we can add: next-bar-open fills, slippage, fees, partial fills.
"""
from typing import Dict, List, Tuple, Iterable
from dataclasses import dataclass

from .state import PortfolioState, Key


@dataclass
class MarketSnapshot:
    prices: Dict[Key, float]  # last trade price per key (quote currency units)


def compute_target_units(state: PortfolioState, targets: List[Tuple[Key, float]], snap: MarketSnapshot) -> Dict[Key, float]:
    """Convert target weights (fraction of equity) into target units per key.

    - state.cash + sum(holdings * price) defines equity.
    - For each key with target weight w, target notional = w * equity.
    - Units = notional / price.
    """
    # compute equity
    equity = state.cash
    for k, u in state.holdings.items():
        p = float(snap.prices.get(k, 0.0))
        equity += float(u) * p
    # compute target units per key
    out: Dict[Key, float] = {}
    for (k, w) in targets:
        price = float(snap.prices.get(k, 0.0))
        if price <= 0.0 or w <= 0.0:
            out[k] = 0.0 if w <= 0.0 else 0.0
            continue
        notional = float(w) * equity
        out[k] = notional / price
    return out


def compute_rebalance_orders(state: PortfolioState, targets: List[Tuple[Key, float]], snap: MarketSnapshot) -> List[Tuple[Key, float, float]]:
    """Return list of (key, delta_units, price) required to reach targets."""
    target_units = compute_target_units(state, targets, snap)
    orders: List[Tuple[Key, float, float]] = []
    for k, tgt_u in target_units.items():
        cur_u = state.position(k)
        if abs(tgt_u - cur_u) <= 1e-9:
            continue
        price = float(snap.prices.get(k, 0.0))
        orders.append((k, tgt_u - cur_u, price))
    return orders


def execute_orders(
    state: PortfolioState,
    orders: Iterable[Tuple[Key, float, float]],
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> Tuple[Dict[str, float], List[Tuple[Key, float, float, float]]]:
    """Execute orders and update state.cash and holdings.

    Returns (summary, fills) where fills are (key, delta_units, exec_price, fee_paid).
    """
    orders_count = 0
    turnover = 0.0
    fills: List[Tuple[Key, float, float, float]] = []
    # fee/slippage multipliers
    for k, delta_u, ref_price in orders:
        if abs(delta_u) <= 1e-12:
            continue
        side = 1.0 if delta_u > 0 else -1.0
        # execution price includes slippage in direction of trade
        exec_price = ref_price * (1.0 + side * (slippage_bps / 1e4))
        notional = abs(delta_u) * exec_price
        fee_paid = notional * (fee_bps / 1e4)
        # cash update: buy consumes cash; sell releases cash; fees always reduce cash
        state.cash += (-notional if side > 0 else notional)
        state.cash -= fee_paid
        # update position
        cur_u = state.position(k)
        state.set_position(k, cur_u + delta_u)
        orders_count += 1
        turnover += notional
        fills.append((k, delta_u, exec_price, fee_paid))
    return {"orders": float(orders_count), "turnover": float(turnover)}, fills


def rebalance_to_targets(state: PortfolioState, targets: List[Tuple[Key, float]], snap: MarketSnapshot, *, fee_bps: float = 0.0, slippage_bps: float = 0.0) -> Dict[str, float]:
    """Rebalance holdings to target weights at snapshot prices (no slippage/fees).

    Returns summary dict with 'orders' (count) and 'turnover' (notional traded).
    """
    # derive orders and execute them using optional fee/slippage
    orders = compute_rebalance_orders(state, targets, snap)
    summary, _fills = execute_orders(state, orders, fee_bps=fee_bps, slippage_bps=slippage_bps)
    return summary
