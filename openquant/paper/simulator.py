from __future__ import annotations
"""Simple paper order simulator using target weights from allocation JSON.

For now, we operate in "notional weights" per key and pretend we can
rebalance immediately at a given price snapshot (no slippage model yet).

Later we can add: next-bar-open fills, slippage, fees, partial fills.
"""
from typing import Dict, List, Tuple, Iterable, Optional
from dataclasses import dataclass, field
import numpy as np

from .state import PortfolioState, Key
from ..risk.kill_switch import KILL_SWITCH
from ..risk.circuit_breaker import CIRCUIT_BREAKER
from ..risk.kelly_criterion import KellyCriterion, compute_rolling_volatility
from ..risk.trade_validator import TRADE_VALIDATOR


@dataclass
class MarketSnapshot:
    prices: Dict[Key, float]  # last trade price per key (quote currency units)
    next_prices: Dict[Key, float] = field(default_factory=dict)  # next-bar prices for fills (optional)
    price_history: Dict[Key, np.ndarray] = field(default_factory=dict)  # historical prices for volatility calculation


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


def compute_target_units_with_kelly(
    state: PortfolioState,
    targets: List[Tuple[Key, float]],
    snap: MarketSnapshot,
    kelly_sizers: Dict[Key, KellyCriterion],
    volatility_window: int = 20,
    annualization_factor: float = 252.0,
) -> Dict[Key, float]:
    """Convert target weights into units with Kelly Criterion position sizing.
    
    Applies adaptive position sizing based on:
    - Kelly Criterion (win rate and payoff ratio)
    - Volatility adjustment (inverse volatility weighting)
    - Drawdown scaling (reduce size during drawdowns)
    
    Args:
        state: Current portfolio state
        targets: List of (key, signal_weight) where signal_weight is the raw strategy signal
        snap: Market snapshot with prices and price history
        kelly_sizers: Dictionary of Kelly sizers per key
        volatility_window: Lookback window for volatility calculation
        annualization_factor: Factor to annualize volatility (252 for daily bars)
        
    Returns:
        Dictionary of target units per key
    """
    # Compute equity
    equity = state.cash
    for k, u in state.holdings.items():
        p = float(snap.prices.get(k, 0.0))
        equity += float(u) * p
        
    # Update Kelly sizers with current equity
    for kelly in kelly_sizers.values():
        kelly.update_equity(equity)
        
    # Compute target units per key with Kelly sizing
    out: Dict[Key, float] = {}
    for (k, signal_weight) in targets:
        price = float(snap.prices.get(k, 0.0))
        
        # Skip if no price or zero signal
        if price <= 0.0 or abs(signal_weight) < 1e-9:
            out[k] = 0.0
            continue
            
        # Get Kelly sizer for this key (create if doesn't exist)
        if k not in kelly_sizers:
            kelly_sizers[k] = KellyCriterion()
            kelly_sizers[k].update_equity(equity)
            
        kelly = kelly_sizers[k]
        
        # Compute volatility if price history available
        volatility = None
        if k in snap.price_history and len(snap.price_history[k]) >= 2:
            volatility = compute_rolling_volatility(
                snap.price_history[k],
                window=volatility_window,
                annualization_factor=annualization_factor,
            )
            
        # Get Kelly-adjusted position size
        kelly_fraction = kelly.compute_position_size(volatility=volatility)
        
        # Apply signal direction and Kelly sizing
        # signal_weight is typically -1, 0, or 1, but could be continuous
        # Kelly fraction is 0 to max_position_size (e.g., 0 to 1.0)
        adjusted_weight = signal_weight * kelly_fraction
        
        # Compute notional and units
        notional = float(adjusted_weight) * equity
        out[k] = notional / price
        
    return out


def compute_rebalance_orders(state: PortfolioState, targets: List[Tuple[Key, float]], snap: MarketSnapshot) -> List[Tuple[Key, float, float, Optional[float], Optional[float]]]:
    """Return list of (key, delta_units, price, sl, tp) required to reach targets."""
    target_units = compute_target_units(state, targets, snap)
    orders: List[Tuple[Key, float, float, Optional[float], Optional[float]]] = []
    for k, tgt_u in target_units.items():
        cur_u = state.position(k)
        if abs(tgt_u - cur_u) <= 1e-9:
            continue
        price = float(snap.prices.get(k, 0.0))
        # SL/TP are not determined here (unless we pass them in targets? yes we should)
        # But targets is List[Tuple[Key, float]] (weight).
        # We need to change targets to include SL/TP or pass a separate map.
        # For now, return None, None. The caller (paper_apply_allocation) should inject them if available.
        orders.append((k, tgt_u - cur_u, price, None, None))
    return orders


def compute_rebalance_orders_with_kelly(
    state: PortfolioState,
    targets: List[Tuple[Key, float]],
    snap: MarketSnapshot,
    kelly_sizers: Dict[Key, KellyCriterion],
    volatility_window: int = 20,
    annualization_factor: float = 252.0,
) -> List[Tuple[Key, float, float, Optional[float], Optional[float]]]:
    """Return list of orders with Kelly position sizing applied.
    
    Args:
        state: Current portfolio state
        targets: List of (key, signal_weight) tuples
        snap: Market snapshot
        kelly_sizers: Dictionary of Kelly sizers per key
        volatility_window: Lookback window for volatility
        annualization_factor: Annualization factor for volatility
        
    Returns:
        List of (key, delta_units, price, sl, tp) orders
    """
    target_units = compute_target_units_with_kelly(
        state, targets, snap, kelly_sizers, volatility_window, annualization_factor
    )
    orders: List[Tuple[Key, float, float, Optional[float], Optional[float]]] = []
    for k, tgt_u in target_units.items():
        cur_u = state.position(k)
        if abs(tgt_u - cur_u) <= 1e-9:
            continue
        price = float(snap.prices.get(k, 0.0))
        orders.append((k, tgt_u - cur_u, price, None, None))
    return orders


def record_closed_trades(
    state: PortfolioState,
    fills: List[Tuple[Key, float, float, float]],
    kelly_sizers: Dict[Key, KellyCriterion],
) -> None:
    """Record closed trades to Kelly sizers for statistics tracking.
    
    When a position is reduced or closed, record the trade outcome.
    
    Args:
        state: Current portfolio state
        fills: List of (key, delta_units, exec_price, fee_paid) from execute_orders
        kelly_sizers: Dictionary of Kelly sizers to update
    """
    for k, delta_u, exec_price, fee_paid in fills:
        # Only record when closing a position (delta_u has opposite sign to current position)
        # or when flipping position
        cur_u = state.position(k)
        prev_u = cur_u - delta_u  # Position before this fill
        
        # Check if we closed or reduced a position
        is_closing = (prev_u > 0 and delta_u < 0) or (prev_u < 0 and delta_u > 0)
        
        if not is_closing:
            continue
            
        # Get entry price
        entry_price = state.avg_price.get(k)
        if entry_price is None or entry_price <= 0:
            continue
            
        # Calculate PnL for the closed portion
        closed_units = min(abs(prev_u), abs(delta_u))
        
        if prev_u > 0:
            # Closing long: profit if exit > entry
            pnl = closed_units * (exec_price - entry_price) - fee_paid
        else:
            # Closing short: profit if exit < entry
            pnl = closed_units * (entry_price - exec_price) - fee_paid
            
        # Record to Kelly sizer
        if k not in kelly_sizers:
            kelly_sizers[k] = KellyCriterion()
            
        kelly_sizers[k].record_trade(
            pnl=pnl,
            entry_price=entry_price,
            exit_price=exec_price,
            size=closed_units,
        )


def execute_orders(
    state: PortfolioState,
    orders: Iterable[Tuple[Key, float, float, Optional[float], Optional[float]]], # Updated signature: (key, delta, price, sl, tp)
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    next_bar_fill: bool = False,
    max_fill_fraction: float = 1.0,
    snap: Optional[MarketSnapshot] = None,
) -> Tuple[Dict[str, float], List[Tuple[Key, float, float, float]]]:
    """Execute orders and update state.cash and holdings.

    Supports next-bar fills (use next_prices if available) and partial fills (max_fill_fraction < 1.0).

    Orders tuple now expects: (key, delta_units, ref_price, sl, tp)
    If sl/tp are None, they are ignored (or kept if existing).

    Returns (summary, fills) where fills are (key, delta_units, exec_price, fee_paid).

    SAFETY: Checks kill switch and circuit breaker before executing.
    If either is active, returns empty fills.
    """
    # KILL SWITCH CHECK - Critical safety mechanism
    # If kill switch is active, refuse to execute any orders
    if KILL_SWITCH.is_active():
        return {"orders": 0, "turnover": 0.0, "kill_switch_blocked": True}, []

    # CIRCUIT BREAKER CHECK - Automatic risk-based halt
    # If circuit breaker is tripped, refuse to execute any orders
    if CIRCUIT_BREAKER.is_tripped():
        return {"orders": 0, "turnover": 0.0, "circuit_breaker_blocked": True}, []

    orders_count = 0
    turnover = 0.0
    blocked_count = 0
    fills: List[Tuple[Key, float, float, float]] = []
    
    # Calculate current equity and positions for validation
    equity = state.cash
    for k, u in state.holdings.items():
        p = float(snap.prices.get(k, 0.0)) if snap else 0.0
        equity += float(u) * p
    
    current_positions = {}
    for k, u in state.holdings.items():
        p = float(snap.prices.get(k, 0.0)) if snap else 0.0
        current_positions[str(k)] = abs(float(u)) * p

    # Normalize input: if orders is list of 3-tuples, pad with None
    # This is a bit hacky to maintain backward compat if caller passes 3-tuples,
    # but better to enforce 5-tuples or check len.
    # Let's assume caller adapts or we check.

    for item in orders:
        if len(item) == 3:
            k, delta_u, ref_price = item # type: ignore
            sl, tp = None, None
        else:
            k, delta_u, ref_price, sl, tp = item # type: ignore

        if abs(delta_u) <= 1e-12:
            continue
        
        side = 1.0 if delta_u > 0 else -1.0
        
        # COMPREHENSIVE RISK VALIDATION
        validation_price = ref_price
        if next_bar_fill and snap and k in snap.next_prices:
            validation_price = snap.next_prices[k]
        
        result = TRADE_VALIDATOR.validate_trade(
            symbol=str(k),
            quantity=abs(delta_u),
            price=validation_price,
            side="buy" if side > 0 else "sell",
            portfolio_value=equity,
            current_positions=current_positions,
            current_equity=equity,
        )
        
        if not result.allowed:
            blocked_count += 1
            from openquant.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Paper trade blocked for {k}: {result.reason}")
            continue
        
        # Use next-bar price if enabled and available
        exec_price = ref_price
        if next_bar_fill and snap and k in snap.next_prices:
            exec_price = snap.next_prices[k]
            
        # execution price includes slippage in direction of trade
        exec_price = exec_price * (1.0 + side * (slippage_bps / 1e4))
        
        # Partial fill: fill only up to max_fill_fraction of the order
        fill_fraction = min(max_fill_fraction, 1.0)
        filled_delta_u = delta_u * fill_fraction
        
        if abs(filled_delta_u) <= 1e-12:
            continue
            
        notional = abs(filled_delta_u) * exec_price
        fee_paid = notional * (fee_bps / 1e4)
        
        # cash update: buy consumes cash; sell releases cash; fees always reduce cash
        state.cash += (-notional if side > 0 else notional)
        state.cash -= fee_paid
        
        # update position and average price
        cur_u = state.position(k)
        new_u = cur_u + filled_delta_u
        
        # Update Avg Price if increasing position
        # If changing side (flip), avg price resets to exec price
        # If reducing, avg price stays same
        
        is_opening = (cur_u == 0) or (cur_u > 0 and filled_delta_u > 0) or (cur_u < 0 and filled_delta_u < 0)
        is_flip = (cur_u > 0 and new_u < 0) or (cur_u < 0 and new_u > 0)
        
        if is_flip:
            # Treated as close all + open new
            state.avg_price[k] = exec_price
        elif is_opening:
            # Weighted average
            # old_val = cur_u * old_price
            # new_part = delta * exec_price
            # total_val = old_val + new_part
            # new_avg = total_val / new_u
            old_avg = state.avg_price.get(k, exec_price)
            total_val = abs(cur_u) * old_avg + abs(filled_delta_u) * exec_price
            state.avg_price[k] = total_val / abs(new_u)
        
        # Update SL/TP if provided
        if sl is not None:
            state.sl_levels[k] = sl
        if tp is not None:
            state.tp_levels[k] = tp
            
        state.set_position(k, new_u)
        
        orders_count += 1
        turnover += notional
        fills.append((k, filled_delta_u, exec_price, fee_paid))
    
    summary = {
        "orders": float(orders_count), 
        "turnover": float(turnover),
        "blocked": float(blocked_count)
    }
    return summary, fills


def check_exits(state: PortfolioState, snap: MarketSnapshot) -> List[Tuple[Key, float, float, Optional[float], Optional[float]]]:
    """Check all positions against SL/TP levels and generate close orders if hit.
    
    Returns list of orders: (key, -units, price, None, None)
    """
    exit_orders = []
    for k, units in state.holdings.items():
        if abs(units) < 1e-9:
            continue
            
        price = snap.prices.get(k)
        if not price:
            continue
            
        sl = state.sl_levels.get(k)
        tp = state.tp_levels.get(k)
        
        hit_exit = False
        reason = ""
        
        # Long Position
        if units > 0:
            if sl and price <= sl:
                hit_exit = True
                reason = "SL"
            elif tp and price >= tp:
                hit_exit = True
                reason = "TP"
        # Short Position
        elif units < 0:
            if sl and price >= sl:
                hit_exit = True
                reason = "SL"
            elif tp and price <= tp:
                hit_exit = True
                reason = "TP"
                
        if hit_exit:
            # Close entire position
            # Order: (key, -units, price, None, None)
            # We use current price as execution price (slippage will be applied in execute)
            exit_orders.append((k, -units, price, None, None))
            
    return exit_orders


def check_daily_loss(state: PortfolioState, snap: MarketSnapshot, limit_pct: float) -> bool:
    """Check if daily loss exceeds limit.
    
    Returns True if trading should stop (limit hit).
    """
    if limit_pct <= 0:
        return False
        
    # Calculate current equity
    equity = state.cash
    for k, u in state.holdings.items():
        p = float(snap.prices.get(k, 0.0))
        equity += float(u) * p
        
    # Update valuation in state for other components
    state.update_valuation(equity - state.cash)
        
    # If start equity is 0 (first run), init it
    if state.daily_start_equity <= 0:
        state.daily_start_equity = equity
        
    # Calculate PnL
    start = state.daily_start_equity
    if start <= 0:
        return False
        
    pnl = equity - start
    pnl_pct = pnl / start
    
    if pnl_pct < -limit_pct:
        return True
        
    return False


def rebalance_to_targets(state: PortfolioState, targets: List[Tuple[Key, float]], snap: MarketSnapshot, *, fee_bps: float = 0.0, slippage_bps: float = 0.0, next_bar_fill: bool = False, max_fill_fraction: float = 1.0) -> Dict[str, float]:
    """Rebalance holdings to target weights at snapshot prices.

    Supports next-bar fills and partial fills.

    Returns summary dict with 'orders' (count) and 'turnover' (notional traded).
    """
    # derive orders and execute them using optional fee/slippage, next-bar, partial fills
    orders = compute_rebalance_orders(state, targets, snap)
    summary, _fills = execute_orders(state, orders, fee_bps=fee_bps, slippage_bps=slippage_bps, next_bar_fill=next_bar_fill, max_fill_fraction=max_fill_fraction, snap=snap)
    return summary


def rebalance_to_targets_with_kelly(
    state: PortfolioState,
    targets: List[Tuple[Key, float]],
    snap: MarketSnapshot,
    kelly_sizers: Dict[Key, KellyCriterion],
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    next_bar_fill: bool = False,
    max_fill_fraction: float = 1.0,
    volatility_window: int = 20,
    annualization_factor: float = 252.0,
) -> Tuple[Dict[str, float], Dict[Key, Dict]]:
    """Rebalance with Kelly Criterion adaptive position sizing.
    
    Applies Kelly-based position sizing that accounts for:
    - Historical win rate and payoff ratio
    - Current volatility (inverse volatility weighting)
    - Current drawdown (reduce size during drawdowns)
    
    Args:
        state: Current portfolio state
        targets: List of (key, signal_weight) tuples
        snap: Market snapshot with prices and price history
        kelly_sizers: Dictionary of Kelly sizers per key
        fee_bps: Fee in basis points
        slippage_bps: Slippage in basis points
        next_bar_fill: Whether to use next bar prices
        max_fill_fraction: Maximum fill fraction
        volatility_window: Lookback window for volatility
        annualization_factor: Annualization factor for volatility
        
    Returns:
        Tuple of (summary dict, kelly stats dict per key)
    """
    # Compute orders with Kelly sizing
    orders = compute_rebalance_orders_with_kelly(
        state, targets, snap, kelly_sizers, volatility_window, annualization_factor
    )
    
    # Execute orders
    summary, fills = execute_orders(
        state, orders,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        next_bar_fill=next_bar_fill,
        max_fill_fraction=max_fill_fraction,
        snap=snap,
    )
    
    # Record closed trades for Kelly statistics
    record_closed_trades(state, fills, kelly_sizers)
    
    # Gather Kelly statistics
    kelly_stats = {}
    for k, kelly in kelly_sizers.items():
        kelly_stats[k] = kelly.get_summary()
        
    return summary, kelly_stats
