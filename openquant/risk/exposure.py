from __future__ import annotations
"""Portfolio exposure capping and allocation helpers for paper-trading phase.

- propose_portfolio_weights: Greedy ranking-based allocation under caps.

All inputs are pure-Python data; no side effects. This module does not place orders.
"""
from typing import List, Dict, Any, Tuple
import math


def _score(row: Dict[str, Any]) -> Tuple[int, float, float, float]:
    """Ranking tuple: has_wfo, wfo_mts, dsr, sharpe (desc)."""
    m = (row.get("metrics") or {})
    w = m.get("wfo_mts")
    d = m.get("dsr")
    s = m.get("sharpe")
    def fin(x):
        try:
            return float(x)
        except Exception:
            return -math.inf
    return (1 if w is not None else 0, fin(w), fin(d), fin(s))


def propose_portfolio_weights(
    rows: List[Dict[str, Any]],
    *,
    max_total_weight: float = 1.0,
    max_symbol_weight: float = 0.2,
    slot_weight: float = 0.05,
    volatility_adjusted: bool = True,  # New parameter
) -> List[Tuple[int, float]]:
    """Propose portfolio weights (index, weight) for OK rows under exposure caps.

    - rows: list of research rows as produced by the runner
    - max_total_weight: total portfolio cap (e.g., 1.0 = 100%)
    - max_symbol_weight: per-symbol cap (e.g., 0.2 = 20%)
    - slot_weight: base weight to assign per accepted candidate (e.g., 0.05 = 5%)
    - volatility_adjusted: if True, adjust weights by inverse volatility

    Returns a list of (row_index, weight) for rows that receive non-zero weight,
    in the order they are accepted by the greedy allocator.
    """
    if not rows:
        return []
    # Filter to OK rows and order by descending score
    candidates = [(i, r) for i, r in enumerate(rows) if (r.get("metrics") or {}).get("ok")]
    candidates.sort(key=lambda t: _score(t[1]), reverse=True)
    
    # Correlation filter disabled for performance (returns not stored)
    # from .correlation import filter_correlated_candidates
    # candidates = filter_correlated_candidates(candidates, threshold=0.8)
    
    # Calculate volatility adjustments if enabled
    # Uses inverse volatility: lower volatility assets get higher weights
    # This implements risk parity-like allocation
    vol_factors = {}
    if volatility_adjusted:
        # Step 1: Collect volatility for each candidate
        for idx, row in candidates:
            symbol = row.get("symbol")
            if not symbol:
                continue

            metrics = row.get("metrics") or {}

            # Try multiple sources for volatility:
            # 1. Direct max_dd (drawdown as proxy for risk)
            # 2. Returns std if available
            # 3. Default to 1.0
            max_dd = metrics.get("max_dd")
            returns = metrics.get("returns")

            vol = None
            if max_dd is not None and max_dd > 0:
                # Use max drawdown as risk proxy (higher DD = higher risk)
                vol = float(max_dd)
            elif returns is not None and hasattr(returns, 'std'):
                vol = float(returns.std())

            if vol is not None and vol > 0:
                # Inverse volatility: lower vol/dd -> higher weight
                vol_factors[symbol] = 1.0 / max(vol, 0.01)
            else:
                vol_factors[symbol] = 1.0

        # Step 2: Normalize to median = 1.0 for stability
        if vol_factors:
            sorted_factors = sorted(vol_factors.values())
            median_factor = sorted_factors[len(sorted_factors) // 2]
            if median_factor > 0:
                for sym in vol_factors:
                    vol_factors[sym] /= median_factor
                    # Clamp to prevent extreme weights (0.5x to 2x of base)
                    vol_factors[sym] = max(0.5, min(2.0, vol_factors[sym]))

    assigned_total = 0.0
    per_symbol: Dict[str, float] = {}
    out: List[Tuple[int, float]] = []

    # Get current holdings symbols for correlation check
    # Note: holdings is not passed to this function currently. 
    # We should update the signature or assume we only check within the proposed batch for now.
    # For now, we check within the proposed batch.
    
    from .forex_correlation import check_portfolio_correlation

    for idx, row in candidates:
        if assigned_total >= max_total_weight - 1e-12:
            break
        sym = row.get("symbol")
        if not isinstance(sym, str):
            continue

        # Check Correlation with *other* symbols in this batch (not the same symbol)
        # We allow multiple entries for the same symbol (different strategies)
        # as long as per-symbol cap is respected
        other_symbols = [s for s in per_symbol.keys() if s != sym]
        if check_portfolio_correlation(sym, other_symbols, threshold=0.8):
            continue
            
        sym_used = per_symbol.get(sym, 0.0)
        remaining_total = max(0.0, max_total_weight - assigned_total)
        remaining_sym = max(0.0, max_symbol_weight - sym_used)
        
        # Apply volatility adjustment
        base_w = slot_weight
        if volatility_adjusted and sym in vol_factors:
            base_w *= vol_factors[sym]
        
        w = min(base_w, remaining_total, remaining_sym)
        if w <= 1e-12:
            continue
        out.append((idx, w))
        assigned_total += w
        per_symbol[sym] = sym_used + w

    return out

