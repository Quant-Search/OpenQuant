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
    vol_factors = {}
    if volatility_adjusted:
        # Calculate inverse volatility factors
        for idx, row in candidates:
            metrics = row.get("metrics") or {}
            returns = metrics.get("returns")
            if returns is not None and hasattr(returns, 'std'):
                vol = returns.std()
                # Inverse volatility: lower vol → higher weight
                # Store by symbol for easier lookup later
                symbol = row.get("symbol")
                if symbol:
                    vol_factors[symbol] = 1.0 / max(vol, 1e-6) if vol > 0 else 1.0
            else:
                symbol = row.get("symbol")
                if symbol:
                    vol_factors[symbol] = 1.0
        
        # Normalize vol_factors
        if vol_factors:
            median_factor = sorted(vol_factors.values())[len(vol_factors) // 2]
            for idx in vol_factors:
                vol_factors[idx] /= median_factor

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
            
        # Check Correlation with *newly proposed* symbols in this batch
        # (We don't have access to existing holdings here easily without changing signature)
        if check_portfolio_correlation(sym, list(per_symbol.keys()), threshold=0.8):
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

