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
) -> List[Tuple[int, float]]:
    """Propose portfolio weights (index, weight) for OK rows under exposure caps.

    - rows: list of research rows as produced by the runner
    - max_total_weight: total portfolio cap (e.g., 1.0 = 100%)
    - max_symbol_weight: per-symbol cap (e.g., 0.2 = 20%)
    - slot_weight: base weight to assign per accepted candidate (e.g., 0.05 = 5%)

    Returns a list of (row_index, weight) for rows that receive non-zero weight,
    in the order they are accepted by the greedy allocator.
    """
    if not rows:
        return []
    # Filter to OK rows and order by descending score
    candidates = [(i, r) for i, r in enumerate(rows) if (r.get("metrics") or {}).get("ok")]
    candidates.sort(key=lambda t: _score(t[1]), reverse=True)

    assigned_total = 0.0
    per_symbol: Dict[str, float] = {}
    out: List[Tuple[int, float]] = []

    for idx, row in candidates:
        if assigned_total >= max_total_weight - 1e-12:
            break
        sym = row.get("symbol")
        if not isinstance(sym, str):
            continue
        sym_used = per_symbol.get(sym, 0.0)
        remaining_total = max(0.0, max_total_weight - assigned_total)
        remaining_sym = max(0.0, max_symbol_weight - sym_used)
        w = min(slot_weight, remaining_total, remaining_sym)
        if w <= 1e-12:
            continue
        out.append((idx, w))
        assigned_total += w
        per_symbol[sym] = sym_used + w

    return out

