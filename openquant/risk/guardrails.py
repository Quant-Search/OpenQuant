from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import math


def apply_guardrails(
    max_drawdown: float,
    cvar: float,
    worst_daily_loss: Optional[float],
    *,
    dd_limit: Optional[float] = None,      # e.g., 0.2 for 20%
    cvar_limit: Optional[float] = None,    # e.g., 0.05 (per-day tail mean)
    daily_loss_cap: Optional[float] = None # e.g., 0.05 for 5% per day
) -> Tuple[bool, List[str]]:
    """
    Evaluate simple risk guardrails for a backtest result.
    Returns (ok, reasons) where reasons lists violated constraints.
    All inputs are positive magnitudes (e.g., max_drawdown=0.25 means -25%).
    """
    reasons: List[str] = []

    if dd_limit is not None and max_drawdown > dd_limit:
        reasons.append(f"max_drawdown {max_drawdown:.4f} > limit {dd_limit:.4f}")

    if cvar_limit is not None and cvar > cvar_limit:
        reasons.append(f"cvar {cvar:.4f} > limit {cvar_limit:.4f}")

    if daily_loss_cap is not None and worst_daily_loss is not None and worst_daily_loss > daily_loss_cap:
        reasons.append(f"worst_daily_loss {worst_daily_loss:.4f} > cap {daily_loss_cap:.4f}")

    return (len(reasons) == 0), reasons


def apply_concentration_limits(
    rows: List[Dict],
    *,
    max_per_symbol: Optional[int] = None,
    max_per_strategy_per_symbol: Optional[int] = None,
) -> List[Dict]:
    """
    Apply concentration limits by marking excess rows as ok=False.
    Ranking preference: wfo_mts (desc) then dsr (desc) then sharpe (desc).
    Limits apply to rows whose metrics.ok is True; non-ok rows remain unchanged.
    """
    if (max_per_symbol is None or max_per_symbol <= 0) and (
        max_per_strategy_per_symbol is None or max_per_strategy_per_symbol <= 0
    ):
        return rows

    def score_of(row: Dict) -> tuple:
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

    # Build symbol -> indices for ok rows
    by_symbol: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        m = r.get("metrics") or {}
        if not m.get("ok", False):
            continue
        sym = r.get("symbol")
        if not isinstance(sym, str):
            continue
        by_symbol.setdefault(sym, []).append(i)

    # Symbol-level cap
    if max_per_symbol is not None and max_per_symbol > 0:
        for sym, idxs in by_symbol.items():
            sorted_idxs = sorted(idxs, key=lambda i: score_of(rows[i]), reverse=True)
            keep = set(sorted_idxs[:max_per_symbol])
            for i in sorted_idxs[max_per_symbol:]:
                rows[i]["metrics"]["ok"] = False

    # Strategy-per-symbol cap
    if max_per_strategy_per_symbol is not None and max_per_strategy_per_symbol > 0:
        # Rebuild groups using possibly updated ok flags
        by_pair: Dict[Tuple[str, str], List[int]] = {}
        for i, r in enumerate(rows):
            m = r.get("metrics") or {}
            if not m.get("ok", False):
                continue
            sym = r.get("symbol"); strat = r.get("strategy")
            if isinstance(sym, str) and isinstance(strat, str):
                by_pair.setdefault((sym, strat), []).append(i)
        for key, idxs in by_pair.items():
            sorted_idxs = sorted(idxs, key=lambda i: score_of(rows[i]), reverse=True)
            for i in sorted_idxs[max_per_strategy_per_symbol:]:
                rows[i]["metrics"]["ok"] = False

    return rows

