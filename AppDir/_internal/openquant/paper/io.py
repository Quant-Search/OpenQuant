from __future__ import annotations
"""JSON I/O utilities for paper-trading state.

We store PortfolioState in a simple JSON file to keep Phase 3 minimal.
Later we can add a DuckDB-backed portfolio ledger if needed.
"""
from typing import Any, Dict, Tuple
import json
from pathlib import Path

from .state import PortfolioState, Key


def save_state(state: PortfolioState, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data: Dict[str, Any] = {
        "cash": float(state.cash),
        "holdings": {"|".join(k): float(v) for k, v in state.holdings.items()},
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_state(path: str | Path) -> PortfolioState:
    p = Path(path)
    if not p.exists():
        return PortfolioState()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    cash = float(data.get("cash", 100_000.0))
    holdings_raw: Dict[str, float] = data.get("holdings", {})
    holdings: Dict[Key, float] = {}
    for skey, val in holdings_raw.items():
        parts = skey.split("|")
        if len(parts) != 4:
            continue
        holdings[(parts[0], parts[1], parts[2], parts[3])] = float(val)
    return PortfolioState(cash=cash, holdings=holdings)

