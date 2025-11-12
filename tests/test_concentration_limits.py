from __future__ import annotations
import copy
from openquant.risk.guardrails import apply_concentration_limits


def _make_rows():
    # Four rows for the same symbol across two strategies with distinct scores
    return [
        {"symbol": "BTC/USDT", "strategy": "sma", "metrics": {"ok": True, "wfo_mts": 0.30, "dsr": 1.0, "sharpe": 1.2}},
        {"symbol": "BTC/USDT", "strategy": "ema", "metrics": {"ok": True, "wfo_mts": 0.50, "dsr": 0.8, "sharpe": 1.1}},
        {"symbol": "BTC/USDT", "strategy": "sma", "metrics": {"ok": True, "wfo_mts": None, "dsr": 0.9, "sharpe": 1.0}},
        {"symbol": "BTC/USDT", "strategy": "ema", "metrics": {"ok": True, "wfo_mts": None, "dsr": 0.2, "sharpe": 0.4}},
    ]


def test_symbol_level_cap_keeps_top_k():
    rows = _make_rows()
    out = apply_concentration_limits(copy.deepcopy(rows), max_per_symbol=2)
    oks = [r["metrics"]["ok"] for r in out]
    # Highest ranks should be the two with non-null wfo_mts: indices 1 and 0 remain True
    assert oks == [True, True, False, False]


def test_strategy_per_symbol_cap_keeps_top_per_strategy():
    rows = _make_rows()
    out = apply_concentration_limits(copy.deepcopy(rows), max_per_strategy_per_symbol=1)
    oks = [r["metrics"]["ok"] for r in out]
    # For SMA group, keep index 0 over 2; for EMA group, keep index 1 over 3
    assert oks == [True, True, False, False]

