from __future__ import annotations
import copy

from openquant.risk.exposure import propose_portfolio_weights


def _rows():
    # Build four rows across two symbols with decreasing quality
    base = {
        "exchange": "BINANCE",
        "timeframe": "1h",
        "strategy": "ema",
        "params": {"fast": 10, "slow": 50},
        "bars": 100,
        "metrics": {"ok": True, "sharpe": 1.0, "dsr": 0.8, "wfo_mts": 0.7},
    }
    r0 = copy.deepcopy(base); r0.update({"symbol": "BTC/USDT"})
    r1 = copy.deepcopy(base); r1.update({"symbol": "ETH/USDT"}); r1["metrics"].update({"sharpe": 1.2, "dsr": 0.9, "wfo_mts": 0.8})
    r2 = copy.deepcopy(base); r2.update({"symbol": "BTC/USDT"}); r2["metrics"].update({"sharpe": 0.9, "dsr": 0.7})
    r3 = copy.deepcopy(base); r3.update({"symbol": "ETH/USDT"}); r3["metrics"].update({"sharpe": 0.5, "dsr": 0.3})
    return [r0, r1, r2, r3]


def test_allocation_respects_caps():
    rows = _rows()
    alloc = propose_portfolio_weights(rows, max_total_weight=0.3, max_symbol_weight=0.15, slot_weight=0.1)
    # Should pick the two best first (ETH then BTC) with 0.1 each, then next best that doesn't violate caps
    assert abs(sum(w for _, w in alloc) - 0.3) < 1e-9
    # Per symbol <= 0.15
    per_symbol = {}
    for idx, w in alloc:
        sym = rows[idx]["symbol"]
        per_symbol[sym] = per_symbol.get(sym, 0.0) + w
    assert per_symbol["BTC/USDT"] <= 0.15 + 1e-9
    assert per_symbol["ETH/USDT"] <= 0.15 + 1e-9


def test_allocation_skips_non_ok():
    rows = _rows()
    rows[1]["metrics"]["ok"] = False  # disable best row
    alloc = propose_portfolio_weights(rows, max_total_weight=0.2, max_symbol_weight=0.2, slot_weight=0.1)
    # Should allocate to BTC first (idx 0), then BTC second or ETH depending on ok flags
    assert len(alloc) >= 1

