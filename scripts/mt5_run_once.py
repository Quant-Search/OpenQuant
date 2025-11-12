"""
One-click MT5 FX research + paper rebalance + MT5 mirroring (single run).

- Fetches data from MT5, researches best configs, writes allocation JSON
- Builds live price snapshot from MT5 ticks
- Rebalances the paper portfolio, records ledger in DuckDB
- Mirrors allocation to MT5 with market orders

Usage:
  python scripts/mt5_run_once.py

Environment (or edit constants below):
  OQ_MT5_TERMINAL = path to terminal64.exe
  OQ_MT5_SERVER   = MT5 server name (e.g., MetaQuotes-Demo)
  OQ_MT5_LOGIN    = account login (int)
  OQ_MT5_PASSWORD = account password
"""
from __future__ import annotations

# Standard library imports
import os  # for environment variables
import json  # for reading allocation json
import sys  # for PYTHONPATH injection when run from scripts/
from pathlib import Path  # for filesystem paths
from datetime import datetime, timezone  # for timestamps
from typing import List, Tuple, Dict  # for type hints

# Ensure repository root is on sys.path when launched from scripts/
try:
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
except Exception:
    pass

# Local project imports
from openquant.data import mt5_source as mt5s  # MT5 data/config helper
from openquant.research.universe_runner import run_universe  # research entrypoint
from openquant.paper.io import load_state, save_state  # paper state IO
from openquant.paper.state import Key as KeyT  # key type for holdings
from openquant.paper.simulator import (
    MarketSnapshot,  # container for current prices
    compute_rebalance_orders,  # compute target delta units
    execute_orders,  # simulate execution (fees + slippage)
)
from openquant.storage.portfolio_db import (
    connect as connect_db,  # connect to DuckDB
    record_rebalance,  # write trades + positions + equity
    OrderFill,  # dataclass to capture fills for DB
)


def _load_allocation_latest(reports_dir: Path) -> List[Dict[str, object]]:
    """Load newest allocation_*.json and return a list of entries.
    Accepts either list[...] or {"allocations": [...]} formats.
    """
    files = sorted(reports_dir.glob("allocation_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No allocation_*.json found in reports/. Run research first.")
    p = files[-1]
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "allocations" in data:
        return list(data["allocations"])  # type: ignore[no-any-return]
    if isinstance(data, list):
        return list(data)
    raise ValueError("Allocation JSON must be a list or a dict with 'allocations'")


def _build_mt5_snapshot(allocation: List[Dict[str, object]]) -> MarketSnapshot:
    """Build a MarketSnapshot using MT5 ticks for all MT5 entries in allocation."""
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError(f"MetaTrader5 module import failed: {e}")

    # Ensure the MT5 connection is initialized (mt5s.configure() must be called first)
    mt5s._ensure_init()  # type: ignore[attr-defined]

    prices: Dict[KeyT, float] = {}
    for e in allocation:
        ex = str(e.get("exchange", "")).upper()
        sym = str(e.get("symbol", ""))
        tf = str(e.get("timeframe", ""))
        strat = str(e.get("strategy", ""))
        if not sym:
            continue
        key: KeyT = (ex, sym, tf, strat)
        if ex == "MT5":
            tick = mt5.symbol_info_tick(sym)
            px = float(getattr(tick, "last", 0.0) or getattr(tick, "bid", 0.0) or getattr(tick, "ask", 0.0) or 0.0)
            prices[key] = px
        else:
            prices[key] = 0.0
    return MarketSnapshot(prices=prices)


def main() -> int:
    # Read credentials strictly from environment (do not hardcode secrets)
    MT5_PATH = os.environ.get("OQ_MT5_TERMINAL")
    MT5_SERVER = os.environ.get("OQ_MT5_SERVER")
    MT5_LOGIN = os.environ.get("OQ_MT5_LOGIN")
    MT5_PASSWORD = os.environ.get("OQ_MT5_PASSWORD")
    if not (MT5_PATH and MT5_SERVER and MT5_LOGIN and MT5_PASSWORD):
        raise SystemExit(
            "Missing MT5 credentials. Please set OQ_MT5_TERMINAL, OQ_MT5_SERVER, OQ_MT5_LOGIN, OQ_MT5_PASSWORD in your environment."
        )
    MT5_LOGIN = int(MT5_LOGIN)

    print("[1/6] Configuring MT5...")
    # Store credentials for lazy init used by data fetch and snapshot
    mt5s.configure(terminal_path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    print("       MT5 configured.")

    print("[2/6] Running MT5 FX research (top_n=5, optuna_trials=12)... This can take several minutes.")
    # Run research with MT5 as the universe source
    # Returns path to a summary report; allocation JSON is written in reports/ as allocation_*.json
    _ = run_universe(exchange="mt5", top_n=5, fetch_workers=4, max_workers=None, optuna_trials=12)
    print("       Research finished.")

    print("[3/6] Loading latest allocation JSON from reports/ ...")
    allocation = _load_allocation_latest(Path("reports"))
    print(f"       Loaded {len(allocation)} entries.")

    print("[4/6] Paper rebalance (build MT5 tick snapshot, compute/execute, record ledger)...")
    # Build live snapshot
    snap = _build_mt5_snapshot(allocation)

    # Load current paper state (creates default if not present)
    state_path = Path("data") / "paper_state.json"
    state = load_state(state_path)

    # Build targets from allocation weights
    targets: List[Tuple[KeyT, float]] = []
    for e in allocation:
        try:
            w = float(e.get("weight", 0.0))
            if w <= 0.0:
                continue
            k: KeyT = (
                str(e.get("exchange", "")).upper(),
                str(e.get("symbol", "")),
                str(e.get("timeframe", "")),
                str(e.get("strategy", "")),
            )
            targets.append((k, w))
        except Exception:
            continue

    # Compute and execute orders with fees/slippage
    orders = compute_rebalance_orders(state, targets, snap)
    summary, fills_raw = execute_orders(state, orders, fee_bps=2.0, slippage_bps=5.0)

    # Record into DuckDB portfolio ledger
    con = connect_db("data/results.duckdb")
    try:
        ts_dt = datetime.now(timezone.utc).replace(tzinfo=None)
        fills: List[OrderFill] = []
        for (k, delta_u, exec_px, fee_paid) in fills_raw:
            side = "BUY" if delta_u > 0 else "SELL"
            notional = abs(delta_u) * exec_px
            fills.append(
                OrderFill(
                    key=k,
                    side=side,
                    delta_units=float(delta_u),
                    exec_price=float(exec_px),
                    notional=float(notional),
                    fee_bps=2.0,
                    slippage_bps=5.0,
                    fee_paid=float(fee_paid),
                )
            )
        record_rebalance(con, ts=ts_dt, fills=fills, state=state, snap=snap)
    finally:
        con.close()

    # Persist state
    save_state(state, state_path)
    print(f"       Paper rebalance done. orders={summary.get('orders',0)} turnover={summary.get('turnover',0.0):.6f}")

    print("[5/6] Mirroring allocation to MT5 (market orders)...")
    from openquant.paper.mt5_bridge import apply_allocation_to_mt5  # local import to avoid MT5 hard dep on import
    targets_mt5 = apply_allocation_to_mt5(
        allocation,
        terminal_path=MT5_PATH,
        login=MT5_LOGIN,
        password=MT5_PASSWORD,
        server=MT5_SERVER,
    )
    print(f"       MT5 target lots: {targets_mt5}")

    print("[6/6] Done. Check MetaTrader 5 terminal for positions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

