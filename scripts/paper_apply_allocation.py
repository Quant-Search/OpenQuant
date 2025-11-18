from __future__ import annotations
"""Apply latest allocation JSON to the paper PortfolioState using live prices.

- Loads newest reports/allocation_*.json unless --allocation is provided
- Loads/saves PortfolioState from/to --state-file (default: data/paper_state.json)
- Fetches prices using ccxt for each (exchange, symbol)
- Computes target units from weights and rebalances holdings
- Writes a short execution summary to reports/paper_execution_*.md
"""
import argparse
import glob
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt  # type: ignore

from openquant.paper.state import PortfolioState, Key
from openquant.paper.io import load_state, save_state
from openquant.paper.simulator import MarketSnapshot, rebalance_to_targets, compute_rebalance_orders, execute_orders
from openquant.storage.portfolio_db import connect as connect_portfolio_db, record_rebalance, OrderFill


def _latest_allocation_path(reports_dir: Path) -> Path | None:
    files = sorted(reports_dir.glob("allocation_*.json"))
    return files[-1] if files else None


def _load_allocation(path: Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Allocation JSON must be a list of entries")
    return data


def _exchange_client(name: str):
    name_l = name.lower()
    if name_l == "binance":
        return ccxt.binance()
    # add more exchanges as needed
    raise ValueError(f"Unsupported exchange: {name}")


def _build_snapshot(allocation: List[Dict[str, object]]) -> MarketSnapshot:
    prices: Dict[Key, float] = {}
    next_prices: Dict[Key, float] = {}  # For next-bar fills, use same price as estimate (can be improved later)
    # group by exchange for fewer clients
    clients: Dict[str, object] = {}
    for entry in allocation:
        ex = str(entry.get("exchange", "")).lower()
        sym = str(entry.get("symbol", ""))
        tf = str(entry.get("timeframe", ""))
        strat = str(entry.get("strategy", ""))
        if not ex or not sym:
            continue
        if ex not in clients:
            clients[ex] = _exchange_client(ex)
        client = clients[ex]
        # fetch ticker price
        try:
            ticker = client.fetch_ticker(sym)  # type: ignore[attr-defined]
            price = float(ticker.get("last") or ticker.get("close") or 0.0)
        except Exception:
            price = 0.0
        key = (ex.upper(), sym, tf, strat)
        prices[key] = price
        next_prices[key] = price  # Placeholder: use current price for next-bar (realistic would fetch next bar)
    return MarketSnapshot(prices=prices, next_prices=next_prices)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--allocation", type=str, default=None, help="Path to allocation JSON; defaults to newest in reports/")
    p.add_argument("--state-file", type=str, default=str(Path("data") / "paper_state.json"))
    p.add_argument("--portfolio-db", type=str, default=str(Path("data") / "results.duckdb"), help="DuckDB path to store portfolio ledger (default: data/results.duckdb)")
    p.add_argument("--fee-bps", type=float, default=0.0, help="Execution fee in bps (default 0.0)")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage in bps applied to price in trade direction (default 0.0)")
    p.add_argument("--fill-model", type=str, choices=["immediate","next_bar"], default="immediate", help="Execution fill model (default immediate)")
    args = p.parse_args()

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    alloc_path = Path(args.allocation) if args.allocation else _latest_allocation_path(reports_dir)
    if not alloc_path or not alloc_path.exists():
        print("No allocation JSON found in reports/. Run universe research first.")
        return 1

    allocation = _load_allocation(alloc_path)

    # Build targets [(Key, weight)]
    targets: List[Tuple[Key, float]] = []
    for entry in allocation:
        try:
            key: Key = (
                str(entry.get("exchange", "")).upper(),
                str(entry.get("symbol", "")),
                str(entry.get("timeframe", "")),
                str(entry.get("strategy", "")),
            )
            w = float(entry.get("weight", 0.0))
            if w <= 0.0:
                continue
            targets.append((key, w))
        except Exception:
            continue

    if not targets:
        print("Allocation JSON has no positive-weight targets.")
        return 0

    # Load state
    state_path = Path(args.state_file)
    state = load_state(state_path)

    # Fetch prices and build orders
    snap = _build_snapshot(allocation)
    orders = compute_rebalance_orders(state, targets, snap)
    # Execute with fee/slippage, next-bar fills, and partial fills (configurable)
    next_bar_fill = getattr(args, 'next_bar_fill', False)
    max_fill_fraction = getattr(args, 'max_fill_fraction', 1.0)
    summary, fills_raw = execute_orders(state, orders, fee_bps=float(args.fee_bps), slippage_bps=float(args.slippage_bps), next_bar_fill=next_bar_fill, max_fill_fraction=max_fill_fraction, snap=snap)

    # Record to portfolio DB
    from datetime import datetime as _dt
    ts_dt = _dt.utcnow()
    con = connect_portfolio_db(args.portfolio_db)
    try:
        fills: List[OrderFill] = []
        for (k, delta_u, exec_px, fee_paid) in fills_raw:
            side = "BUY" if delta_u > 0 else "SELL"
            notional = abs(delta_u) * exec_px
            fills.append(OrderFill(
                key=k,
                side=side,
                delta_units=float(delta_u),
                exec_price=float(exec_px),
                notional=float(notional),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slippage_bps),
                fee_paid=float(fee_paid),
            ))
        record_rebalance(con, ts=ts_dt, fills=fills, state=state, snap=snap)
    finally:
        con.close()

    # Save state
    save_state(state, state_path)

    # Write summary
    ts = ts_dt.strftime("%Y%m%d_%H%M%S")
    lines = [
        f"time: {ts}",
        f"state_file: {state_path}",
        f"allocation_file: {alloc_path}",
        f"orders: {summary['orders']}",
        f"turnover: {summary['turnover']:.6f}",
        f"fee_bps: {float(args.fee_bps)}",
        f"slippage_bps: {float(args.slippage_bps)}",
        f"portfolio_db: {args.portfolio_db}",
    ]
    with open(reports_dir / f"paper_execution_{ts}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Rebalance complete:", ", ".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

