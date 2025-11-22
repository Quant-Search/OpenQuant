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


def check_schedule(hours: str | None, days: str | None) -> bool:
    """Check if current UTC time is within allowed schedule."""
    now = datetime.utcnow()
    
    # Check Day
    if days:
        allowed_days = [d.strip().upper()[:3] for d in days.split(",")]
        current_day = now.strftime("%a").upper()
        if current_day not in allowed_days:
            return False
            
    # Check Time
    if hours:
        try:
            start_str, end_str = hours.split("-")
            start_time = datetime.strptime(start_str.strip(), "%H:%M").time()
            end_time = datetime.strptime(end_str.strip(), "%H:%M").time()
            current_time = now.time()
            
            if start_time <= end_time:
                if not (start_time <= current_time <= end_time):
                    return False
            else:
                # Crosses midnight (e.g. 22:00-02:00)
                if not (current_time >= start_time or current_time <= end_time):
                    return False
        except ValueError:
            print(f"Warning: Invalid trading-hours format '{hours}'. Expected 'HH:MM-HH:MM'. Ignoring.")
            
    return True



def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--allocation", type=str, default=None, help="Path to allocation JSON; defaults to newest in reports/")
    p.add_argument("--state-file", type=str, default=str(Path("data") / "paper_state.json"))
    p.add_argument("--portfolio-db", type=str, default=str(Path("data") / "results.duckdb"), help="DuckDB path to store portfolio ledger (default: data/results.duckdb)")
    p.add_argument("--fee-bps", type=float, default=0.0, help="Execution fee in bps (default 0.0)")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage in bps applied to price in trade direction (default 0.0)")
    p.add_argument("--fill-model", type=str, choices=["immediate","next_bar"], default="immediate", help="Execution fill model (default immediate)")
    p.add_argument("--daily-loss-limit", type=float, default=0.02, help="Daily loss limit as fraction (default 0.02 = 2%)")
    p.add_argument("--close-on-limit", action="store_true", help="Close all positions if daily limit hit")
    p.add_argument("--trading-hours", type=str, default=None, help="Allowed trading hours in UTC (e.g. '09:30-16:00')")
    p.add_argument("--trading-days", type=str, default=None, help="Allowed trading days (e.g. 'MON,TUE,WED,THU,FRI')")
    args = p.parse_args()

    # Check Schedule
    if not check_schedule(args.trading_hours, args.trading_days):
        print("Trading schedule restriction: Current time is outside allowed window.")
        return 0


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
    
    # Check Daily Reset
    today = datetime.utcnow().strftime("%Y-%m-%d")
    # We need current equity to reset properly if needed, but we don't have prices yet.
    # We will do it after snapshot.

    # Fetch prices and build orders
    snap = _build_snapshot(allocation)
    
    # Calculate Equity for Reset check
    current_equity = state.cash
    for k, u in state.holdings.items():
        p = float(snap.prices.get(k, 0.0))
        current_equity += float(u) * p
        
    state.check_daily_reset(today, current_equity)
    
    # Check Daily Loss Limit
    from openquant.paper.simulator import check_daily_loss, close_all_positions_orders
    if check_daily_loss(state, snap, float(args.daily_loss_limit)):
        print(f"DAILY LOSS LIMIT HIT! Loss > {args.daily_loss_limit:.1%}")
        
        if args.close_on_limit:
            print("Closing all positions due to daily limit...")
            # We need a helper to generate close orders for everything
            # Or just use check_exits with forced close?
            # Let's create a simple close_all helper in simulator or here.
            # For now, let's iterate holdings and close them.
            close_orders = []
            for k, u in state.holdings.items():
                if abs(u) > 1e-9:
                    p = float(snap.prices.get(k, 0.0))
                    close_orders.append((k, -u, p, None, None))
            
            if close_orders:
                summary_close, fills_close = execute_orders(state, close_orders, fee_bps=float(args.fee_bps), slippage_bps=float(args.slippage_bps), snap=snap)
                
                # Record closes
                con = connect_portfolio_db(args.portfolio_db)
                try:
                    fills_db = []
                    for (k, delta_u, exec_px, fee_paid) in fills_close:
                        side = "BUY" if delta_u > 0 else "SELL"
                        notional = abs(delta_u) * exec_px
                        fills_db.append(OrderFill(
                            key=k, side=side, delta_units=float(delta_u), exec_price=float(exec_px),
                            notional=float(notional), fee_bps=float(args.fee_bps), slippage_bps=float(args.slippage_bps), fee_paid=float(fee_paid)
                        ))
                    record_rebalance(con, ts=datetime.utcnow(), fills=fills_db, state=state, snap=snap)
                finally:
                    con.close()
                    
            save_state(state, state_path)
            return 0
        else:
            print("Trading stopped for today. Use --close-on-limit to liquidate.")
            return 0
    
    # 1. Check Exits (SL/TP)
    # If any position hits SL/TP, we close it first.
    # This might conflict with rebalance if rebalance wants to buy it back immediately.
    # But usually if we hit SL, we want to stay out until next signal.
    # The allocation we loaded is the "new" signal.
    # If the new signal says "Buy" but we are at SL price, should we buy?
    # Maybe we should process exits, update state, THEN rebalance.
    
    from openquant.paper.simulator import check_exits
    exit_orders = check_exits(state, snap)
    if exit_orders:
        print(f"Executing {len(exit_orders)} stop-loss/take-profit orders...")
        summary_ex, fills_ex = execute_orders(state, exit_orders, fee_bps=float(args.fee_bps), slippage_bps=float(args.slippage_bps), snap=snap)
        # Record exits
        # ... (we should record these too, reusing the recording logic below would be better)
        # Let's combine exits and rebalance orders?
        # No, rebalance calculates target based on current holding. If we exit, holding becomes 0.
        # So we should execute exits, then re-calculate rebalance?
        # If we re-calculate rebalance, and the allocation still says "Buy", we will buy again immediately.
        # This implies we need a "cooldown" or the allocation generation should have known about the SL.
        # For now, let's assume the allocation is authoritative.
        # BUT, if the allocation was generated "offline" without knowing we just hit SL in the last minute...
        # A simple rule: If we exit via SL/TP, we do NOT re-enter the same symbol in this run.
        
        # Execute exits
        # We need to record them.
        # Let's just add them to a list of executed fills to record later.
        pass # We will handle execution below in a unified way if possible, or separate.
        
    # Actually, let's execute exits first, update state, then compute rebalance, 
    # but filter out re-entries for symbols that just exited?
    
    # For simplicity in this iteration:
    # 1. Execute Exits.
    # 2. Compute Rebalance.
    # 3. Execute Rebalance.
    
    fills_all = []
    
    if exit_orders:
        summary_ex, fills_ex = execute_orders(state, exit_orders, fee_bps=float(args.fee_bps), slippage_bps=float(args.slippage_bps), snap=snap)
        fills_all.extend(fills_ex)
        
    # Re-compute targets? 
    # If we just closed EURUSD because of SL, state.position is 0.
    # If allocation says EURUSD weight 0.5, compute_rebalance will say "Buy 0.5".
    # We need to prevent this.
    
    exited_keys = set(k for k, _, _, _, _ in exit_orders)
    
    # Filter targets to exclude exited keys
    targets_filtered = [t for t in targets if t[0] not in exited_keys]
    
    orders = compute_rebalance_orders(state, targets_filtered, snap)
    
    # Inject SL/TP from allocation into orders
    # orders is list of (key, delta, price, None, None)
    # We need to map key -> (sl, tp) from allocation
    
    # Build map from key to (sl, tp)
    # We need to reconstruct key from allocation entries to match
    sl_tp_map = {}
    for entry in allocation:
        try:
            k: Key = (
                str(entry.get("exchange", "")).upper(),
                str(entry.get("symbol", "")),
                str(entry.get("timeframe", "")),
                str(entry.get("strategy", "")),
            )
            sl = float(entry.get("sl", 0.0))
            tp = float(entry.get("tp", 0.0))
            if sl > 0 or tp > 0:
                sl_tp_map[k] = (sl if sl > 0 else None, tp if tp > 0 else None)
        except:
            pass
            
    # Update orders with SL/TP
    orders_with_risk = []
    for (k, delta, price, _, _) in orders:
        sl, tp = sl_tp_map.get(k, (None, None))
        orders_with_risk.append((k, delta, price, sl, tp))
    
    orders = orders_with_risk
    # Execute with fee/slippage, next-bar fills, and partial fills (configurable)
    next_bar_fill = getattr(args, 'next_bar_fill', False)
    max_fill_fraction = getattr(args, 'max_fill_fraction', 1.0)
    summary, fills_raw = execute_orders(state, orders, fee_bps=float(args.fee_bps), slippage_bps=float(args.slippage_bps), next_bar_fill=next_bar_fill, max_fill_fraction=max_fill_fraction, snap=snap)
    
    fills_all.extend(fills_raw)

    # Record to portfolio DB
    from datetime import datetime as _dt
    ts_dt = _dt.utcnow()
    con = connect_portfolio_db(args.portfolio_db)
    try:
        fills: List[OrderFill] = []
        for (k, delta_u, exec_px, fee_paid) in fills_all:
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

