from __future__ import annotations
"""MetaTrader5 bridge for demo/live integration (guarded import).

- Tries to initialize MT5 if installed
- Can login if credentials provided
- Provides helpers to map symbols, compute volumes and send market orders

NOTE: UI chart attachment requires an EA; this bridge focuses on trading sync.
"""
from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path
import json
import time

_MT5 = None


def _lazy_import() -> Optional[Any]:
    global _MT5
    if _MT5 is not None:
        return _MT5
    try:
        import MetaTrader5 as mt5  # type: ignore
        _MT5 = mt5
        return _MT5
    except Exception:
        _MT5 = None
        return None


def is_available() -> bool:
    return _lazy_import() is not None


def validate_credentials(login: Optional[int] = None, server: Optional[str] = None, terminal_path: Optional[str] = None) -> Tuple[bool, str]:
    """Validate MT5 credentials before attempting connection."""
    if terminal_path:
        p = Path(terminal_path)
        if not p.exists():
            return False, f"Terminal path does not exist: {terminal_path}"
    
    if login is not None:
        try:
            int(login)
        except ValueError:
            return False, f"Login must be an integer, got: {login}"
            
    if login and not server:
        return False, "Server must be provided if login is specified"
        
    return True, "OK"


def init(login: Optional[int] = None, server: Optional[str] = None, password: Optional[str] = None, *, terminal_path: Optional[str] = None) -> bool:
    # 1. Validate inputs first
    is_valid, msg = validate_credentials(login, server, terminal_path)
    if not is_valid:
        # We can't log easily here without circular imports or setup, so just return False
        # Ideally we'd raise or log. For now, let's print to stderr if possible or just fail safe.
        return False

    mt5 = _lazy_import()
    if not mt5:
        return False
    try:
        # If already initialized and connected, reuse the session
        try:
            acc = mt5.account_info()
            if acc is not None:
                if login and int(getattr(acc, "login", 0) or 0) != int(login):
                    # Logged in to a different account -> try switching
                    if not (password and server):
                        return False
                    if not mt5.login(login, password=password, server=server):  # type: ignore[arg-type]
                        return False
                return True
        except Exception:
            pass

        # Not connected: Try initialize (optionally pass path for portable installs)
        ok = False
        last_err = None
        if terminal_path:
            try:
                Path(terminal_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            ok = mt5.initialize(path=terminal_path)  # type: ignore[arg-type]
            if not ok and hasattr(mt5, "last_error"):
                try:
                    last_err = mt5.last_error()
                except Exception:
                    last_err = None
        if not ok:
            ok = mt5.initialize()  # type: ignore[arg-type]
            if not ok and hasattr(mt5, "last_error") and last_err is None:
                try:
                    last_err = mt5.last_error()
                except Exception:
                    last_err = None
        if not ok:
            return False
        # Optional login
        if login and password and server:
            ok2 = mt5.login(login, password=password, server=server)  # type: ignore[arg-type]
            if not ok2:
                return False
        return True
    except Exception:
        return False


def shutdown() -> None:
    mt5 = _lazy_import()
    if not mt5:
        return
    try:
        mt5.shutdown()
    except Exception:
        pass


def _load_symbol_map(path: str | Path = "data/mt5_symbol_map.json") -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {str(k): str(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


def map_symbol(binance_symbol: str, *, symmap: Optional[Dict[str, str]] = None) -> str:
    """Best-effort mapping like BTC/USDT -> BTCUSD.
    Allow override via data/mt5_symbol_map.json.
    """
    sm = symmap or _load_symbol_map()
    if binance_symbol in sm:
        return sm[binance_symbol]
    s = binance_symbol.upper()
    if s.endswith("/USDT"):
        s = s.replace("/USDT", "USD")
    return s.replace("/", "")


def _round_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    n = round(x / step)
    return max(n * step, step)


def _ensure_symbol(mt5, sym: str) -> bool:  # type: ignore[no-untyped-def]
    info = mt5.symbol_info(sym)
    if info is None:
        mt5.symbol_select(sym, True)
        info = mt5.symbol_info(sym)
    return info is not None


def positions_by_symbol(mt5) -> Dict[str, float]:  # type: ignore[no-untyped-def]
    out: Dict[str, float] = {}
    try:
        poss = mt5.positions_get()
        if poss is None:
            return {}
        for p in poss:
            vol = float(p.volume)
            sym = str(p.symbol)
            # netting assumption
            out[sym] = out.get(sym, 0.0) + (vol if int(p.type) == 0 else -vol)
        return out
    except Exception:
        return {}


def account_equity(mt5) -> float:  # type: ignore[no-untyped-def]
    try:
        info = mt5.account_info()
        return float(info.equity) if info else 0.0
    except Exception:
        return 0.0


def apply_allocation_to_mt5(
    allocation: List[Dict[str, object]],
    *,
    volume_min_floor: float = 0.01,
    price_cache: Optional[Dict[str, float]] = None,
    terminal_path: Optional[str] = None,
    login: Optional[int] = None,
    password: Optional[str] = None,
    server: Optional[str] = None,
) -> Dict[str, float]:
    """Mirror allocation weights into MT5 as approximate net positions.

    We compute volume lots so that notional ≈ weight * equity using contract_size and price.
    Returns dict with {symbol: target_volume_lots} for visibility.
    """
    mt5 = _lazy_import()
    if not mt5:
        raise RuntimeError("MetaTrader5 module not available")
    if not init(login=login, password=password, server=server, terminal_path=terminal_path):
        raise RuntimeError("Failed to initialize/login to MT5")

    eq = max(1e-6, account_equity(mt5))
    symmap = _load_symbol_map()
    targets: Dict[str, float] = {}

    for entry in allocation:
        sym_b = str(entry.get("symbol", ""))
        w = float(entry.get("weight", 0.0))
        if w <= 0.0 or not sym_b:
            continue
        sym_mt5 = map_symbol(sym_b, symmap=symmap)
        if not _ensure_symbol(mt5, sym_mt5):
            continue
        info = mt5.symbol_info(sym_mt5)
        if info is None:
            continue
        tick = mt5.symbol_info_tick(sym_mt5)
        price = float(getattr(tick, "last", 0.0) or getattr(tick, "bid", 0.0) or getattr(tick, "ask", 0.0) or 0.0)
        if price <= 0.0:
            continue
        contract = float(getattr(info, "trade_contract_size", 1.0) or 1.0)
        vmin = float(getattr(info, "volume_min", volume_min_floor) or volume_min_floor)
        vstep = float(getattr(info, "volume_step", volume_min_floor) or volume_min_floor)
        vmax = float(getattr(info, "volume_max", 100.0) or 100.0)
        target_notional = w * eq
        vol = target_notional / max(1e-9, (contract * price))
        vol = min(max(_round_step(vol, vstep), vmin), vmax)
        targets[sym_mt5] = vol

    # Compute current -> target deltas and send market orders
    current = positions_by_symbol(mt5)
    for sym, tgt in targets.items():
        cur = current.get(sym, 0.0)
        delta = tgt - cur
        if abs(delta) < 1e-9:
            continue
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": sym,
            "volume": abs(delta),
            "type": mt5.ORDER_TYPE_BUY if delta > 0 else mt5.ORDER_TYPE_SELL,
            "deviation": 20,
            "magic": 987654321,
            "comment": "OpenQuant",
        }
        # Request price at send time
        tick = mt5.symbol_info_tick(sym)
        expected_price = 0.0
        if tick:
            expected_price = float(getattr(tick, "ask", 0.0) if delta > 0 else getattr(tick, "bid", 0.0))
            if expected_price > 0:
                req["price"] = expected_price
        
        # Add SL/TP if provided in allocation
        # Entry format: {"symbol": "...", "weight": ..., "sl": 1.2345, "tp": 1.2445}
        sl = float(entry.get("sl", 0.0))
        tp = float(entry.get("tp", 0.0))
        if sl > 0:
            req["sl"] = sl
        if tp > 0:
            req["tp"] = tp

        # Send order and check result
        res = mt5.order_send(req)
        
        # Alert on failure
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            code = res.retcode if res else "None"
            comment = res.comment if res else "No result"
            from ..utils.alerts import send_alert
            send_alert(
                subject=f"MT5 Order Failed: {sym}",
                body=f"Order {req['type']} {req['volume']} failed. Code: {code}, Comment: {comment}",
                severity="ERROR"
            )
            continue

        # Alert on high slippage (if we had an expected price)
        if expected_price > 0 and res.price > 0:
            slippage_pct = abs(res.price - expected_price) / expected_price
            # Threshold: 0.1% (10 bps)
            if slippage_pct > 0.001:
                from ..utils.alerts import send_alert
                send_alert(
                    subject=f"MT5 High Slippage: {sym}",
                    body=f"Slippage {slippage_pct:.4%}. Expected {expected_price}, Got {res.price}",
                    severity="WARNING"
                )

        time.sleep(0.1)

    # Export signals for visualization (auto-detects MT5 path)
    try:
        # Convert targets dict back to list format for export if needed, 
        # but we have the original 'allocation' list which is better.
        # We might want to annotate it with actual fills? 
        # For now, just export the target allocation signals as requested.
        export_signals_to_csv(allocation)
    except Exception as e:
        print(f"Signal export failed: {e}")

    return targets


def modify_position(symbol: str, sl: Optional[float] = None, tp: Optional[float] = None) -> bool:
    """Modify SL/TP of an existing position for the given symbol.
    
    Finds the position ticket for the symbol and sends a TRADE_ACTION_SLTP request.
    """
    mt5 = _lazy_import()
    if not mt5:
        return False
        
    # Find position ticket
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return False
        
    # Assume one net position per symbol (hedging mode not supported here yet)
    pos = positions[0]
    ticket = pos.ticket
    
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "symbol": symbol,
    }
    
    # If SL/TP not provided, keep existing
    if sl is not None:
        req["sl"] = float(sl)
    else:
        req["sl"] = pos.sl
        
    if tp is not None:
        req["tp"] = float(tp)
    else:
        req["tp"] = pos.tp
        
    res = mt5.order_send(req)
    
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        code = res.retcode if res else "None"
        comment = res.comment if res else "No result"
        from ..utils.alerts import send_alert
        send_alert(
            subject=f"MT5 Modify Failed: {symbol}",
            body=f"Modify SL/TP failed. Code: {code}, Comment: {comment}",
            severity="ERROR"
        )
        return False
        
    return True


def export_signals_to_csv(allocations: List[Dict[str, Any]], path: str = "data/signals.csv", mt5_data_path: Optional[str] = None) -> None:
    """Export signals to CSV for MT5 EA visualization.
    Format: Symbol,Side,Weight,Timestamp
    
    Writes to:
    1. `path` (default: data/signals.csv)
    2. `[MT5_DATA_PATH]/MQL5/Files/signals.csv` (if MT5 is available or mt5_data_path provided)
    """
    import csv
    from datetime import datetime
    import shutil
    
    # 1. Write to local data/signals.csv
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare rows
    rows = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for alloc in allocations:
        sym = str(alloc.get("symbol", ""))
        weight = float(alloc.get("weight", 0.0))
        
        # Determine side based on weight sign
        side = "BUY" if weight > 0 else "SELL"
        if weight == 0: side = "FLAT"
        
        if sym:
            rows.append([sym, side, f"{weight:.4f}", ts])

    # Write local
    file_exists = p.exists()
    try:
        with open(p, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Symbol", "Side", "Weight", "Timestamp"])
            writer.writerows(rows)
    except Exception as e:
        print(f"Failed to write local signals CSV: {e}")

    # 2. Write to MT5 MQL5/Files directory
    target_dir = None
    if mt5_data_path:
        target_dir = Path(mt5_data_path) / "MQL5" / "Files"
    else:
        # Try to auto-detect via MT5 API
        mt5 = _lazy_import()
        if mt5:
            try:
                info = mt5.terminal_info()
                if info and info.data_path:
                    target_dir = Path(info.data_path) / "MQL5" / "Files"
            except Exception:
                pass
    
    if target_dir:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / "signals.csv"
            
            # We overwrite or append? The EA reads the file. 
            # Usually for visualization of *latest* signals, we might want to append or just keep latest.
            # The EA reads line by line. Let's append to match local behavior, 
            # but maybe we should truncate if it gets too big? 
            # For now, append is safer for history.
            
            target_exists = target_file.exists()
            with open(target_file, "a", newline="") as f:
                writer = csv.writer(f)
                if not target_exists:
                    writer.writerow(["Symbol", "Side", "Weight", "Timestamp"])
                writer.writerows(rows)
                
            # print(f"Exported signals to MT5: {target_file}")
        except Exception as e:
            print(f"Failed to write MT5 signals CSV: {e}")


def close_all_positions() -> int:
    """Emergency: Close ALL open positions immediately.
    Returns number of closed positions.
    """
    mt5 = _lazy_import()
    if not mt5:
        return 0
        
    count = 0
    try:
        positions = mt5.positions_get()
        if not positions:
            return 0
            
        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            if not tick:
                continue
                
            # Close logic: Send opposite order
            type_op = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if type_op == mt5.ORDER_TYPE_SELL else tick.ask
            
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": type_op,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": 987654321,
                "comment": "Emergency Close",
            }
            
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                count += 1
                
    except Exception:
        pass
        
    return count

