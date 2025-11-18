"""MT5 data source and symbol discovery.

Safe, optional integration:
- This module does not require MetaTrader5 to import.
- All MT5 calls are guarded; functions raise RuntimeError if MT5 is unavailable
  or not initialized with credentials.

Usage:
    from openquant.data import mt5_source as mt5s
    mt5s.configure(terminal_path="C:/Program Files/MetaTrader 5/terminal64.exe",
                   login=12345678, password="***", server="MetaQuotes-Demo")
    df = mt5s.fetch_ohlcv("EURUSD", timeframe="1h")
"""
from __future__ import annotations
from typing import Optional, Dict, List
from datetime import datetime, timezone
import pandas as pd

_MT5 = None  # lazy import handle
_CFG: Dict[str, object] = {
    "terminal_path": None,
    "login": None,
    "password": None,
    "server": None,
}
_INITIALIZED = False


def is_available() -> bool:
    """Return True if MetaTrader5 module can be imported."""
    global _MT5
    if _MT5 is not None:
        return True
    try:
        import MetaTrader5 as mt5  # type: ignore
        _ = mt5.SELECT_BY_SYMBOL  # touch an attribute
        return True
    except Exception:
        return False


def configure(*, terminal_path: Optional[str] = None,
              login: Optional[int] = None,
              password: Optional[str] = None,
              server: Optional[str] = None) -> None:
    """Store connection parameters for lazy initialization."""
    _CFG["terminal_path"] = terminal_path
    _CFG["login"] = login
    _CFG["password"] = password
    _CFG["server"] = server


def _ensure_init() -> object:
    """Initialize and login to MT5 once using stored credentials.
    Returns the mt5 module object.
    """
    global _MT5, _INITIALIZED
    if _INITIALIZED and _MT5 is not None:
        return _MT5
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError("MetaTrader5 module not available") from e
    ok = False
    last_err = None
    try:
        if _CFG.get("terminal_path"):
            ok = mt5.initialize(path=str(_CFG["terminal_path"]))
            if not ok and hasattr(mt5, "last_error"):
                try:
                    last_err = mt5.last_error()
                except Exception:
                    last_err = None
        if not ok:
            # Fallback: try attaching to any already running terminal
            ok = mt5.initialize()
            if not ok and last_err is None and hasattr(mt5, "last_error"):
                try:
                    last_err = mt5.last_error()
                except Exception:
                    last_err = None
    except Exception:
        ok = False
    if not ok:
        raise RuntimeError(f"Failed to initialize MT5 (path={_CFG.get('terminal_path')}) last_error={last_err}")
    try:
        if _CFG.get("login") and _CFG.get("password") and _CFG.get("server"):
            ok_login = mt5.login(int(_CFG["login"]), password=str(_CFG["password"]), server=str(_CFG["server"]))
            if not ok_login:
                raise RuntimeError("MT5 login failed; check server/login/password")
    except Exception as e:
        # If login not provided, assume already logged-in terminal (ok)
        if _CFG.get("login") or _CFG.get("password") or _CFG.get("server"):
            raise
    _MT5 = mt5
    _INITIALIZED = True
    return _MT5


_TF_MAP = {
    "1d": "TIMEFRAME_D1",
    "4h": "TIMEFRAME_H4",
    "1h": "TIMEFRAME_H1",
    "30m": "TIMEFRAME_M30",
    "15m": "TIMEFRAME_M15",
}


def _tf_const(mt5: object, tf: str) -> int:
    key = _TF_MAP.get(tf.lower())
    if not key or not hasattr(mt5, key):
        raise ValueError(f"Unsupported timeframe for MT5: {tf}")
    return getattr(mt5, key)


def fetch_ohlcv(symbol: str,
                timeframe: str = "1h",
                since: Optional[datetime] = None,
                limit: Optional[int] = None) -> pd.DataFrame:
    """Fetch OHLCV from MT5 and return standardized DataFrame.
    Index is UTC datetime; columns: Open, High, Low, Close, Volume.
    """
    mt5 = _ensure_init()
    tf_const = _tf_const(mt5, timeframe)

    # Build rates array using either range or count
    try:
        if since is not None:
            # Ensure timezone-aware UTC
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
            end = datetime.now(timezone.utc)
            rates = mt5.copy_rates_range(symbol, tf_const, since, end)
        else:
            cnt = int(limit) if limit and int(limit) > 0 else 1000
            rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, cnt)
    except Exception as e:
        raise RuntimeError(f"MT5 copy_rates failed for {symbol} {timeframe}: {e}")

    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"]).astype(float)

    import numpy as np
    r = pd.DataFrame(rates)
    # MT5 returns 'time' in seconds since epoch
    r["time"] = pd.to_datetime(r["time"], unit="s", utc=True)
    r = r.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","tick_volume":"Volume"})
    r = r.set_index("time")[ ["Open","High","Low","Close","Volume"] ].astype(float)
    r = r.sort_index()
    r = r[~r.index.duplicated(keep="last")]
    r = r[(r[["Open","High","Low","Close","Volume"]] >= 0).all(axis=1)]
    return r


def discover_fx_symbols(top_n: int = 20) -> List[str]:
    """Discover commonly traded FX symbols available in the MT5 terminal.
    Falls back to standard majors if discovery fails.
    """
    majors = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD","XAUUSD"]
    try:
        mt5 = _ensure_init()
        infos = mt5.symbols_get()
        if not infos:
            return majors[:top_n]
        avail: List[str] = []
        for inf in infos:
            try:
                name = str(getattr(inf, "name", ""))
                if not name:
                    continue
                # Filter obvious FX majors and gold; skip symbols with suffixes that don't trade
                if name in majors or name.endswith("USD") or name.startswith("USD"):
                    sinfo = mt5.symbol_info(name)
                    if sinfo and getattr(sinfo, "trade_mode", 0) != mt5.SYMBOL_TRADE_MODE_DISABLED:
                        avail.append(name)
            except Exception:
                continue
        # Prioritize canonical majors order
        ordered = [s for s in majors if s in avail]
        # Fill with any other USD crosses
        rest = [s for s in avail if s not in ordered]
        out = ordered + rest
        if top_n and top_n > 0:
            out = out[:top_n]
        return out
    except Exception:
        return majors[:top_n]

