"""Universe discovery utilities for ccxt exchanges (e.g., Binance)."""
from __future__ import annotations
from typing import List, Dict

import ccxt  # type: ignore

from .ccxt_source import _exchange
from ..utils.logging import get_logger
from ..utils.retry import retry_call
from ..utils.rate_limit import get_rate_limiter

LOGGER = get_logger(__name__)


def discover_usdt_symbols(exchange: str = "binance", top_n: int = 30) -> List[str]:
    """Discover active spot USDT markets and rank by 24h quoteVolume when available.
    Fallback to market existence if tickers not available.
    Returns ccxt symbols like "BTC/USDT".
    """
    ex = _exchange(exchange)
    limiter = get_rate_limiter(exchange, rate_per_sec=6.0, capacity=6)

    # Load markets to filter spot/active/USDT
    limiter.acquire()
    markets = retry_call(lambda: ex.load_markets(), retries=3, base_delay=0.5)
    symbols = [m for m, info in markets.items() if info.get("active") and info.get("quote") == "USDT" and info.get("spot")]

    volumes: Dict[str, float] = {s: 0.0 for s in symbols}
    try:
        limiter.acquire()
        tickers = retry_call(lambda: ex.fetch_tickers(), retries=3, base_delay=0.5)  # may be rate-limited or unsupported
        for s in symbols:
            t = tickers.get(s)
            if not t:
                continue
            # Prefer quoteVolume; fallback to baseVolume * last
            qv = t.get("quoteVolume")
            if qv is None and "info" in t:
                qv = t["info"].get("quoteVolume")
            try:
                volumes[s] = float(qv) if qv is not None else float(t.get("baseVolume", 0.0)) * float(t.get("last", 0.0))
            except Exception:
                volumes[s] = 0.0
    except Exception as e:
        LOGGER.warning(f"fetch_tickers failed or not supported: {e}")

    # Rank symbols by volume desc
    ranked = sorted(symbols, key=lambda s: volumes.get(s, 0.0), reverse=True)
    if not ranked:
        ranked = symbols
    return ranked[: top_n]

