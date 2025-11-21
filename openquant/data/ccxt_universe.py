"""Universe discovery utilities for ccxt exchanges (e.g., Binance)."""
from __future__ import annotations
from typing import List, Dict

import ccxt  # type: ignore

from .ccxt_source import _exchange
from ..utils.logging import get_logger
from ..utils.retry import retry_call
from ..utils.rate_limit import get_rate_limiter

LOGGER = get_logger(__name__)


def discover_symbols(exchange: str = "binance", top_n: int = 30) -> List[str]:
    """Discover active markets (Crypto, Stocks, Forex) and rank by volume/liquidity.
    Supports:
    - Crypto: Binance, Kraken, Coinbase (USDT/USD pairs)
    - Stocks: Alpaca (SP500/Nasdaq top liquid)
    - Forex: Oanda (Majors)
    """
    exchange = exchange.lower()
    
    # --- Alpaca (Stocks) ---
    if exchange == "alpaca":
        # Return a curated list of liquid stocks for now
        # In future, fetch from Alpaca API assets endpoint
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", "NFLX", "INTC",
            "SPY", "QQQ", "IWM", "GLD", "SLV", "USO", "TLT", "HYG", "LQD", "JPM"
        ][:top_n]

    # --- Oanda (Forex) ---
    if exchange == "oanda":
        return [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD",
            "EUR/GBP", "EUR/JPY", "GBP/JPY"
        ][:top_n]

    # --- Crypto (CCXT) ---
    ex = _exchange(exchange)
    limiter = get_rate_limiter(exchange, rate_per_sec=6.0, capacity=6)

    # Load markets to filter spot/active/USDT
    limiter.acquire()
    markets = retry_call(lambda: ex.load_markets(), retries=3, base_delay=0.5)
    
    # Filter logic
    symbols = []
    for m, info in markets.items():
        if not info.get("active"): continue
        if not info.get("spot"): continue
        
        # Quote currency filter
        quote = info.get("quote")
        if quote in ["USDT", "USD", "USDC"]:
            symbols.append(m)

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

