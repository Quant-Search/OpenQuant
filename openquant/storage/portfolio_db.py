from __future__ import annotations
"""DuckDB-backed portfolio ledger (trades, positions, equity) for paper trading.

By default we store the portfolio tables in the same DB as results (data/results.duckdb)
so the Streamlit dashboard can read both.
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
from pathlib import Path
import duckdb  # type: ignore
from datetime import datetime

from openquant.paper.state import PortfolioState, Key
from openquant.paper.simulator import MarketSnapshot


def connect(db_path: str | Path = "data/results.duckdb"):
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(p))
    ensure_schema(con)
    return con


def ensure_schema(con) -> None:  # type: ignore
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_trades (
            ts TIMESTAMP,
            exchange VARCHAR,
            symbol VARCHAR,
            timeframe VARCHAR,
            strategy VARCHAR,
            side VARCHAR,
            delta_units DOUBLE,
            price DOUBLE,
            notional DOUBLE,
            fee_bps DOUBLE,
            slippage_bps DOUBLE,
            fee_paid DOUBLE,
            note VARCHAR
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_positions (
            ts TIMESTAMP,
            exchange VARCHAR,
            symbol VARCHAR,
            timeframe VARCHAR,
            strategy VARCHAR,
            units DOUBLE
        );
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_equity (
            ts TIMESTAMP,
            equity DOUBLE,
            cash DOUBLE
        );
        """
    )


@dataclass
class OrderFill:
    key: Key
    side: str
    delta_units: float
    exec_price: float
    notional: float
    fee_bps: float
    slippage_bps: float
    fee_paid: float
    note: str = ""


def record_rebalance(
    con,
    *,
    ts: datetime,
    fills: Iterable[OrderFill],
    state: PortfolioState,
    snap: MarketSnapshot,
) -> None:
    """Record a rebalance into trades, positions, and equity tables."""
    # trades
    for f in fills:
        ex, sym, tf, strat = f.key
        con.execute(
            """
            INSERT INTO portfolio_trades
            (ts, exchange, symbol, timeframe, strategy, side, delta_units, price, notional, fee_bps, slippage_bps, fee_paid, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts, ex, sym, tf, strat, f.side, float(f.delta_units), float(f.exec_price), float(f.notional),
                float(f.fee_bps), float(f.slippage_bps), float(f.fee_paid), f.note,
            ),
        )
    # positions snapshot
    for k, u in state.holdings.items():
        ex, sym, tf, strat = k
        con.execute(
            """
            INSERT INTO portfolio_positions (ts, exchange, symbol, timeframe, strategy, units)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ts, ex, sym, tf, strat, float(u)),
        )
    # equity snapshot
    # Compute equity using provided snapshot
    equity = float(state.cash)
    for k, u in state.holdings.items():
        price = float(snap.prices.get(k, 0.0))
        equity += float(u) * price
    con.execute(
        """
        INSERT INTO portfolio_equity (ts, equity, cash) VALUES (?, ?, ?)
        """,
        (ts, equity, float(state.cash)),
    )

