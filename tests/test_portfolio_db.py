from __future__ import annotations
from pathlib import Path
from datetime import datetime
import duckdb  # type: ignore

from openquant.paper.state import PortfolioState
from openquant.paper.simulator import MarketSnapshot
from openquant.storage.portfolio_db import connect, record_rebalance, OrderFill


def test_record_rebalance_inserts_rows(tmp_path: Path):
    dbp = tmp_path / "p.duckdb"
    con = connect(dbp)
    try:
        # state with one position
        key = ("BINANCE","BTC/USDT","1h","ema")
        state = PortfolioState(cash=1000.0, holdings={key: 1.0})
        snap = MarketSnapshot(prices={key: 100.0})
        # one SELL trade of 0.5 units
        fill = OrderFill(key=key, side="SELL", delta_units=-0.5, exec_price=100.0, notional=50.0, fee_bps=1.0, slippage_bps=0.0, fee_paid=0.05)
        record_rebalance(con, ts=datetime.utcnow(), fills=[fill], state=state, snap=snap)
        # verify rows
        t = con.execute("SELECT COUNT(*) FROM portfolio_trades").fetchone()[0]
        p = con.execute("SELECT COUNT(*) FROM portfolio_positions").fetchone()[0]
        e = con.execute("SELECT COUNT(*) FROM portfolio_equity").fetchone()[0]
        assert t >= 1 and p >= 1 and e >= 1
    finally:
        con.close()

