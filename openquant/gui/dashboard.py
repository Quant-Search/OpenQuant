"""Streamlit dashboard: view research results and status.
Run: streamlit run openquant/gui/dashboard.py
"""
from __future__ import annotations
import os
from pathlib import Path
import json
import sys
# Ensure repository root is on sys.path when launched via Streamlit from subdir
try:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
except Exception:
    pass

import duckdb  # type: ignore
import pandas as pd
import streamlit as st  # type: ignore
import plotly.express as px  # type: ignore

DB_PATH = Path(os.environ.get("OPENQUANT_RESULTS_DB", "data/results.duckdb"))


@st.cache_resource
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(database=str(db_path))


def _query(con, sql: str, params: tuple = ()):  # type: ignore
    return con.execute(sql, params).df()


def load_runs(con):
    _ensure_schema(con)
    return _query(con, "SELECT DISTINCT run_id, MIN(ts) as ts FROM results GROUP BY run_id ORDER BY ts DESC")


def _ensure_schema(con):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            ts TIMESTAMP,
            run_id VARCHAR,
            exchange VARCHAR,
            symbol VARCHAR,
            timeframe VARCHAR,
            strategy VARCHAR,
            params JSON,
            sharpe DOUBLE,
            dsr DOUBLE,
            max_dd DOUBLE,
            cvar DOUBLE,
            n_trades INTEGER,
            bars INTEGER,
            ok BOOLEAN,
            wfo_mts DOUBLE
        );
        """
    )


def main():
    st.set_page_config(page_title="OpenQuant Dashboard", layout="wide")
    st.title("OpenQuant Research Dashboard")
    con = _connect(DB_PATH)

    runs = load_runs(con)
    if runs.empty:
        st.info("No runs yet. Execute scripts/run_universe_research.py to populate results.")
        return

    run_id = st.selectbox("Run ID", runs["run_id"].tolist(), index=0)
    df = _query(con, "SELECT * FROM results WHERE run_id = ?", (run_id,))

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        symbols = ["(all)"] + sorted(df["symbol"].unique().tolist())
        strategies = ["(all)"] + sorted(df["strategy"].unique().tolist())
        timeframes = ["(all)"] + sorted(df["timeframe"].unique().tolist())
        sym = st.selectbox("Symbol", symbols)
        strat = st.selectbox("Strategy", strategies)
        tf = st.selectbox("Timeframe", timeframes)
        ok_only = st.checkbox("Only OK rows", value=True)

    q = df.copy()
    if sym != "(all)":
        q = q[q.symbol == sym]
    if strat != "(all)":
        q = q[q.strategy == strat]
    if tf != "(all)":
        q = q[q.timeframe == tf]
    if ok_only:
        q = q[q.ok]

    st.subheader("Top results")
    if not q.empty:
        # Prefer WFO mean test Sharpe if present, then DSR, then Sharpe
        sort_cols = [c for c in ["wfo_mts", "dsr", "sharpe"] if c in q.columns]
        q = q.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        cols = [c for c in ["exchange","symbol","timeframe","strategy","sharpe","dsr","wfo_mts","max_dd","cvar","n_trades","bars","params","ok"] if c in q.columns]
        st.dataframe(q[cols].head(100))
        fig = px.histogram(q, x="sharpe", nbins=40, title="Sharpe distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No rows after filters")

    # Paper Portfolio section
    st.subheader("Paper Portfolio (Ledger)")
    try:
        eq = _query(con, "SELECT * FROM portfolio_equity ORDER BY ts DESC LIMIT 1")
        if not eq.empty:
            st.metric("Equity", f"{eq['equity'].iloc[0]:.2f}", help=f"Cash: {eq['cash'].iloc[0]:.2f}")
        pos = _query(con, "SELECT * FROM portfolio_positions WHERE ts=(SELECT MAX(ts) FROM portfolio_positions) ORDER BY exchange,symbol,timeframe,strategy")
        if not pos.empty:
            st.dataframe(pos, use_container_width=True)
        tr = _query(con, "SELECT * FROM portfolio_trades ORDER BY ts DESC LIMIT 50")
        if not tr.empty:
            st.write("Recent trades")
            st.dataframe(tr, use_container_width=True)
        if eq.empty and pos.empty and tr.empty:  # type: ignore[attr-defined]
            st.info("No portfolio ledger yet. Run scripts/paper_apply_allocation.py to populate.")
    except Exception as e:
        st.info("Portfolio tables not found yet. They will appear after the first paper execution.")

    # Robot Control (no CLI needed)
    st.subheader("Robot Control")
    from openquant.research.universe_runner import run_universe  # lazy import
    from openquant.paper.io import load_state, save_state
    from openquant.paper.simulator import MarketSnapshot, compute_rebalance_orders, execute_orders
    from openquant.storage.portfolio_db import connect as _pdb_connect, record_rebalance, OrderFill
    from openquant.paper.mt5_bridge import apply_allocation_to_mt5, is_available as mt5_available
    import ccxt  # type: ignore
    import json

    col1, col2 = st.columns(2)
    with col1:
        top_n = st.number_input("Research top_n", min_value=1, max_value=200, value=5)
        fee_bps = st.number_input("Fee (bps)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)
        slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, max_value=200.0, value=5.0, step=0.5)
    with col2:
        st.write("MT5 Settings")
        mt5_path = st.text_input("Terminal path (terminal64.exe)", value=os.environ.get("OQ_MT5_TERMINAL", ""))
        mt5_server = st.text_input("Server", value=os.environ.get("OQ_MT5_SERVER", ""))
        mt5_login = st.text_input("Login (int)", value=os.environ.get("OQ_MT5_LOGIN", ""))
        mt5_password = st.text_input("Password", type="password", value=os.environ.get("OQ_MT5_PASSWORD", ""))
        st.caption(f"MT5 Python available: {mt5_available()}")
    use_mt5 = st.checkbox("Use MT5 FX mode (research FX symbols in your MT5)", value=True)

    def _latest_alloc_path() -> Path | None:
        r = Path("reports")
        files = sorted(r.glob("allocation_*.json"))
        return files[-1] if files else None

    def _load_alloc(p: Path):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "allocations" in data:
            return data["allocations"]
        if isinstance(data, list):
            return data
        raise ValueError("Allocation JSON must be a list or a dict with 'allocations'")

    def _build_snap(alloc):
        prices = {}
        clients = {}
        # Try to initialize MT5 if any allocation entries are for MT5
        try_mt5 = any(str(e.get("exchange","")) .lower() == "mt5" for e in alloc)
        mt5 = None
        if try_mt5:
            try:
                import MetaTrader5 as _mt5  # type: ignore
                mt5 = _mt5
                if mt5_path:
                    mt5.initialize(path=mt5_path)
                else:
                    mt5.initialize()
                if mt5_login.strip() and mt5_password and mt5_server:
                    try:
                        mt5.login(int(mt5_login), password=mt5_password, server=mt5_server)
                    except Exception:
                        pass
            except Exception:
                mt5 = None
        for e in alloc:
            ex = str(e.get("exchange","binance")).lower()
            sym = str(e.get("symbol",""))
            tf = str(e.get("timeframe",""))
            strat = str(e.get("strategy",""))
            if not sym:
                continue
            px = 0.0
            if ex == "mt5" and mt5 is not None:
                try:
                    tick = mt5.symbol_info_tick(sym)
                    px = float(getattr(tick, "last", 0.0) or getattr(tick, "bid", 0.0) or getattr(tick, "ask", 0.0) or 0.0)
                except Exception:
                    px = 0.0
            else:
                if ex not in clients:
                    clients[ex] = ccxt.binance() if ex == "binance" else ccxt.binance()
                cli = clients[ex]
                try:
                    t = cli.fetch_ticker(sym)
                    px = float(t.get("last") or t.get("close") or 0.0)
                except Exception:
                    px = 0.0
            prices[(ex.upper(), sym, tf, strat)] = px
        return MarketSnapshot(prices=prices)

    if st.button("Run research now and apply allocation (paper + MT5)"):
        with st.spinner("Running research..."):
            if use_mt5:
                try:
                    from openquant.data import mt5_source as _mt5s
                    login_int = int(mt5_login) if mt5_login.strip() else None
                    _mt5s.configure(terminal_path=(mt5_path or None), login=login_int, password=(mt5_password or None), server=(mt5_server or None))
                except Exception as e:
                    st.warning(f"MT5 configure failed: {e}")
                out = run_universe(exchange="mt5", top_n=int(top_n))
            else:
                out = run_universe(top_n=int(top_n))
        alloc_p = _latest_alloc_path()
        if not alloc_p:
            st.error("No allocation JSON found.")
        else:
            alloc = _load_alloc(alloc_p)
            # Paper apply + ledger
            state = load_state(Path("data")/"paper_state.json")
            snap = _build_snap(alloc)
            targets = []
            for e in alloc:
                key = (str(e.get("exchange","binance")).upper(), str(e.get("symbol","")), str(e.get("timeframe","")), str(e.get("strategy","")))
                w = float(e.get("weight",0.0))
                if w>0:
                    targets.append((key,w))
            orders = compute_rebalance_orders(state, targets, snap)
            summary, fills_raw = execute_orders(state, orders, fee_bps=float(fee_bps), slippage_bps=float(slippage_bps))
            con = _pdb_connect(DB_PATH)
            from datetime import datetime as _dt
            ts_dt = _dt.utcnow()
            try:
                fills = []
                for (k, du, ex_px, fee_paid) in fills_raw:
                    fills.append(OrderFill(key=k, side=("BUY" if du>0 else "SELL"), delta_units=float(du), exec_price=float(ex_px), notional=float(abs(du)*ex_px), fee_bps=float(fee_bps), slippage_bps=float(slippage_bps), fee_paid=float(fee_paid)))
                record_rebalance(con, ts=ts_dt, fills=fills, state=state, snap=snap)
            finally:
                con.close()
            save_state(state, Path("data")/"paper_state.json")
            st.success(f"Paper applied: orders={summary['orders']}, turnover={summary['turnover']:.2f}")
            # MT5 apply
            try:
                login_int = int(mt5_login) if mt5_login.strip() else None
            except Exception:
                login_int = None
            try:
                targets_mt5 = apply_allocation_to_mt5(alloc, terminal_path=(mt5_path or None), login=login_int, password=(mt5_password or None), server=(mt5_server or None))
                st.success(f"MT5 applied: {targets_mt5}")
            except Exception as e:
                st.warning(f"MT5 apply failed or unavailable: {e}")


if __name__ == "__main__":
    main()

