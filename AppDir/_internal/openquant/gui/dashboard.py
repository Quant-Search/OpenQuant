"""Streamlit dashboard: Robot Control Center.
Run: streamlit run openquant/gui/dashboard.py
"""
from __future__ import annotations
import os
from pathlib import Path
import json
import sys
import time
import threading
from datetime import datetime

# Ensure repository root is on sys.path
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

from openquant.gui.scheduler import SCHEDULER
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)
DB_PATH = Path(os.environ.get("OPENQUANT_RESULTS_DB", "data/results.duckdb"))

# --- Helper Functions ---

@st.cache_resource
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(database=str(db_path))

def _query(con, sql: str, params: tuple = ()):
    return con.execute(sql, params).df()

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

# --- Page Views ---

def view_dashboard(con):
    st.header("📊 Research Dashboard")
    
    # Load Runs
    _ensure_schema(con)
    runs = _query(con, "SELECT DISTINCT run_id, MIN(ts) as ts FROM results GROUP BY run_id ORDER BY ts DESC")
    
    if runs.empty:
        st.info("No research runs found. Go to 'Robot Control' to start one.")
        return

    run_id = st.selectbox("Select Run ID", runs["run_id"].tolist(), index=0)
    df = _query(con, "SELECT * FROM results WHERE run_id = ?", (run_id,))

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        sym = st.selectbox("Symbol", ["(all)"] + sorted(df["symbol"].unique().tolist()))
    with col2:
        strat = st.selectbox("Strategy", ["(all)"] + sorted(df["strategy"].unique().tolist()))
    with col3:
        ok_only = st.checkbox("Show Only OK", value=True)

    q = df.copy()
    if sym != "(all)": q = q[q.symbol == sym]
    if strat != "(all)": q = q[q.strategy == strat]
    if ok_only: q = q[q.ok]

    # Metrics
    st.subheader("Top Results")
    if not q.empty:
        sort_cols = [c for c in ["wfo_mts", "dsr", "sharpe"] if c in q.columns]
        q = q.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        st.dataframe(q.head(50), use_container_width=True)
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(q, x="sharpe", nbins=20, title="Sharpe Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "max_dd" in q.columns:
                fig2 = px.scatter(q, x="max_dd", y="sharpe", color="symbol", title="Risk vs Reward")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No results match filters.")

    # Portfolio Ledger
    st.divider()
    st.subheader("📜 Paper Portfolio Ledger")
    try:
        eq = _query(con, "SELECT * FROM portfolio_equity ORDER BY ts DESC LIMIT 1")
        if not eq.empty:
            st.metric("Current Equity", f"${eq['equity'].iloc[0]:,.2f}", delta=None)
        
        pos = _query(con, "SELECT * FROM portfolio_positions WHERE ts=(SELECT MAX(ts) FROM portfolio_positions) ORDER BY symbol")
        if not pos.empty:
            st.caption("Current Positions")
            st.dataframe(pos, use_container_width=True)
    except Exception:
        st.caption("No portfolio history yet.")

def view_robot_control():
    st.header("🤖 Robot Control Center")
    
    # Status Panel
    st.subheader("Status")
    c1, c2, c3 = st.columns(3)
    
    status_color = "green" if SCHEDULER.is_running else "red"
    c1.markdown(f"**State**: :{status_color}[{SCHEDULER.status_message}]")
    c2.metric("Last Run", SCHEDULER.last_run_time.strftime("%H:%M:%S") if SCHEDULER.last_run_time else "Never")
    c3.metric("Next Run", SCHEDULER.next_run_time.strftime("%H:%M:%S") if SCHEDULER.next_run_time else "Manual Only")

    if SCHEDULER.error_message:
        st.error(f"Last Error: {SCHEDULER.error_message}")

    st.divider()

    # Auto-Pilot Controls
    st.subheader("Auto-Pilot")
    
    # Load config from session or defaults
    if "robot_config" not in st.session_state:
        st.session_state.robot_config = {
            "interval": 60,
            "top_n": 10,
            "use_mt5": False,
            "mt5_path": os.environ.get("OQ_MT5_TERMINAL", ""),
            "mt5_login": os.environ.get("OQ_MT5_LOGIN", ""),
            "mt5_pass": os.environ.get("OQ_MT5_PASSWORD", ""),
            "mt5_server": os.environ.get("OQ_MT5_SERVER", "")
        }
    
    cfg = st.session_state.robot_config
    
    with st.expander("Configuration", expanded=not SCHEDULER.is_running):
        c_int, c_top = st.columns(2)
        cfg["interval"] = c_int.number_input("Interval (minutes)", min_value=1, value=cfg["interval"], disabled=SCHEDULER.is_running)
        cfg["top_n"] = c_top.number_input("Universe Size (Top N)", min_value=1, value=cfg["top_n"], disabled=SCHEDULER.is_running)
        
        cfg["use_mt5"] = st.checkbox("Enable MT5 Live Trading", value=cfg["use_mt5"], disabled=SCHEDULER.is_running)
        if cfg["use_mt5"]:
            cfg["mt5_path"] = st.text_input("MT5 Path", value=cfg["mt5_path"], disabled=SCHEDULER.is_running)
            c_l, c_p, c_s = st.columns(3)
            cfg["mt5_login"] = c_l.text_input("Login", value=cfg["mt5_login"], disabled=SCHEDULER.is_running)
            cfg["mt5_pass"] = c_p.text_input("Password", value=cfg["mt5_pass"], type="password", disabled=SCHEDULER.is_running)
            cfg["mt5_server"] = c_s.text_input("Server", value=cfg["mt5_server"], disabled=SCHEDULER.is_running)

    # Actions
    col_start, col_stop, col_once = st.columns(3)
    
    if col_start.button("🚀 Start Auto-Pilot", disabled=SCHEDULER.is_running, type="primary"):
        run_cfg = {
            "top_n": cfg["top_n"],
            "use_mt5": cfg["use_mt5"],
            "mt5_creds": {
                "path": cfg["mt5_path"],
                "login": cfg["mt5_login"],
                "password": cfg["mt5_pass"],
                "server": cfg["mt5_server"]
            }
        }
        SCHEDULER.start(interval_minutes=cfg["interval"], config=run_cfg)
        st.rerun()

    if col_stop.button("🛑 Stop Auto-Pilot", disabled=not SCHEDULER.is_running):
        SCHEDULER.stop()
        st.rerun()

    if col_once.button("⚡ Run Once Now", disabled=SCHEDULER.is_running):
        with st.spinner("Running single cycle..."):
            # Use scheduler's logic but synchronously
            SCHEDULER.run_config.update({
                "top_n": cfg["top_n"],
                "use_mt5": cfg["use_mt5"],
                "mt5_creds": {
                    "path": cfg["mt5_path"],
                    "login": cfg["mt5_login"],
                    "password": cfg["mt5_pass"],
                    "server": cfg["mt5_server"]
                }
            })
            try:
                SCHEDULER._run_cycle()
                SCHEDULER.last_run_time = datetime.now()
                st.success("Cycle completed successfully!")
            except Exception as e:
                st.error(f"Cycle failed: {e}")

    # Logs (Tail)
    st.subheader("Live Logs")
    # Simple log tailing from file
    log_file = Path("openquant.log")
    if log_file.exists():
        with open(log_file, "r") as f:
            lines = f.readlines()
            st.text_area("Log Output", "".join(lines[-20:]), height=300)
    else:
        st.caption("No log file found yet.")
        
    # Auto-refresh for logs/status
    if SCHEDULER.is_running:
        time.sleep(2)
        st.rerun()

def view_settings():
    st.header("⚙️ Settings")
    
    # Risk Configuration
    st.subheader("Risk Management")
    if "risk_config" not in st.session_state:
        st.session_state.risk_config = {
            "dd_limit": 0.20,
            "daily_loss_cap": 0.05,
            "cvar_limit": 0.08,
            "max_exposure_per_symbol": 0.20
        }
    
    rc = st.session_state.risk_config
    
    with st.form("risk_form"):
        c1, c2 = st.columns(2)
        rc["dd_limit"] = c1.number_input("Max Drawdown Limit (0.2 = 20%)", 0.01, 1.0, rc["dd_limit"], 0.01)
        rc["daily_loss_cap"] = c2.number_input("Daily Loss Cap (0.05 = 5%)", 0.01, 1.0, rc["daily_loss_cap"], 0.01)
        rc["cvar_limit"] = c1.number_input("CVaR Limit (95%)", 0.01, 1.0, rc["cvar_limit"], 0.01)
        rc["max_exposure_per_symbol"] = c2.number_input("Max Exposure per Symbol", 0.01, 1.0, rc["max_exposure_per_symbol"], 0.01)
        
        if st.form_submit_button("Save Risk Settings"):
            # In a real app, save to a config file. For now, we update session state and notify scheduler.
            st.success("Risk settings updated (Session only)")
            # Pass these to scheduler if running
            SCHEDULER.run_config.update(rc)

    st.divider()
    st.subheader("Environment Variables")
    with st.expander("View Environment"):
        st.json(dict(os.environ))

# --- Main App ---

def main():
    st.set_page_config(page_title="OpenQuant Robot", layout="wide", page_icon="🤖")
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("OpenQuant")
        page = st.radio("Navigation", ["Dashboard", "Robot Control", "Settings"])
        st.divider()
        st.caption(f"OS: {sys.platform}")
        st.caption(f"Time: {datetime.now().strftime('%H:%M')}")

    con = _connect(DB_PATH)
    
    if page == "Dashboard":
        view_dashboard(con)
    elif page == "Robot Control":
        view_robot_control()
    elif page == "Settings":
        view_settings()

if __name__ == "__main__":
    main()

