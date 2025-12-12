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
        sort_cols = [c for c in ["wfo_mts", "dsr", "sharpe", "sortino", "alpha_sharpe"] if c in q.columns]
        q = q.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        cols_to_show = [c for c in ["exchange","symbol","timeframe","strategy","sharpe","sortino","alpha_sharpe","bench_sharpe","win_rate","profit_factor","p_value","bull_sharpe","bear_sharpe","volatile_sharpe","calm_sharpe","mc_sharpe_p05","mc_sharpe_p95","mc_dd_p95","dsr","max_dd","cvar","n_trades","bars","wfo_mts","ok"] if c in q.columns]
        st.dataframe(q.head(50)[cols_to_show], width='stretch')
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(q, x="sharpe", nbins=20, title="Sharpe Distribution")
            st.plotly_chart(fig, width='stretch')
        with c2:
            if "max_dd" in q.columns:
                fig2 = px.scatter(q, x="max_dd", y="sortino" if "sortino" in q.columns else "sharpe", color="symbol", title="Risk vs Reward")
                st.plotly_chart(fig2, width='stretch')

        st.divider()
        st.subheader("📐 Regime Comparison")
        try:
            regime_cols = [c for c in ["bull_sharpe","bear_sharpe","volatile_sharpe","calm_sharpe"] if c in q.columns]
            if regime_cols:
                avg_regime = q[regime_cols].mean().to_dict()
                reg_df = pd.DataFrame({"Regime": list(avg_regime.keys()), "Sharpe": list(avg_regime.values())})
                figr = px.bar(reg_df, x="Regime", y="Sharpe", title="Average Regime Sharpe")
                st.plotly_chart(figr, use_container_width=True)
                if "strategy" in q.columns:
                    heat = q.groupby(["strategy"])[regime_cols].mean().reset_index()
                    heat_melt = heat.melt(id_vars=["strategy"], var_name="Regime", value_name="Sharpe")
                    fig_heat = px.density_heatmap(heat_melt, x="Regime", y="strategy", z="Sharpe", title="Strategy vs Regime Performance")
                    st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Regime metrics not available in this run.")
        except Exception as e:
            st.warning(f"Regime comparison error: {e}")
        st.divider()
        st.subheader("🧮 Allocation Preview")
        try:
            c1, c2, c3 = st.columns(3)
            total_w = c1.number_input("Max Total Exposure", 0.1, 1.0, 1.0, 0.05)
            per_sym = c2.number_input("Max Per Symbol", 0.05, 1.0, 0.20, 0.05)
            slot_w = c3.number_input("Slot Weight", 0.01, 0.50, 0.05, 0.01)
            reg_bias = st.selectbox("Regime Bias", ["(auto)", "bull", "bear", "volatile", "calm"], index=0)
            from openquant.risk.exposure import propose_portfolio_weights
            rows = []
            for _, r in q.iterrows():
                metrics = {k: r.get(k) for k in q.columns if k not in ["exchange","symbol","timeframe","strategy","params","bars","ok","run_id","ts"]}
                rows.append({"exchange": r.get("exchange"), "symbol": r.get("symbol"), "timeframe": r.get("timeframe"), "strategy": r.get("strategy"), "metrics": metrics})
            if rows:
                if reg_bias == "(auto)":
                    try:
                        rr = {"bull_sharpe": float(q.get("bull_sharpe", pd.Series()).mean() or 0), "bear_sharpe": float(q.get("bear_sharpe", pd.Series()).mean() or 0)}
                        reg_bias = "bull" if rr["bull_sharpe"] >= rr["bear_sharpe"] else "bear"
                    except Exception:
                        reg_bias = None
                alloc = propose_portfolio_weights(rows, max_total_weight=float(total_w), max_symbol_weight=float(per_sym), slot_weight=float(slot_w), regime_bias=(None if reg_bias == "(auto)" else reg_bias))
                if alloc:
                    preview = []
                    for idx, w in alloc:
                        row = rows[idx]
                        preview.append({"exchange": row["exchange"], "symbol": row["symbol"], "timeframe": row["timeframe"], "strategy": row["strategy"], "weight": float(w)})
                    st.dataframe(pd.DataFrame(preview), use_container_width=True)
                else:
                    st.info("No allocation candidates.")
            else:
                st.info("No rows to allocate.")
        except Exception as e:
            st.warning(f"Allocation preview error: {e}")
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
            st.dataframe(pos, width='stretch')
    except Exception:
        st.caption("No portfolio history yet.")

    # Alpaca Live View
    if st.session_state.get("robot_config", {}).get("use_alpaca"):
        st.divider()
        st.subheader("🦙 Alpaca Live Status")
        
        cfg = st.session_state.robot_config
        try:
            from openquant.broker.alpaca_broker import AlpacaBroker
            broker = AlpacaBroker(api_key=cfg["alpaca_key"], secret_key=cfg["alpaca_secret"], paper=cfg["alpaca_paper"])
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Equity", f"${broker.get_equity():,.2f}")
            c1.metric("Cash", f"${broker.get_cash():,.2f}")
            
            positions = broker.get_positions()
            if positions:
                st.caption("Live Positions")
                st.json(positions)
            else:
                st.info("No open positions on Alpaca.")
                
            if st.button("🚨 EMERGENCY CLOSE ALL (Alpaca)", type="primary"):
                broker.close_all_positions()
                st.error("Emergency Close Triggered! All positions liquidated.")
                time.sleep(2)
                st.rerun()
                
        except Exception as e:
            st.error(f"Alpaca Connection Failed: {e}")

        # TCA Section
        st.divider()
        st.subheader("📉 Transaction Cost Analysis (TCA)")
        try:
            from openquant.analysis.tca import TCAMonitor
            tca = TCAMonitor()
            stats = tca.get_stats()
            
            if stats.get("count", 0) > 0:
                c1, c2, c3 = st.columns(3)
                c1.metric("Orders Tracked", stats["count"])
                # Color code slippage: Green if < 1bps, Red if > 5bps
                slip = stats['avg_slippage_bps']
                delta_color = "normal"
                if slip > 5: delta_color = "inverse"
                elif slip < 1: delta_color = "off" # or normal
                
                c2.metric("Avg Slippage", f"{slip:.2f} bps", delta=f"{-slip:.2f} bps", delta_color="inverse")
                c3.metric("Total Fees", f"${stats['total_fees']:.2f}")
                
                if "recent_orders" in stats:
                    st.caption("Recent Execution Quality")
                    st.dataframe(stats["recent_orders"])
            else:
                st.info("No TCA data yet. Place trades to see slippage analysis.")
        except Exception as e:
            st.error(f"TCA Error: {e}")

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

        cfg["use_alpaca"] = st.checkbox("Enable Alpaca Trading", value=cfg.get("use_alpaca", False), disabled=SCHEDULER.is_running)
        if cfg["use_alpaca"]:
            c_k, c_s = st.columns(2)
            cfg["alpaca_key"] = c_k.text_input("Alpaca Key ID", value=cfg.get("alpaca_key", os.environ.get("APCA_API_KEY_ID", "")), disabled=SCHEDULER.is_running)
            cfg["alpaca_secret"] = c_s.text_input("Alpaca Secret Key", value=cfg.get("alpaca_secret", os.environ.get("APCA_API_SECRET_KEY", "")), type="password", disabled=SCHEDULER.is_running)
            cfg["alpaca_paper"] = st.checkbox("Paper Mode", value=cfg.get("alpaca_paper", True), disabled=SCHEDULER.is_running)

        cfg["auto_apply_actions"] = st.checkbox("Auto-Apply Optimization Actions", value=cfg.get("auto_apply_actions", True), disabled=SCHEDULER.is_running)
        cfg["apply_actions_to_live"] = st.checkbox("Apply Actions to Live Cycle Adjustments", value=cfg.get("apply_actions_to_live", True), disabled=SCHEDULER.is_running)

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
            },
            "use_alpaca": cfg.get("use_alpaca", False),
            "alpaca_key": cfg.get("alpaca_key"),
            "alpaca_secret": cfg.get("alpaca_secret"),
            "alpaca_paper": cfg.get("alpaca_paper", True),
            "auto_apply_actions": cfg.get("auto_apply_actions", True),
            "apply_actions_to_live": cfg.get("apply_actions_to_live", True)
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
                },
                "auto_apply_actions": cfg.get("auto_apply_actions", True),
                "apply_actions_to_live": cfg.get("apply_actions_to_live", True)
            })
            try:
                SCHEDULER._run_cycle()
                SCHEDULER.last_run_time = datetime.now()
                st.success("Cycle completed successfully!")
            except Exception as e:
                st.error(f"Cycle failed: {e}")

    # Logs (Tail)
    st.subheader("Live Logs")
    # Tail newest log file from logs/
    logs_dir = Path("logs")
    latest_log = None
    try:
        if logs_dir.exists():
            log_candidates = sorted(logs_dir.glob("openquant_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if log_candidates:
                latest_log = log_candidates[0]
    except Exception:
        latest_log = None
    if latest_log and latest_log.exists():
        with open(latest_log, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            st.text_area("Log Output", "".join(lines[-20:]), height=300)
            st.caption(f"Tailing: {latest_log.name}")
    else:
        st.caption("No log file found yet.")
        
    # Auto-refresh for logs/status
    if SCHEDULER.is_running:
        time.sleep(2)
        st.rerun()

def view_risk_monitor():
    """Risk monitoring and emergency controls."""
    st.header("Risk Monitor")

    # Import risk modules
    from openquant.risk.kill_switch import KILL_SWITCH
    from openquant.risk.circuit_breaker import CIRCUIT_BREAKER
    from openquant.risk.market_hours import MarketHours, MarketType

    # Status Cards
    col1, col2, col3 = st.columns(3)

    # Kill Switch Status
    with col1:
        kill_active = KILL_SWITCH.is_active()
        st.metric(
            "Kill Switch",
            "ACTIVE" if kill_active else "OK",
            delta=None,
            delta_color="inverse" if kill_active else "normal"
        )
        if kill_active:
            st.error("Trading halted. Remove data/STOP to resume.")
            if st.button("Deactivate Kill Switch"):
                try:
                    Path("data/STOP").unlink()
                    st.success("Kill switch deactivated")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            if st.button("Activate Kill Switch", type="primary"):
                Path("data").mkdir(exist_ok=True)
                Path("data/STOP").touch()
                st.warning("Kill switch activated!")
                st.rerun()

    # Circuit Breaker Status
    with col2:
        cb_status = CIRCUIT_BREAKER.get_status()
        cb_tripped = cb_status["is_tripped"]
        st.metric(
            "Circuit Breaker",
            "TRIPPED" if cb_tripped else "OK",
            delta=None,
            delta_color="inverse" if cb_tripped else "normal"
        )
        if cb_tripped:
            st.error("Trading halted due to risk limits.")
            if st.button("Reset Circuit Breaker"):
                CIRCUIT_BREAKER.reset()
                st.success("Circuit breaker reset")
                st.rerun()

    # Market Hours
    with col3:
        # Default to forex for display
        mh = MarketHours(MarketType.FOREX)
        is_open = mh.is_open()
        st.metric(
            "Forex Market",
            "OPEN" if is_open else "CLOSED",
            delta=None
        )
        if not is_open:
            next_open = mh.next_open()
            st.info(f"Opens: {next_open.strftime('%Y-%m-%d %H:%M EST')}")

    st.divider()

    # Circuit Breaker Details
    st.subheader("Circuit Breaker Thresholds")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Daily Loss Limit", f"{CIRCUIT_BREAKER.daily_loss_limit:.1%}")
    col_b.metric("Drawdown Limit", f"{CIRCUIT_BREAKER.drawdown_limit:.1%}")
    col_c.metric("Volatility Limit", f"{CIRCUIT_BREAKER.volatility_limit:.1%}")

    # State Info
    with st.expander("Circuit Breaker State"):
        st.json(cb_status)

    st.divider()

    # Audit Trail
    st.subheader("Audit Trail (Recent Events)")
    try:
        from openquant.storage.audit_trail import AUDIT_TRAIL
        events = AUDIT_TRAIL.query(limit=20)
        if events:
            audit_df = pd.DataFrame(events)
            st.dataframe(audit_df, use_container_width=True)
        else:
            st.info("No audit events yet.")
    except Exception as e:
        st.warning(f"Could not load audit trail: {e}")

    st.divider()
    
    # Intelligent Alerts Panel
    st.subheader("🔔 Intelligent Alerts")
    try:
        from openquant.reporting.intelligent_alerts import IntelligentAlerts
        alerts_sys = IntelligentAlerts()
        
        # Load alerts from file
        alerts_file = Path("data/alerts_history.json")
        if alerts_file.exists():
            with open(alerts_file, "r") as f:
                all_alerts = json.load(f)
        else:
            all_alerts = []
            
        # Filter to last 24 hours
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(hours=24)
        recent_alerts = []
        for a in all_alerts:
            try:
                alert_time = datetime.fromisoformat(a.get("timestamp", ""))
                if alert_time > cutoff:
                    recent_alerts.append(a)
            except:
                pass
                
        if recent_alerts:
            # Count by severity
            critical = len([a for a in recent_alerts if a.get("severity") == "critical"])
            warnings = len([a for a in recent_alerts if a.get("severity") == "warning"])
            info = len([a for a in recent_alerts if a.get("severity") == "info"])
            
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Critical", critical, delta_color="inverse" if critical > 0 else "normal")
            c2.metric("🟡 Warning", warnings)
            c3.metric("🔵 Info", info)
            
            # Show alerts as expandable list
            for alert in reversed(recent_alerts[-10:]):  # Last 10
                severity = alert.get("severity", "info")
                sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")
                
                with st.expander(f"{sev_icon} {alert.get('message', 'No message')}"):
                    st.caption(f"Time: {alert.get('timestamp', 'Unknown')[:19]}")
                    st.caption(f"Type: {alert.get('type', 'Unknown')}")
                    if alert.get('details'):
                        st.json(alert['details'])
        else:
            st.success("No alerts in the last 24 hours!")
            
        # Manual alert checks
        if st.button("🔍 Run Alert Checks Now"):
            from openquant.reporting.performance_tracker import PERFORMANCE_TRACKER
            
            # Check drawdown
            if PERFORMANCE_TRACKER.equity_curve:
                eq_values = [pt.get("equity", 0) for pt in PERFORMANCE_TRACKER.equity_curve]
                dd_alert = alerts_sys.check_drawdown(eq_values, threshold=0.25)
                if dd_alert:
                    alerts_sys.add_alert(dd_alert)
                    st.warning(f"Alert generated: {dd_alert.message}")
                    
            # Check PnL anomaly
            if PERFORMANCE_TRACKER.trades:
                pnl_list = [t.pnl_usd for t in PERFORMANCE_TRACKER.trades]
                pnl_alert = alerts_sys.check_pnl_anomaly(pnl_list)
                if pnl_alert:
                    alerts_sys.add_alert(pnl_alert)
                    st.warning(f"Alert generated: {pnl_alert.message}")
                    
            # Save alerts
            alerts_sys.save_alerts()
            st.success("Alert checks completed!")
            st.rerun()

        st.subheader("⚙️ Diagnostics Thresholds")
        try:
            cfg_path = Path("data/diagnostics_config.json")
            curr = {"wfo_drop": 0.2, "profit_factor_min": 1.2, "mc_dd_p95_max": 0.25, "p_value_max": 0.10}
            if cfg_path.exists():
                try:
                    with open(cfg_path, "r") as f:
                        curr.update(json.load(f))
                except Exception:
                    pass
            c1, c2, c3, c4 = st.columns(4)
            wfo_drop = c1.number_input("WFO Degradation Threshold", 0.0, 2.0, float(curr.get("wfo_drop", 0.2)), 0.05)
            pf_min = c2.number_input("Min Profit Factor", 0.5, 5.0, float(curr.get("profit_factor_min", 1.2)), 0.1)
            mc_dd = c3.number_input("MC Max DD (p95)", 0.0, 1.0, float(curr.get("mc_dd_p95_max", 0.25)), 0.01)
            pval_max = c4.number_input("Max p-value", 0.0, 1.0, float(curr.get("p_value_max", 0.10)), 0.01)
            if st.button("Save Thresholds"):
                try:
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(cfg_path, "w") as f:
                        json.dump({
                            "wfo_drop": float(wfo_drop),
                            "profit_factor_min": float(pf_min),
                            "mc_dd_p95_max": float(mc_dd),
                            "p_value_max": float(pval_max)
                        }, f, indent=2)
                    st.success("Diagnostics thresholds saved")
                except Exception as e:
                    st.error(f"Failed to save thresholds: {e}")
        except Exception as e:
            st.warning(f"Thresholds UI error: {e}")

        st.subheader("📑 Diagnostic Report")
        try:
            rep_file = Path("data/diagnostic_report.json")
            if rep_file.exists():
                with open(rep_file, "r") as f:
                    rep = json.load(f)
                c1, c2, c3 = st.columns(3)
                agg = rep.get("aggregate", {})
                c1.metric("Mean Sharpe", f"{agg.get('mean_sharpe', 0):.2f}")
                c2.metric("Mean PF", f"{agg.get('mean_profit_factor', 0):.2f}")
                c3.metric("Median p-value", f"{agg.get('median_p_value', 1):.2f}")
                st.write("Top Results")
                top = rep.get("top_results", [])
                if top:
                    st.dataframe(pd.DataFrame(top), use_container_width=True)
                st.write("Recommendations")
                recs = rep.get("recommendations", [])
                if recs:
                    st.json(recs)
                st.write("ROI Projection")
                st.json(rep.get("roi_projection", {}))
            else:
                st.info("No diagnostic report yet.")
        except Exception as e:
            st.warning(f"Failed to load diagnostic report: {e}")

        st.subheader("🛠️ Optimization Actions")
        if st.button("Generate Optimization Actions"):
            try:
                from openquant.reporting.intelligent_alerts import IntelligentAlerts
                alerts = IntelligentAlerts()
                actions = alerts.propose_optimization_actions("data/results.duckdb")
                st.json(actions)
                st.success("Optimization actions generated and saved.")
            except Exception as e:
                st.error(f"Failed to generate actions: {e}")
            
    except Exception as e:
        st.error(f"Alerts error: {e}")


def view_settings():
    st.header("Settings")

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
            st.success("Risk settings updated (Session only)")
            SCHEDULER.run_config.update(rc)

    st.divider()

    # Circuit Breaker Configuration
    st.subheader("Circuit Breaker Thresholds")
    from openquant.risk.circuit_breaker import CIRCUIT_BREAKER

    with st.form("circuit_breaker_form"):
        cb1, cb2, cb3 = st.columns(3)
        new_daily = cb1.number_input("Daily Loss Limit", 0.01, 0.5, CIRCUIT_BREAKER.daily_loss_limit, 0.01)
        new_dd = cb2.number_input("Drawdown Limit", 0.01, 0.5, CIRCUIT_BREAKER.drawdown_limit, 0.01)
        new_vol = cb3.number_input("Volatility Limit", 0.01, 0.5, CIRCUIT_BREAKER.volatility_limit, 0.01)

        if st.form_submit_button("Update Thresholds"):
            CIRCUIT_BREAKER.daily_loss_limit = new_daily
            CIRCUIT_BREAKER.drawdown_limit = new_dd
            CIRCUIT_BREAKER.volatility_limit = new_vol
            CIRCUIT_BREAKER._save_state()
            st.success("Circuit breaker thresholds updated")

    st.divider()
    st.subheader("Environment Variables")
    with st.expander("View Environment"):
        # Filter sensitive keys
        safe_env = {k: ("***" if any(s in k.lower() for s in ["password", "secret", "key", "token"]) else v)
                    for k, v in os.environ.items()}
        st.json(safe_env)

def view_charting(con):
    st.header("📈 Interactive Charting")
    
    # Sidebar Controls for Charting
    with st.sidebar:
        st.divider()
        st.subheader("Chart Settings")
        exchange_name = st.selectbox("Exchange", ["binance", "kraken", "coinbase"], index=0)
        symbol = st.text_input("Symbol", value="BTC/USDT")
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
        limit = st.number_input("Candles", min_value=50, max_value=1000, value=200)
    
    if st.button("Load Chart"):
        with st.spinner(f"Fetching {symbol} from {exchange_name}..."):
            try:
                import ccxt
                from openquant.utils.plotting import create_interactive_chart
                
                # 1. Fetch OHLC
                ex_class = getattr(ccxt, exchange_name)()
                ohlcv = ex_class.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv:
                    st.error("No data returned.")
                    return
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # 2. Fetch Trades from DB
                trades = []
                try:
                    # Query portfolio_trades for this symbol
                    # We might need to handle symbol format differences (BTC/USDT vs BTCUSDT)
                    # For now, exact match.
                    t_df = _query(con, "SELECT ts as timestamp, side, price, delta_units as size FROM portfolio_trades WHERE symbol = ? ORDER BY ts", (symbol,))
                    if not t_df.empty:
                        trades = t_df.to_dict('records')
                        st.success(f"Found {len(trades)} historical trades.")
                except Exception as e:
                    st.warning(f"Could not load trades: {e}")
                
                # 3. Indicators (Simple MA for demo)
                indicators = {}
                indicators["SMA 20"] = df['Close'].rolling(20).mean()
                indicators["SMA 50"] = df['Close'].rolling(50).mean()
                
                # 4. Plot
                fig = create_interactive_chart(df, symbol=symbol, indicators=indicators, trades=trades)
                st.plotly_chart(fig, width='stretch')
                
            except Exception as e:
                st.error(f"Error loading chart: {e}")

def view_performance():
    """Real-time profit and loss tracking."""
    st.header("💰 Performance")
    
    try:
        from openquant.reporting.performance_tracker import PERFORMANCE_TRACKER
        stats = PERFORMANCE_TRACKER.get_stats(lookback_days=30)
    except Exception as e:
        st.error(f"Error loading performance tracker: {e}")
        return
        
    # Top Row - Key Metrics
    c1, c2, c3, c4 = st.columns(4)
    
    total_return = stats.get("total_return_pct", 0)
    c1.metric(
        "Total Return",
        f"{total_return:+.1%}",
        delta=f"${stats.get('total_pnl', 0):+,.0f}"
    )
    
    daily_pnl = PERFORMANCE_TRACKER.get_daily_pnl()
    c2.metric(
        "Today's P&L",
        f"${daily_pnl:+,.2f}",
        delta="Today" if daily_pnl >= 0 else "Loss Today",
        delta_color="normal" if daily_pnl >= 0 else "inverse"
    )
    
    win_rate = stats.get("win_rate", 0)
    c3.metric(
        "Win Rate",
        f"{win_rate:.1%}",
        delta="Profitable" if win_rate > 0.5 else "Below 50%",
        delta_color="normal" if win_rate > 0.5 else "inverse"
    )
    
    current_dd = stats.get("current_drawdown", 0)
    c4.metric(
        "Current Drawdown",
        f"{current_dd:.1%}",
        delta="Safe" if current_dd < 0.25 else "Warning",
        delta_color="normal" if current_dd < 0.25 else "inverse"
    )
    
    st.divider()
    
    # Second Row - Detailed Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expectancy", f"${stats.get('expectancy', 0):+.2f}")
    c2.metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")
    c3.metric("Sharpe (est)", f"{stats.get('sharpe_estimate', 0):.2f}")
    c4.metric("Total Trades", stats.get("total_trades", 0))
    
    st.divider()
    
    # Equity Curve
    st.subheader("📈 Equity Curve")
    if PERFORMANCE_TRACKER.equity_curve:
        eq_df = pd.DataFrame(PERFORMANCE_TRACKER.equity_curve)
        if not eq_df.empty and 'timestamp' in eq_df.columns:
            eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'])
            
            fig = px.line(
                eq_df, x='timestamp', y='equity',
                title="Account Equity Over Time"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown chart
            if 'drawdown' in eq_df.columns:
                fig2 = px.area(
                    eq_df, x='timestamp', y='drawdown',
                    title="Drawdown",
                    color_discrete_sequence=['red']
                )
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No equity data yet. Start trading to see the equity curve.")
        
    st.divider()
    
    # Wins vs Losses
    st.subheader("📊 Win/Loss Analysis")
    c1, c2 = st.columns(2)
    
    with c1:
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        if wins + losses > 0:
            fig = px.pie(
                values=[wins, losses],
                names=['Wins', 'Losses'],
                title="Trade Outcomes",
                color_discrete_sequence=['green', 'red']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades recorded yet.")
            
    with c2:
        avg_win = stats.get("avg_win", 0)
        avg_loss = stats.get("avg_loss", 0)
        if avg_win > 0 or avg_loss > 0:
            fig = px.bar(
                x=['Average Win', 'Average Loss'],
                y=[avg_win, -avg_loss],
                title="Average Trade Size",
                color=['Win', 'Loss'],
                color_discrete_map={'Win': 'green', 'Loss': 'red'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trade data available.")
            
    st.divider()
    
    # Recent Trades
    st.subheader("📜 Recent Trades")
    if PERFORMANCE_TRACKER.trades:
        trades_df = pd.DataFrame([
            {
                "Time": t.timestamp[:19],
                "Symbol": t.symbol,
                "Side": t.side,
                "P&L": f"${t.pnl_usd:+.2f}",
                "P&L %": f"{t.pnl_pct:+.2%}",
                "Strategy": t.strategy
            }
            for t in PERFORMANCE_TRACKER.trades[-20:][::-1]  # Last 20, newest first
        ])
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("No trades recorded yet.")
        
    # Auto-refresh
    if SCHEDULER.is_running:
        time.sleep(5)
        st.rerun()

def view_ai_analytics():
    st.header("🧠 AI & Optimization Analytics")
    
    tab1, tab2 = st.tabs(["🧬 Genetic Evolution", "🤖 Machine Learning"])
    
    with tab1:
        st.subheader("Genetic Optimization Progress")
        gen_file = Path("data/genetic_population.json")
        if gen_file.exists():
            try:
                with open(gen_file, "r") as f:
                    data = json.load(f)
                    
                gen = data.get("generation", 0)
                pop = data.get("population", [])
                
                st.metric("Current Generation", gen)
                st.metric("Population Size", len(pop))
                
                if pop:
                    df_pop = pd.DataFrame(pop)
                    
                    # Fitness Distribution
                    fig = px.histogram(df_pop, x="fitness", nbins=20, title="Population Fitness Distribution")
                    st.plotly_chart(fig, width='stretch')
                    
                    # Top Genomes
                    st.caption("Top Performing Genomes")
                    st.dataframe(df_pop.sort_values("fitness", ascending=False).head(10), width='stretch')
                    
                    # Scatter Params (if available)
                    if "params" in df_pop.columns:
                        pass
            except Exception as e:
                st.error(f"Error loading genetic data: {e}")
        else:
            st.info("No genetic optimization data found. Run the genetic optimizer to see results here.")
            
    with tab2:
        st.subheader("ML Strategy Insights")
        ml_file = Path("data/ml_metrics.json")
        if ml_file.exists():
            try:
                with open(ml_file, "r") as f:
                    metrics = json.load(f)
                    
                st.caption(f"Last Updated: {metrics.get('timestamp')}")
                st.metric("Model Type", metrics.get("model_type", "Unknown"))
                
                # Feature Importance
                imp = metrics.get("feature_importance", {})
                if imp:
                    df_imp = pd.DataFrame(list(imp.items()), columns=["Feature", "Importance"])
                    df_imp = df_imp.sort_values("Importance", ascending=True)
                    
                    fig = px.bar(df_imp, x="Importance", y="Feature", orientation='h', title="Feature Importance")
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No feature importance data available.")
                    
            except Exception as e:
                st.error(f"Error loading ML metrics: {e}")
        else:
            st.info("No ML metrics found. Run a backtest with MLStrategy to generate data.")

# --- Main App ---

def main():
    st.set_page_config(page_title="OpenQuant Robot", layout="wide", page_icon="🤖")

    # Import risk modules for status display
    from openquant.risk.kill_switch import KILL_SWITCH
    from openquant.risk.circuit_breaker import CIRCUIT_BREAKER

    # Sidebar Navigation
    with st.sidebar:
        st.title("OpenQuant")

        # Status indicators at top of sidebar
        kill_active = KILL_SWITCH.is_active()
        cb_tripped = CIRCUIT_BREAKER.is_tripped()

        if kill_active:
            st.error("KILL SWITCH ACTIVE")
        elif cb_tripped:
            st.warning("CIRCUIT BREAKER TRIPPED")
        elif SCHEDULER.is_running:
            st.success("Robot Running")
        else:
            st.info("Robot Stopped")

        st.divider()

        # Navigation
        page = st.radio(
            "Navigation",
            ["Robot Control", "💰 Performance", "Risk Monitor", "Dashboard", "AI Analytics", "Charting", "Settings"],
            index=0  # Default to Robot Control
        )

        st.divider()
        st.caption(f"OS: {sys.platform}")
        st.caption(f"Time: {datetime.now().strftime('%H:%M')}")

    con = _connect(DB_PATH)

    if page == "Robot Control":
        view_robot_control()
    elif page == "💰 Performance":
        view_performance()
    elif page == "Risk Monitor":
        view_risk_monitor()
    elif page == "Dashboard":
        view_dashboard(con)
    elif page == "AI Analytics":
        view_ai_analytics()
    elif page == "Charting":
        view_charting(con)
    elif page == "Settings":
        view_settings()

if __name__ == "__main__":
    main()

