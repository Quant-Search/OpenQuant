#!/usr/bin/env python3
"""
OpenQuant Professional Trading Dashboard
=========================================

A professional trading platform interface with:
1. MT5 configuration and connection
2. Robot control (start/stop)
3. Strategy backtesting & optimization
4. Performance analytics

Usage:
    streamlit run robot/dashboard.py
"""

import sys
from pathlib import Path

# Add project root to path for imports when running via streamlit
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime
import time
import threading

# Import robot modules (absolute imports)
from robot.config import (
    Config,
    save_credentials,
    load_saved_credentials,
    delete_saved_credentials,
    _find_mt5_terminal
)
from robot.data_fetcher import DataFetcher
from robot.strategy import KalmanStrategy
from robot.trader import Trader
from robot.risk_manager import RiskManager
from robot.backtester import Backtester, run_backtest
from robot.performance import evaluate_strategy_quality
from robot.optimizer import run_optimization, update_config_with_best_params
from robot.theme import get_full_css, metric_card, status_badge, header_html


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OpenQuant Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark theme CSS
st.markdown(get_full_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize all session state variables."""
    # Robot state
    if "robot_running" not in st.session_state:
        st.session_state.robot_running = False
    if "trader" not in st.session_state:
        # Initialize Trader in paper mode (no real trades)
        st.session_state.trader = Trader(mode="paper")
    if "trade_log" not in st.session_state:
        st.session_state.trade_log = []
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    # MT5 state
    if "mt5_connected" not in st.session_state:
        st.session_state.mt5_connected = False

init_session_state()


# ---------------------------------------------------------------------------
# Sidebar - MT5 Configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("MT5 Configuration")
    
    # Load saved credentials
    saved = load_saved_credentials()
    detected_path = _find_mt5_terminal()
    
    # Status indicator
    if Config.is_mt5_configured():
        st.success("MT5 Configured")
    else:
        st.warning("MT5 Not Configured")
    
    # Auto-detected path display
    if detected_path:
        st.info(f"Auto-detected: {detected_path[:30]}...")
    
    # Credential form
    with st.expander("MT5 Credentials", expanded=not Config.is_mt5_configured()):
        login = st.number_input(
            "Login (Account Number)",
            value=saved.get("login", 0),
            min_value=0,
            step=1
        )
        password = st.text_input(
            "Password",
            value="",  # Never pre-fill password
            type="password",
            help="Enter password to update"
        )
        server = st.text_input(
            "Server",
            value=saved.get("server", ""),
            help="e.g., ICMarkets-Demo"
        )
        terminal_path = st.text_input(
            "Terminal Path",
            value=saved.get("terminal_path", detected_path or ""),
            help="Path to terminal64.exe"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save", width="stretch"):
                if login and password and server and terminal_path:
                    if save_credentials(login, password, server, terminal_path):
                        Config.reload_credentials()
                        st.success("Saved!")
                        st.rerun()
                    else:
                        st.error("Save failed")
                else:
                    st.error("Fill all fields")
        with col2:
            if st.button("Clear", width="stretch"):
                delete_saved_credentials()
                st.rerun()
    
    st.divider()
    
    # Strategy Configuration
    st.header("Strategy Settings")
    symbols = st.multiselect(
        "Symbols",
        options=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"],
        default=Config.SYMBOLS[:3]
    )
    timeframe = st.selectbox(
        "Timeframe",
        options=["1h", "4h", "1d"],
        index=0
    )
    risk_pct = st.slider(
        "Risk per Trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5
    )


# ---------------------------------------------------------------------------
# Main Content - Professional Header
# ---------------------------------------------------------------------------

# Custom header with logo and status
st.markdown(header_html(
    "OpenQuant Trading Platform",
    "Professional Algorithmic Trading System • Kalman Filter Strategy"
), unsafe_allow_html=True)

# Connection status bar
conn_status = "running" if st.session_state.mt5_connected else "stopped"
robot_status = "running" if st.session_state.robot_running else "stopped"
st.markdown(f"""
<div style="display: flex; gap: 1rem; margin-bottom: 1.5rem;">
    {status_badge("MT5 " + ("Connected" if st.session_state.mt5_connected else "Disconnected"), conn_status)}
    {status_badge("Robot " + ("Active" if st.session_state.robot_running else "Inactive"), robot_status)}
</div>
""", unsafe_allow_html=True)

# Tab layout with icons
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚡ Control", "📊 Positions", "📝 Trades",
    "🔬 Backtest", "⚙️ Optimize", "📈 Analytics"
])


# ---------------------------------------------------------------------------
# Tab 1: Robot Control - Professional Layout
# ---------------------------------------------------------------------------

with tab1:
    # Key metrics row with custom styled cards
    equity = st.session_state.trader.get_equity()
    positions = st.session_state.trader.get_positions()
    pnl = equity - 10000  # P&L vs initial
    pnl_status = "positive" if pnl >= 0 else "negative"

    st.markdown(f"""
    <div class="grid-4">
        {metric_card("Robot Status", "🟢 Active" if st.session_state.robot_running else "🔴 Inactive", "neutral")}
        {metric_card("Paper Equity", f"${equity:,.2f}", pnl_status)}
        {metric_card("Today's P&L", f"${pnl:+,.2f}", pnl_status)}
        {metric_card("Open Positions", str(len(positions)), "neutral")}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Control buttons
    col_start, col_stop, col_run_once = st.columns(3)

    with col_start:
        if st.button("Start Robot", width="stretch", type="primary",
                    disabled=st.session_state.robot_running):
            st.session_state.robot_running = True
            st.rerun()

    with col_stop:
        if st.button("Stop Robot", width="stretch",
                    disabled=not st.session_state.robot_running):
            st.session_state.robot_running = False
            st.rerun()

    with col_run_once:
        run_once = st.button("Run Once", width="stretch")

    # Execute one cycle
    if run_once or st.session_state.robot_running:
        with st.spinner("Executing trading cycle..."):
            try:
                # Initialize components
                fetcher = DataFetcher(use_mt5=Config.is_mt5_configured())
                strategy = KalmanStrategy(
                    process_noise=Config.PROCESS_NOISE,
                    measurement_noise=Config.MEASUREMENT_NOISE,
                    threshold=Config.SIGNAL_THRESHOLD
                )
                trader = st.session_state.trader

                # Process each symbol
                results = []
                for symbol in symbols:
                    # Fetch data
                    df = fetcher.fetch(symbol, timeframe, bars=500)
                    if df.empty or len(df) < 50:
                        results.append({
                            "symbol": symbol,
                            "status": "No data",
                            "signal": "-"
                        })
                        continue

                    # Get price and generate signal
                    price = float(df['Close'].iloc[-1])
                    trader.update_paper_prices({symbol: price})

                    signals = strategy.generate_signals(df)
                    signal = int(signals.iloc[-1])
                    signal_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}[signal]

                    # Calculate ATR for position sizing
                    atr = RiskManager.calculate_atr(df)

                    # Check current position
                    positions = trader.get_positions()
                    current_pos = positions.get(symbol, 0)

                    action = "HOLD"

                    # Execute trades
                    if signal == 1 and current_pos <= 0:  # Go long
                        equity = trader.get_equity()
                        sl, tp = RiskManager.calculate_stops(
                            price, atr, "LONG",
                            Config.STOP_LOSS_ATR_MULT,
                            Config.TAKE_PROFIT_ATR_MULT
                        )
                        size = RiskManager.calculate_position_size(
                            equity, price, sl, risk_pct / 100
                        )
                        volume = max(0.01, round(size / 100000, 2))
                        if trader.place_order(symbol, "BUY", volume, sl, tp):
                            action = f"BUY {volume}"
                            st.session_state.trade_log.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "symbol": symbol,
                                "action": "BUY",
                                "volume": volume,
                                "price": price
                            })

                    elif signal == -1 and current_pos >= 0:  # Go short
                        equity = trader.get_equity()
                        sl, tp = RiskManager.calculate_stops(
                            price, atr, "SHORT",
                            Config.STOP_LOSS_ATR_MULT,
                            Config.TAKE_PROFIT_ATR_MULT
                        )
                        size = RiskManager.calculate_position_size(
                            equity, price, sl, risk_pct / 100
                        )
                        volume = max(0.01, round(size / 100000, 2))
                        if trader.place_order(symbol, "SELL", volume, sl, tp):
                            action = f"SELL {volume}"
                            st.session_state.trade_log.append({
                                "time": datetime.now().strftime("%H:%M:%S"),
                                "symbol": symbol,
                                "action": "SELL",
                                "volume": volume,
                                "price": price
                            })

                    results.append({
                        "symbol": symbol,
                        "price": f"{price:.5f}",
                        "signal": signal_str,
                        "action": action
                    })

                # Display results
                st.dataframe(
                    pd.DataFrame(results),
                    width="stretch",
                    hide_index=True
                )

                st.session_state.last_run = datetime.now()

            except Exception as e:
                st.error(f"Error: {e}")

    # Auto-refresh if running
    if st.session_state.robot_running:
        st.info(f"Next cycle in {Config.LOOP_INTERVAL_SECONDS}s. Refresh to see updates.")


# ---------------------------------------------------------------------------
# Tab 2: Positions
# ---------------------------------------------------------------------------

with tab2:
    positions = st.session_state.trader.get_positions()

    if not positions:
        st.info("No open positions")
    else:
        pos_data = []
        for symbol, size in positions.items():
            pos_data.append({
                "Symbol": symbol,
                "Size": size,
                "Side": "LONG" if size > 0 else "SHORT"
            })
        st.dataframe(pd.DataFrame(pos_data), width="stretch", hide_index=True)

    # Paper account summary
    st.divider()
    st.subheader("Paper Account")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Balance", f"${st.session_state.trader._paper_cash:,.2f}")
    with col2:
        st.metric("Equity", f"${st.session_state.trader.get_equity():,.2f}")
    with col3:
        pnl = st.session_state.trader.get_equity() - 10000
        st.metric("P&L", f"${pnl:,.2f}", delta=f"{pnl/100:.1f}%")


# ---------------------------------------------------------------------------
# Tab 3: Trade Log
# ---------------------------------------------------------------------------

with tab3:
    if not st.session_state.trade_log:
        st.info("No trades yet")
    else:
        st.dataframe(
            pd.DataFrame(st.session_state.trade_log[::-1]),  # Most recent first
            width="stretch",
            hide_index=True
        )

    if st.button("Clear Log"):
        st.session_state.trade_log = []
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 4: Backtest
# ---------------------------------------------------------------------------

with tab4:
    st.subheader("🔬 Strategy Backtesting")
    st.info("Test your strategy on historical data before risking real money!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bt_symbol = st.selectbox("Symbol", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"])
    with col2:
        bt_timeframe = st.selectbox("Timeframe", ["H1", "H4", "D1"], key="bt_tf")
    with col3:
        bt_bars = st.number_input("Bars", min_value=100, max_value=5000, value=1000)
    with col4:
        bt_threshold = st.number_input("Signal Threshold", min_value=0.5, max_value=3.0,
                                        value=1.5, step=0.1)

    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner(f"Backtesting {bt_symbol} on {bt_bars} bars..."):
            try:
                result = run_backtest(
                    symbol=bt_symbol,
                    timeframe=bt_timeframe,
                    bars=bt_bars,
                    threshold=bt_threshold
                )
                st.session_state.backtest_result = result
                st.success(f"Backtest complete! {result.metrics.total_trades} trades analyzed.")
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    # Display results if available
    if "backtest_result" in st.session_state and st.session_state.backtest_result:
        result = st.session_state.backtest_result
        m = result.metrics

        st.divider()

        # Key metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Return", f"{m.total_return:.2%}")
        col2.metric("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")
        col3.metric("Max Drawdown", f"{m.max_drawdown:.2%}")
        col4.metric("Win Rate", f"{m.win_rate:.1%}")
        col5.metric("Total Trades", m.total_trades)

        # Equity curve chart
        st.subheader("📈 Equity Curve")
        st.line_chart(result.equity_curve)

        # Trade list
        st.subheader("📝 Trade History")
        if result.trades:
            trade_data = [{
                "Entry": t.entry_time.strftime("%Y-%m-%d %H:%M") if t.entry_time else "-",
                "Exit": t.exit_time.strftime("%Y-%m-%d %H:%M") if t.exit_time else "-",
                "Direction": t.direction,
                "P&L": f"${t.pnl:.2f}",
                "P&L %": f"{t.pnl_pct:.2%}",
                "Reason": t.exit_reason
            } for t in result.trades[-20:]]  # Last 20 trades
            st.dataframe(pd.DataFrame(trade_data), hide_index=True)


# ---------------------------------------------------------------------------
# Tab 5: Performance Stats
# ---------------------------------------------------------------------------

with tab5:
    st.subheader("📈 Performance Analytics")

    if "backtest_result" not in st.session_state or not st.session_state.backtest_result:
        st.warning("Run a backtest first to see performance statistics!")
    else:
        result = st.session_state.backtest_result
        m = result.metrics
        assessment = evaluate_strategy_quality(m)

        # Strategy Health Score
        st.subheader("🏥 Strategy Health Check")

        health_data = []
        color_map = {"excellent": "🟢", "good": "🟡", "acceptable": "🟠", "poor": "🔴"}

        for metric_name, (status, value) in assessment.items():
            health_data.append({
                "Metric": metric_name.replace("_", " ").title(),
                "Value": value,
                "Status": f"{color_map.get(status, '⚪')} {status.upper()}"
            })

        st.dataframe(pd.DataFrame(health_data), hide_index=True, use_container_width=True)

        # Detailed metrics
        st.divider()
        st.subheader("📊 Detailed Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Returns**")
            st.write(f"- Total Return: {m.total_return:.2%}")
            st.write(f"- Annualized Return: {m.annualized_return:.2%}")
            st.write(f"- Expectancy: ${m.expectancy:.2f}/trade")

            st.markdown("**Risk Metrics**")
            st.write(f"- Sharpe Ratio: {m.sharpe_ratio:.2f}")
            st.write(f"- Sortino Ratio: {m.sortino_ratio:.2f}")
            st.write(f"- Calmar Ratio: {m.calmar_ratio:.2f}")
            st.write(f"- Max Drawdown: {m.max_drawdown:.2%}")
            st.write(f"- DD Duration: {m.max_drawdown_duration} bars")

        with col2:
            st.markdown("**Trade Statistics**")
            st.write(f"- Total Trades: {m.total_trades}")
            st.write(f"- Winning: {m.winning_trades} ({m.win_rate:.1%})")
            st.write(f"- Losing: {m.losing_trades}")
            st.write(f"- Profit Factor: {m.profit_factor:.2f}")

            st.markdown("**Win/Loss Analysis**")
            st.write(f"- Average Win: ${m.avg_win:.2f}")
            st.write(f"- Average Loss: ${m.avg_loss:.2f}")
            st.write(f"- Largest Win: ${m.largest_win:.2f}")
            st.write(f"- Largest Loss: ${m.largest_loss:.2f}")
            st.write(f"- Risk/Reward: {m.risk_reward_ratio:.2f}")

        # Statistical significance
        st.divider()
        st.subheader("📐 Statistical Significance")

        if m.is_statistically_significant:
            st.success(f"✅ Results are statistically significant (p={m.p_value:.4f})")
        else:
            st.warning(f"⚠️ Results may not be reliable (p={m.p_value:.4f})")
            if m.total_trades < 30:
                st.info(f"Need at least 30 trades for significance. Current: {m.total_trades}")

        # Interpretation guide
        with st.expander("📖 How to Interpret These Metrics"):
            st.markdown("""
            **Sharpe Ratio** - Risk-adjusted return
            - < 1.0: Subpar
            - 1.0 - 2.0: Good
            - 2.0 - 3.0: Very Good
            - > 3.0: Excellent

            **Max Drawdown** - Largest peak-to-trough decline
            - < 10%: Conservative
            - 10-20%: Moderate
            - 20-30%: Aggressive
            - > 30%: High Risk

            **Profit Factor** - Gross Profit / Gross Loss
            - < 1.0: Losing money
            - 1.0 - 1.5: Break-even to marginal
            - 1.5 - 2.0: Good
            - > 2.0: Excellent

            **Win Rate** - Depends on Risk/Reward ratio
            - At 1:1 R:R, need > 50% to be profitable
            - At 2:1 R:R, need > 33% to be profitable

            **Statistical Significance** - p-value < 0.05 means results are likely not due to chance
            """)


# ---------------------------------------------------------------------------
# Tab 6: Parameter Optimization
# ---------------------------------------------------------------------------

with tab6:
    st.markdown('<div class="section-title">⚙️ Strategy Parameter Optimization</div>', unsafe_allow_html=True)

    st.info("""
    **Walk-Forward Optimization**: Find profitable parameters while preventing overfitting.
    The optimizer tests on 70% of data (in-sample) and validates on 30% (out-of-sample).
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        opt_symbol = st.selectbox("Symbol", ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"], key="opt_sym")
    with col2:
        opt_timeframe = st.selectbox("Timeframe", ["H1", "H4", "D1"], key="opt_tf")
    with col3:
        opt_bars = st.number_input("Historical Bars", min_value=500, max_value=10000, value=2000, step=500)

    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Threshold Grid**")
            thresh_min = st.number_input("Min", value=1.0, step=0.5, key="t_min")
            thresh_max = st.number_input("Max", value=3.0, step=0.5, key="t_max")
        with col2:
            st.markdown("**Minimum Requirements**")
            min_trades = st.number_input("Min Trades", value=20, min_value=10)
            min_sharpe = st.number_input("Min Sharpe", value=0.5, step=0.1)

    if st.button("🚀 Run Optimization", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Fetching historical data...")
            progress_bar.progress(10)

            status_text.text("Running optimization (125 combinations)...")
            progress_bar.progress(30)

            report = run_optimization(
                symbol=opt_symbol,
                timeframe=opt_timeframe,
                bars=opt_bars,
                save_results=True
            )

            progress_bar.progress(100)
            status_text.empty()

            st.session_state.optimization_report = report
            st.success(f"✅ Optimization complete! Tested {report.total_combinations} combinations.")

        except Exception as e:
            st.error(f"Optimization failed: {e}")
            progress_bar.empty()
            status_text.empty()

    # Display results
    if "optimization_report" in st.session_state and st.session_state.optimization_report:
        report = st.session_state.optimization_report

        st.markdown("---")
        st.subheader("📊 Optimization Results")

        # Best parameters card
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">🏆 Best Parameters Found</div>
            <div class="grid-3">
                {metric_card("Threshold", f"{report.best_params.get('threshold', 0):.2f}", "positive")}
                {metric_card("Process Noise", f"{report.best_params.get('process_noise', 0):.4f}", "positive")}
                {metric_card("Measurement Noise", f"{report.best_params.get('measurement_noise', 0):.2f}", "positive")}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tested", report.total_combinations)
        col2.metric("Profitable Combos", report.profitable_combinations)
        col3.metric("Success Rate", f"{100*report.profitable_combinations/max(report.total_combinations,1):.1f}%")

        # Top results table
        if report.all_results:
            st.subheader("📋 Top 10 Parameter Combinations")
            top_results = []
            for r in report.all_results[:10]:
                top_results.append({
                    "Threshold": r.params.get("threshold", 0),
                    "Proc.Noise": r.params.get("process_noise", 0),
                    "Meas.Noise": r.params.get("measurement_noise", 0),
                    "IS Return": f"{r.in_sample_metrics.total_return:.2%}",
                    "IS Sharpe": f"{r.in_sample_metrics.sharpe_ratio:.2f}",
                    "OOS Return": f"{r.out_of_sample_metrics.total_return:.2%}" if r.out_of_sample_metrics else "-",
                    "OOS Sharpe": f"{r.out_of_sample_metrics.sharpe_ratio:.2f}" if r.out_of_sample_metrics else "-",
                    "Robust": "✅" if r.is_robust else "❌",
                    "Score": f"{r.score:.2f}"
                })
            st.dataframe(pd.DataFrame(top_results), hide_index=True, use_container_width=True)

        # Apply best params button
        st.markdown("---")
        if st.button("✅ Apply Best Parameters to Strategy", type="primary"):
            update_config_with_best_params(report.best_params)
            st.success(f"""
            ✅ Strategy updated with optimized parameters:
            - Threshold: {report.best_params.get('threshold', 0):.2f}
            - Process Noise: {report.best_params.get('process_noise', 0):.4f}
            - Measurement Noise: {report.best_params.get('measurement_noise', 0):.2f}
            """)


# ---------------------------------------------------------------------------
# Footer - Professional
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: var(--text-secondary); font-size: 0.8rem; padding: 1rem 0;">
    OpenQuant Trading Platform v1.0 • Last Run: {st.session_state.last_run or 'Never'} •
    <span style="color: var(--accent-blue);">Algorithmic Trading Made Simple</span>
</div>
""", unsafe_allow_html=True)

