"""Real-Time Performance Monitor Dashboard.

Streamlit-based dashboard showing live P&L, positions, risk metrics,
circuit breaker status, and TCA statistics with auto-refresh.

Run: streamlit run openquant/gui/realtime_monitor.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
from typing import Dict, Any, List

try:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
except Exception:
    pass

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from openquant.reporting.performance_tracker import PERFORMANCE_TRACKER
from openquant.risk.portfolio_guard import GUARD
from openquant.risk.circuit_breaker import CIRCUIT_BREAKER
from openquant.risk.kill_switch import KILL_SWITCH
from openquant.analysis.tca import TCAMonitor
from openquant.paper.state import PortfolioState
from openquant.broker.abstract import Broker
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)

st.set_page_config(
    page_title="Real-Time Monitor",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed"
)

def load_portfolio_state() -> PortfolioState:
    """Load current portfolio state from disk."""
    state_file = Path("data/portfolio_state.json")
    state = PortfolioState()
    
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                data = json.load(f)
                state.cash = data.get("cash", 100_000.0)
                state.holdings = {tuple(k): v for k, v in data.get("holdings", {}).items()}
                state._last_positions_value = data.get("_last_positions_value", 0.0)
                state.daily_start_equity = data.get("daily_start_equity", 0.0)
        except Exception as e:
            LOGGER.error(f"Failed to load portfolio state: {e}")
    
    return state

def get_current_positions() -> List[Dict[str, Any]]:
    """Get current positions from portfolio state."""
    state = load_portfolio_state()
    positions = []
    
    for key, units in state.holdings.items():
        if abs(units) < 1e-9:
            continue
            
        exchange, symbol, timeframe, strategy = key
        avg_price = state.avg_price.get(key, 0.0)
        sl = state.sl_levels.get(key)
        tp = state.tp_levels.get(key)
        
        positions.append({
            "symbol": symbol,
            "strategy": strategy,
            "units": units,
            "side": "LONG" if units > 0 else "SHORT",
            "avg_price": avg_price,
            "notional": abs(units * avg_price),
            "sl": sl,
            "tp": tp,
            "exchange": exchange,
            "timeframe": timeframe
        })
    
    return positions

def calculate_greeks(positions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate portfolio Greeks (simplified for spot trading).
    
    For spot positions, we compute sensitivities:
    - Delta: 1.0 for long, -1.0 for short (normalized by notional)
    - Gamma: 0 (no convexity in spot)
    - Vega: 0 (no volatility exposure in spot)
    - Theta: 0 (no time decay in spot)
    """
    total_notional = sum(p["notional"] for p in positions)
    
    if total_notional == 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    
    net_delta = sum(
        (p["units"] * p["avg_price"]) / total_notional 
        for p in positions
    )
    
    return {
        "delta": net_delta,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0
    }

def calculate_var(equity_curve: List[Dict[str, Any]], confidence: float = 0.95) -> Dict[str, float]:
    """Calculate Value at Risk metrics.
    
    VaR: Maximum expected loss at given confidence level
    CVaR (Expected Shortfall): Average loss beyond VaR threshold
    """
    if not equity_curve or len(equity_curve) < 2:
        return {"var": 0.0, "cvar": 0.0, "confidence": confidence}
    
    equities = [pt["equity"] for pt in equity_curve]
    returns = []
    
    for i in range(1, len(equities)):
        if equities[i-1] > 0:
            ret = (equities[i] - equities[i-1]) / equities[i-1]
            returns.append(ret)
    
    if not returns:
        return {"var": 0.0, "cvar": 0.0, "confidence": confidence}
    
    returns = np.array(returns)
    
    var = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= var]
    cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var
    
    return {
        "var": float(var),
        "cvar": float(cvar),
        "confidence": confidence
    }

def calculate_drawdown_stats(equity_curve: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate detailed drawdown statistics."""
    if not equity_curve:
        return {
            "current_dd": 0.0,
            "max_dd": 0.0,
            "avg_dd": 0.0,
            "dd_duration_days": 0
        }
    
    equities = [pt["equity"] for pt in equity_curve]
    timestamps = [pt["timestamp"] for pt in equity_curve]
    
    peak = 0.0
    max_dd = 0.0
    current_dd = 0.0
    drawdowns = []
    in_drawdown = False
    dd_start = None
    
    for i, eq in enumerate(equities):
        if eq > peak:
            peak = eq
            if in_drawdown and dd_start is not None:
                try:
                    start_time = datetime.fromisoformat(timestamps[dd_start])
                    end_time = datetime.fromisoformat(timestamps[i])
                    duration = (end_time - start_time).days
                    drawdowns.append(duration)
                except Exception:
                    pass
            in_drawdown = False
            dd_start = None
        
        if peak > 0:
            dd = (peak - eq) / peak
            current_dd = dd
            max_dd = max(max_dd, dd)
            
            if dd > 0.01 and not in_drawdown:
                in_drawdown = True
                dd_start = i
    
    avg_dd_duration = np.mean(drawdowns) if drawdowns else 0
    
    return {
        "current_dd": current_dd,
        "max_dd": max_dd,
        "avg_dd": np.mean([d for d in [current_dd, max_dd] if d > 0]) if max_dd > 0 else 0.0,
        "dd_duration_days": avg_dd_duration
    }

def render_live_pnl_section():
    """Render live P&L and performance metrics."""
    st.header("💰 Live P&L & Performance")
    
    state = load_portfolio_state()
    equity = state.cash + state._last_positions_value
    stats = PERFORMANCE_TRACKER.get_stats(lookback_days=30)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_return = stats.get("total_return_pct", 0)
    delta_color = "normal" if total_return >= 0 else "inverse"
    col1.metric(
        "Total Return",
        f"{total_return:+.2%}",
        delta=f"${stats.get('total_pnl', 0):+,.0f}",
        delta_color=delta_color
    )
    
    daily_pnl = PERFORMANCE_TRACKER.get_daily_pnl()
    col2.metric(
        "Today's P&L",
        f"${daily_pnl:+,.2f}",
        delta="Today",
        delta_color="normal" if daily_pnl >= 0 else "inverse"
    )
    
    col3.metric("Current Equity", f"${equity:,.2f}")
    col4.metric("Cash", f"${state.cash:,.2f}")
    col5.metric("Positions Value", f"${state._last_positions_value:,.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Win Rate", f"{stats.get('win_rate', 0):.1%}")
    col2.metric("Profit Factor", f"{stats.get('profit_factor', 0):.2f}")
    col3.metric("Expectancy", f"${stats.get('expectancy', 0):+.2f}")
    col4.metric("Total Trades", stats.get("total_trades", 0))
    
    if PERFORMANCE_TRACKER.equity_curve:
        st.subheader("📈 Equity Curve")
        eq_df = pd.DataFrame(PERFORMANCE_TRACKER.equity_curve[-200:])
        
        if not eq_df.empty and 'timestamp' in eq_df.columns:
            eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_df['timestamp'],
                y=eq_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='#00d4ff', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 255, 0.1)'
            ))
            
            fig.update_layout(
                title="Account Equity Over Time",
                xaxis_title="Time",
                yaxis_title="Equity ($)",
                template="plotly_dark",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_positions_section():
    """Render current positions."""
    st.header("📋 Current Positions")
    
    positions = get_current_positions()
    
    if not positions:
        st.info("No open positions")
        return
    
    df = pd.DataFrame(positions)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Open Positions", len(positions))
    col2.metric("Total Notional", f"${df['notional'].sum():,.2f}")
    
    long_notional = df[df['side'] == 'LONG']['notional'].sum()
    short_notional = df[df['side'] == 'SHORT']['notional'].sum()
    net_exposure = long_notional - short_notional
    col3.metric("Net Exposure", f"${net_exposure:+,.2f}")
    
    display_cols = ['symbol', 'side', 'units', 'avg_price', 'notional', 'strategy']
    if 'sl' in df.columns:
        display_cols.append('sl')
    if 'tp' in df.columns:
        display_cols.append('tp')
    
    st.dataframe(
        df[display_cols].style.format({
            'units': '{:.4f}',
            'avg_price': '${:.2f}',
            'notional': '${:,.2f}',
            'sl': '${:.2f}',
            'tp': '${:.2f}'
        }),
        use_container_width=True
    )
    
    fig = px.pie(
        df,
        values='notional',
        names='symbol',
        title='Position Distribution by Symbol',
        hole=0.4
    )
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)

def render_risk_metrics_section():
    """Render risk metrics including Greeks, VaR, and drawdown."""
    st.header("⚠️ Risk Metrics")
    
    positions = get_current_positions()
    equity_curve = PERFORMANCE_TRACKER.equity_curve
    
    tab1, tab2, tab3 = st.tabs(["Greeks", "VaR/CVaR", "Drawdown"])
    
    with tab1:
        st.subheader("Portfolio Greeks")
        greeks = calculate_greeks(positions)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Delta", f"{greeks['delta']:.4f}")
        col2.metric("Gamma", f"{greeks['gamma']:.4f}")
        col3.metric("Vega", f"{greeks['vega']:.4f}")
        col4.metric("Theta", f"{greeks['theta']:.4f}")
        
        st.info("Note: Greeks are simplified for spot positions. Delta represents net directional exposure.")
    
    with tab2:
        st.subheader("Value at Risk Analysis")
        
        var_95 = calculate_var(equity_curve, confidence=0.95)
        var_99 = calculate_var(equity_curve, confidence=0.99)
        
        col1, col2, col3, col4 = st.columns(4)
        
        state = load_portfolio_state()
        equity = state.cash + state._last_positions_value
        
        col1.metric(
            "VaR (95%)",
            f"{var_95['var']:.2%}",
            delta=f"${equity * var_95['var']:+,.0f}",
            delta_color="inverse"
        )
        col2.metric(
            "CVaR (95%)",
            f"{var_95['cvar']:.2%}",
            delta=f"${equity * var_95['cvar']:+,.0f}",
            delta_color="inverse"
        )
        col3.metric(
            "VaR (99%)",
            f"{var_99['var']:.2%}",
            delta=f"${equity * var_99['var']:+,.0f}",
            delta_color="inverse"
        )
        col4.metric(
            "CVaR (99%)",
            f"{var_99['cvar']:.2%}",
            delta=f"${equity * var_99['cvar']:+,.0f}",
            delta_color="inverse"
        )
        
        cvar_from_guard = GUARD.calculate_cvar(confidence=0.95)
        st.metric("Portfolio CVaR (Historical)", f"{cvar_from_guard:.2%}")
        
        st.markdown("""
        **VaR (Value at Risk)**: Maximum expected loss at given confidence level  
        **CVaR (Conditional VaR)**: Average loss beyond VaR threshold (tail risk)
        """)
    
    with tab3:
        st.subheader("Drawdown Analysis")
        
        dd_stats = calculate_drawdown_stats(equity_curve)
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_dd = dd_stats['current_dd']
        dd_limit = GUARD.limits.get('dd_limit', 0.20)
        dd_pct_of_limit = (current_dd / dd_limit * 100) if dd_limit > 0 else 0
        
        col1.metric(
            "Current Drawdown",
            f"{current_dd:.2%}",
            delta=f"{dd_pct_of_limit:.0f}% of limit",
            delta_color="inverse" if current_dd > dd_limit * 0.8 else "normal"
        )
        col2.metric("Max Drawdown", f"{dd_stats['max_dd']:.2%}")
        col3.metric("Avg Drawdown", f"{dd_stats['avg_dd']:.2%}")
        col4.metric("Avg DD Duration", f"{dd_stats['dd_duration_days']:.0f} days")
        
        if equity_curve and len(equity_curve) > 1:
            eq_df = pd.DataFrame(equity_curve[-200:])
            eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_df['timestamp'],
                y=eq_df['drawdown'] * -100,
                mode='lines',
                name='Drawdown %',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.2)'
            ))
            
            fig.add_hline(
                y=dd_limit * 100,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Limit: {dd_limit:.0%}"
            )
            
            fig.update_layout(
                title="Drawdown Over Time",
                xaxis_title="Time",
                yaxis_title="Drawdown (%)",
                template="plotly_dark",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_circuit_breaker_section():
    """Render circuit breaker and safety system status."""
    st.header("🚨 Circuit Breaker & Safety Status")
    
    cb_status = CIRCUIT_BREAKER.get_status()
    kill_active = KILL_SWITCH.is_active()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if kill_active:
            st.error("🔴 KILL SWITCH ACTIVE")
            st.caption("All trading halted")
        else:
            st.success("🟢 Kill Switch OK")
    
    with col2:
        if cb_status['is_tripped']:
            st.error("🔴 CIRCUIT BREAKER TRIPPED")
            st.caption("Trading halted by risk limits")
        else:
            st.success("🟢 Circuit Breaker OK")
    
    with col3:
        if cb_status['daily_loss_tripped']:
            st.warning("⚠️ Daily Loss Limit Hit")
        else:
            st.success("✓ Daily Loss OK")
    
    with col4:
        if cb_status['drawdown_tripped']:
            st.warning("⚠️ Drawdown Limit Hit")
        else:
            st.success("✓ Drawdown OK")
    
    st.divider()
    
    st.subheader("Risk Limits & Thresholds")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Daily Loss Limit",
        f"{CIRCUIT_BREAKER.daily_loss_limit:.1%}",
        help="Maximum allowed daily loss before halt"
    )
    col2.metric(
        "Drawdown Limit",
        f"{CIRCUIT_BREAKER.drawdown_limit:.1%}",
        help="Maximum drawdown before halt"
    )
    col3.metric(
        "CVaR Limit",
        f"{GUARD.limits.get('cvar_limit', 0.08):.1%}",
        help="Maximum Conditional VaR"
    )
    col4.metric(
        "Max Exposure/Symbol",
        f"{GUARD.limits.get('max_exposure_per_symbol', 0.20):.1%}",
        help="Maximum exposure per symbol"
    )
    
    state = load_portfolio_state()
    equity = state.cash + state._last_positions_value
    
    if cb_status['start_of_day_equity'] > 0:
        daily_pnl_pct = (equity - cb_status['start_of_day_equity']) / cb_status['start_of_day_equity']
        daily_loss_used = abs(min(0, daily_pnl_pct)) / CIRCUIT_BREAKER.daily_loss_limit
    else:
        daily_loss_used = 0
    
    if cb_status['peak_equity'] > 0:
        current_dd = (cb_status['peak_equity'] - equity) / cb_status['peak_equity']
        dd_used = current_dd / CIRCUIT_BREAKER.drawdown_limit
    else:
        dd_used = 0
    
    st.subheader("Limit Utilization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=daily_loss_used * 100,
            title={'text': "Daily Loss Limit Usage (%)"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if daily_loss_used > 0.8 else "orange" if daily_loss_used > 0.5 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        fig.update_layout(height=250, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=dd_used * 100,
            title={'text': "Drawdown Limit Usage (%)"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if dd_used > 0.8 else "orange" if dd_used > 0.5 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            }
        ))
        fig.update_layout(height=250, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Circuit Breaker Details"):
        st.json(cb_status)

def render_tca_section():
    """Render Transaction Cost Analysis statistics."""
    st.header("💸 Transaction Cost Analysis (TCA)")
    
    try:
        tca = TCAMonitor()
        stats = tca.get_stats()
        
        if stats.get("count", 0) == 0:
            st.info("No TCA data yet. Place trades to see execution quality metrics.")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Orders Tracked", stats["count"])
        
        avg_slip = stats['avg_slippage_bps']
        slip_color = "inverse" if avg_slip > 5 else "normal" if avg_slip > 1 else "off"
        col2.metric(
            "Avg Slippage",
            f"{avg_slip:.2f} bps",
            delta=f"{-avg_slip:.2f} bps",
            delta_color=slip_color,
            help="Lower is better. <1bps is excellent, >5bps needs attention"
        )
        
        col3.metric("Total Fees", f"${stats['total_fees']:.2f}")
        
        worst_slip = stats.get('worst_slippage', 0)
        col4.metric(
            "Worst Slippage",
            f"{worst_slip:.2f} bps",
            delta_color="inverse" if worst_slip > 10 else "normal"
        )
        
        if "recent_orders" in stats and stats["recent_orders"]:
            st.subheader("Recent Execution Quality")
            
            recent_df = pd.DataFrame(stats["recent_orders"])
            
            if not recent_df.empty:
                display_cols = ['order_id', 'symbol', 'side', 'quantity', 
                               'arrival_price', 'fill_price', 'slippage_bps', 
                               'fee', 'status']
                
                available_cols = [c for c in display_cols if c in recent_df.columns]
                
                st.dataframe(
                    recent_df[available_cols].style.format({
                        'quantity': '{:.4f}',
                        'arrival_price': '${:.4f}',
                        'fill_price': '${:.4f}',
                        'slippage_bps': '{:.2f}',
                        'fee': '${:.4f}'
                    }),
                    use_container_width=True
                )
                
                if 'slippage_bps' in recent_df.columns and len(recent_df) > 1:
                    fig = px.histogram(
                        recent_df,
                        x='slippage_bps',
                        nbins=20,
                        title='Slippage Distribution (bps)',
                        labels={'slippage_bps': 'Slippage (bps)'}
                    )
                    fig.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"TCA Error: {e}")
        LOGGER.error(f"TCA rendering error: {e}")

def render_header():
    """Render dashboard header with refresh controls."""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.title("📊 Real-Time Performance Monitor")
    
    with col2:
        st.metric("Server Time", datetime.now().strftime("%H:%M:%S"))
    
    with col3:
        refresh_interval = st.selectbox(
            "Auto-Refresh",
            options=[0, 5, 10, 30, 60],
            index=2,
            format_func=lambda x: "Off" if x == 0 else f"{x}s",
            key="refresh_interval"
        )
    
    return refresh_interval

def main():
    """Main dashboard application."""
    refresh_interval = render_header()
    
    st.divider()
    
    render_live_pnl_section()
    
    st.divider()
    
    render_positions_section()
    
    st.divider()
    
    render_risk_metrics_section()
    
    st.divider()
    
    render_circuit_breaker_section()
    
    st.divider()
    
    render_tca_section()
    
    st.divider()
    
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if refresh_interval > 0:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
