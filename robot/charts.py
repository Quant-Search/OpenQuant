"""
Trade Visualization Charts
===========================
Interactive Plotly charts for trade analysis and visualization.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .strategy import KalmanStrategy


@dataclass
class TradeMarker:
    """Represents a trade for visualization."""
    time: pd.Timestamp
    price: float
    side: str  # "BUY" or "SELL"
    pnl: Optional[float] = None
    is_entry: bool = True
    duration: Optional[str] = None
    exit_price: Optional[float] = None


def create_price_chart(
    df: pd.DataFrame,
    title: str = "Price Chart",
    show_volume: bool = True
) -> go.Figure:
    """Create an interactive candlestick chart with volume."""
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.8, 0.2]
        )
    else:
        fig = go.Figure()

    candlestick = go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    )

    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
        colors = ['#26a69a' if c >= o else '#ef5350'
                  for o, c in zip(df['Open'], df['Close'])]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                   marker_color=colors, opacity=0.5),
            row=2, col=1
        )
    else:
        fig.add_trace(candlestick)

    fig.update_layout(
        title=title, xaxis_rangeslider_visible=False,
        template='plotly_dark', height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig


def add_trade_markers(fig: go.Figure, trades: List[TradeMarker], row: int = 1) -> go.Figure:
    """Add trade entry/exit markers to a chart."""
    entries_buy = [t for t in trades if t.is_entry and t.side == "BUY"]
    entries_sell = [t for t in trades if t.is_entry and t.side == "SELL"]
    exits_profit = [t for t in trades if not t.is_entry and t.pnl and t.pnl > 0]
    exits_loss = [t for t in trades if not t.is_entry and t.pnl and t.pnl <= 0]

    if entries_buy:
        fig.add_trace(go.Scatter(
            x=[t.time for t in entries_buy], y=[t.price for t in entries_buy],
            mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff88'),
            name='Buy Entry',
            hovertemplate='<b>BUY</b><br>Price: %{y:.5f}<br>Time: %{x}<extra></extra>'
        ), row=row, col=1)

    if entries_sell:
        fig.add_trace(go.Scatter(
            x=[t.time for t in entries_sell], y=[t.price for t in entries_sell],
            mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff4444'),
            name='Sell Entry',
            hovertemplate='<b>SELL</b><br>Price: %{y:.5f}<br>Time: %{x}<extra></extra>'
        ), row=row, col=1)

    if exits_profit:
        fig.add_trace(go.Scatter(
            x=[t.time for t in exits_profit], y=[t.price for t in exits_profit],
            mode='markers', marker=dict(symbol='diamond', size=10, color='#00ff88',
                                        line=dict(width=1, color='white')),
            name='Profit Exit',
            hovertemplate='<b>EXIT</b><br>P&L: $%{customdata:.2f}<extra></extra>',
            customdata=[t.pnl for t in exits_profit]
        ), row=row, col=1)

    if exits_loss:
        fig.add_trace(go.Scatter(
            x=[t.time for t in exits_loss], y=[t.price for t in exits_loss],
            mode='markers', marker=dict(symbol='diamond', size=10, color='#ff4444',
                                        line=dict(width=1, color='white')),
            name='Loss Exit',
            hovertemplate='<b>EXIT</b><br>P&L: $%{customdata:.2f}<extra></extra>',
            customdata=[t.pnl for t in exits_loss]
        ), row=row, col=1)

    return fig


def add_kalman_signals(
    fig: go.Figure, df: pd.DataFrame, threshold: float = 1.5,
    process_noise: float = 1e-5, measurement_noise: float = 1e-3, row: int = 1
) -> go.Figure:
    """Add Kalman filter estimate and signal bands to chart."""
    strategy = KalmanStrategy(
        process_noise=process_noise, measurement_noise=measurement_noise, threshold=threshold
    )
    prices = df['Close'].values
    estimates, deviations = strategy._kalman_filter(prices)

    fig.add_trace(go.Scatter(
        x=df.index, y=estimates, mode='lines', name='Kalman Estimate',
        line=dict(color='#ffaa00', width=2, dash='dot'),
        hovertemplate='Kalman: %{y:.5f}<extra></extra>'
    ), row=row, col=1)

    dev_series = pd.Series(deviations, index=df.index)
    rolling_std = dev_series.rolling(window=50).std().fillna(1.0)
    upper_band = estimates + threshold * rolling_std.values
    lower_band = estimates - threshold * rolling_std.values

    fig.add_trace(go.Scatter(
        x=df.index, y=upper_band, mode='lines', name=f'Upper ({threshold}σ)',
        line=dict(color='#ff4444', width=1, dash='dash'), opacity=0.5
    ), row=row, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=lower_band, mode='lines', name=f'Lower (-{threshold}σ)',
        line=dict(color='#00ff88', width=1, dash='dash'), opacity=0.5
    ), row=row, col=1)

    return fig



def create_equity_curve_chart(
    equity_curve: pd.Series,
    title: str = "Equity Curve",
    initial_capital: float = 10000
) -> go.Figure:
    """Create an equity curve chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        mode='lines', name='Equity', fill='tozeroy',
        line={'color': '#26a69a', 'width': 2},
        fillcolor='rgba(38, 166, 154, 0.2)'
    ))

    fig.add_hline(
        y=initial_capital, line_dash="dash", line_color="gray",
        annotation_text=f"Initial: ${initial_capital:,.0f}"
    )

    peak_equity = equity_curve.max()
    peak_idx = equity_curve.idxmax()
    fig.add_trace(go.Scatter(
        x=[peak_idx], y=[peak_equity], mode='markers',
        marker={'size': 10, 'color': 'gold', 'symbol': 'star'},
        name=f'Peak: ${peak_equity:,.0f}'
    ))

    fig.update_layout(
        title=title, xaxis_title='Date', yaxis_title='Equity ($)',
        template='plotly_dark', height=400, showlegend=True
    )
    return fig


def create_drawdown_chart(equity_curve: pd.Series) -> go.Figure:
    """Create a drawdown visualization chart."""
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        mode='lines', name='Drawdown', fill='tozeroy',
        line={'color': '#ef5350', 'width': 1},
        fillcolor='rgba(239, 83, 80, 0.3)'
    ))

    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    fig.add_trace(go.Scatter(
        x=[max_dd_idx], y=[max_dd],
        mode='markers+text',
        marker={'size': 10, 'color': 'red', 'symbol': 'x'},
        text=[f'{max_dd:.1f}%'], textposition='bottom center',
        name=f'Max DD: {max_dd:.1f}%'
    ))

    fig.update_layout(
        title='Drawdown Analysis', xaxis_title='Date', yaxis_title='Drawdown (%)',
        template='plotly_dark', height=300,
        yaxis={'zeroline': True, 'zerolinecolor': 'gray'}
    )
    return fig


def create_trade_analysis_chart(trades: List[Dict[str, Any]]) -> go.Figure:
    """Create a chart analyzing trade P&L distribution."""
    if not trades:
        fig = go.Figure()
        fig.add_annotation(text="No trades to analyze", x=0.5, y=0.5)
        return fig

    pnls = [t.get('pnl', 0) for t in trades]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('P&L Distribution', 'Cumulative P&L')
    )

    fig.add_trace(
        go.Histogram(x=pnls, name='P&L', marker_color='#26a69a', opacity=0.7),
        row=1, col=1
    )

    cumulative = np.cumsum(pnls)
    fig.add_trace(
        go.Scatter(y=cumulative, mode='lines+markers', name='Cumulative',
                   line={'color': '#26a69a', 'width': 2}),
        row=1, col=2
    )

    fig.update_layout(template='plotly_dark', height=350, showlegend=False)
    return fig