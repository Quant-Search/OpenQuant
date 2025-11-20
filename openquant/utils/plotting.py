"""Plotting utilities for OpenQuant Dashboard."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional

def create_interactive_chart(
    df: pd.DataFrame, 
    symbol: str = "Unknown", 
    indicators: Optional[Dict[str, pd.Series]] = None,
    trades: Optional[List[Dict[str, Any]]] = None
) -> go.Figure:
    """Create an interactive candlestick chart with indicators and trade markers.
    
    Args:
        df: DataFrame with index as Datetime and columns Open, High, Low, Close, Volume.
        symbol: Symbol name for title.
        indicators: Dict of {name: Series} to overlay (e.g. {"EMA 20": ...}).
        trades: List of trade dicts (timestamp, side, price, size) to plot markers.
    """
    # Create subplots: Row 1 = Price, Row 2 = Volume
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05, 
        row_heights=[0.8, 0.2],
        subplot_titles=(f"{symbol} Price", "Volume")
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # 2. Indicators
    if indicators:
        for name, series in indicators.items():
            # Align series with df index if needed
            fig.add_trace(
                go.Scatter(
                    x=series.index, 
                    y=series.values, 
                    mode='lines', 
                    name=name,
                    line=dict(width=1)
                ),
                row=1, col=1
            )

    # 3. Trade Markers
    if trades:
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []
        
        for t in trades:
            ts = pd.to_datetime(t.get('timestamp'))
            price = float(t.get('price', 0.0))
            side = str(t.get('side', '')).upper()
            
            if side == 'BUY':
                buy_x.append(ts)
                buy_y.append(price)
            elif side == 'SELL':
                sell_x.append(ts)
                sell_y.append(price)
        
        if buy_x:
            fig.add_trace(
                go.Scatter(
                    x=buy_x, y=buy_y,
                    mode='markers',
                    name='Buy',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )
            
        if sell_x:
            fig.add_trace(
                go.Scatter(
                    x=sell_x, y=sell_y,
                    mode='markers',
                    name='Sell',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=1, col=1
            )

    # 4. Volume
    if 'Volume' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index, 
                y=df['Volume'], 
                name='Volume',
                marker_color='rgba(100, 100, 100, 0.5)'
            ),
            row=2, col=1
        )

    # Layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark"
    )
    
    return fig
