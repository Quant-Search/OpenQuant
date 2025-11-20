"""Minimal vectorized backtest engine for long/flat signals."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.Series


def backtest_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    fee_bps: float = 1.0,
    weight: float = 1.0,
    stop_loss_atr: float = None,
    take_profit_atr: float = None,
    spread_bps: float = 0.0,
    leverage: float = 1.0,  # Forex leverage (1x = crypto, 50x = typical Forex)
    swap_long: float = 0.0,
    swap_short: float = 0.0,
    pip_value: float = 0.0001,
) -> BacktestResult:
    """Backtest long/flat signals on Close prices with fees in basis points per trade.
    Args:
        df: OHLCV DataFrame with 'Close', 'High', 'Low'.
        signals: Series of {0,1,-1} same index as df. 1=long, -1=short, 0=flat.
        fee_bps: Fee per change in position (entry/exit) in basis points.
        weight: Fraction of capital allocated when in position (0..).
        stop_loss_atr: Stop loss distance in ATR multiples (e.g., 2.0).
        take_profit_atr: Take profit distance in ATR multiples (e.g., 3.0).
        spread_bps: Bid-ask spread in basis points (e.g., 5.0 = 0.05% spread).
        leverage: Leverage multiplier (1.0 = no leverage, 50.0 = 50x Forex leverage).
        swap_long: Swap cost for long positions in pips per day (negative = cost).
        swap_short: Swap cost for short positions in pips per day.
        pip_value: Value of 1 pip in quote currency (e.g., 0.0001 for EURUSD).
    Returns:
        BacktestResult with returns and equity curve (starting at 1.0).
    """
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column")
    w = max(0.0, float(weight))
    lev = max(1.0, float(leverage))

    px = df["Close"].astype(float)
    # Allow -1 (short), 0 (flat), 1 (long)
    sig = signals.reindex(px.index).fillna(0).astype(int)
    sig = sig.clip(-1, 1)  # Ensure valid range

    # Calculate ATR if SL/TP is enabled
    atr = None
    if stop_loss_atr is not None or take_profit_atr is not None:
        if "High" in df.columns and "Low" in df.columns:
            high = df["High"].astype(float)
            low = df["Low"].astype(float)
            tr = pd.concat([
                high - low,
                (high - px.shift(1)).abs(),
                (low - px.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
        else:
            # Fallback: use simple volatility
            atr = px.pct_change().rolling(14).std() * px

    ret = px.pct_change(fill_method=None).fillna(0.0)
    pos = sig.shift(1).fillna(0).astype(int)
    
    # SL/TP Logic: Modify position based on intraday price movements
    if stop_loss_atr or take_profit_atr:
        entry_price = pd.Series(0.0, index=px.index)
        sl_price = pd.Series(float('inf'), index=px.index)
        tp_price = pd.Series(float('inf'), index=px.index)
        
        for i in range(1, len(px)):
            if pos.iloc[i] == 1 and pos.iloc[i-1] == 0:
                # New entry
                entry_price.iloc[i] = px.iloc[i-1]
                if stop_loss_atr and atr is not None:
                    sl_price.iloc[i] = entry_price.iloc[i] - stop_loss_atr * atr.iloc[i-1]
                if take_profit_atr and atr is not None:
                    tp_price.iloc[i] = entry_price.iloc[i] + take_profit_atr * atr.iloc[i-1]
            elif pos.iloc[i] == 1:
                # Continue position
                entry_price.iloc[i] = entry_price.iloc[i-1]
                sl_price.iloc[i] = sl_price.iloc[i-1]
                tp_price.iloc[i] = tp_price.iloc[i-1]
                
                # Check SL hit
                if stop_loss_atr and px.iloc[i] <= sl_price.iloc[i]:
                    pos.iloc[i] = 0  # Force exit
                # Check TP hit
                elif take_profit_atr and px.iloc[i] >= tp_price.iloc[i]:
                    pos.iloc[i] = 0  # Force exit

    pos_change = pos.diff().abs().fillna(pos.abs())
    
    # Fee + Spread cost
    # Fee + Spread cost
    fee = pos_change * w * lev * (fee_bps / 10000.0)
    spread_cost = pos_change * w * lev * (spread_bps / 10000.0) if spread_bps > 0 else 0.0

    # Swap Cost Calculation (Overnight holding)
    # Identify days where position was held overnight (23:59 -> 00:00)
    # Simplified: Apply swap if position held at end of bar and bar crosses day boundary
    # For 1h/4h data, this approximates daily rollover
    swap_cost = pd.Series(0.0, index=px.index)
    if swap_long != 0.0 or swap_short != 0.0:
        # Detect day change
        day_change = px.index.day != pd.Series(px.index).shift(1).fillna(px.index.day).values
        # If position held during day change, apply swap
        # Swap is in pips per day. 
        # Cost = (Swap Pips * Pip Value) / Price * Position Size * Leverage
        # Note: Swap is usually absolute currency value per lot, here we approximate as % of price
        
        # Convert pips to percentage of price approx
        # 1 pip = 0.0001. Price = 1.1000. % = 0.0001/1.1000 ~= 0.009%
        swap_pct_long = (swap_long * pip_value) / px
        swap_pct_short = (swap_short * pip_value) / px
        
        # Apply to long positions
        swap_cost += (pos > 0) * day_change * swap_pct_long.abs() * w * lev * (-1 if swap_long < 0 else 1)
        # Apply to short positions
        swap_cost += (pos < 0) * day_change * swap_pct_short.abs() * w * lev * (-1 if swap_short < 0 else 1)
        
        # Note: swap_long is usually negative (cost). If positive, it's a gain.
        # The formula above adds the swap value (positive or negative) to returns.
        # But wait, strat_ret subtracts costs. Let's align.
        # We will SUBTRACT swap_cost. So swap_cost should be positive for a cost.
        
        # Re-do:
        # Cost = -1 * Swap Rate (if swap is negative, cost is positive)
        # Gain = -1 * Swap Rate (if swap is positive, cost is negative -> gain)
        
        s_long = -1 * swap_long * pip_value / px
        s_short = -1 * swap_short * pip_value / px
        
        swap_impact = pd.Series(0.0, index=px.index)
        swap_impact += (pos > 0) * day_change * s_long * w * lev
        swap_impact += (pos < 0) * day_change * s_short * w * lev
        
        # swap_impact is the % return impact. Negative = loss, Positive = gain.
        # We add it to strat_ret.
    else:
        swap_impact = 0.0

    # Leverage amplifies both returns and costs
    strat_ret = (pos * w * lev) * ret - fee - spread_cost + swap_impact
    equity = (1.0 + strat_ret).cumprod()

    trades = pos_change

    return BacktestResult(equity_curve=equity, returns=strat_ret, positions=pos, trades=trades)


def sharpe(returns: pd.Series, freq: str = "1h", risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe Ratio."""
    if returns.empty:
        return 0.0
    
    # Annualization factor
    if freq == "1h":
        factor = (24 * 365) ** 0.5
    elif freq == "4h":
        factor = (6 * 365) ** 0.5
    elif freq == "1d":
        factor = 365 ** 0.5
    else:
        factor = 252 ** 0.5 # Default to trading days
        
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    if std_ret == 0:
        return 0.0
        
    return factor * (mean_ret - risk_free_rate) / std_ret

