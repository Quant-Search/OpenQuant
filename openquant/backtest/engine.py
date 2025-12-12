"""Minimal vectorized backtest engine for long/flat signals."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Callable
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

# Lazy import GPU backtest (peut ne pas être disponible)
try:
    from .gpu_backtest import backtest_signals_gpu, is_gpu_backtest_available
    GPU_AVAILABLE = is_gpu_backtest_available()
except ImportError:
    GPU_AVAILABLE = False
    backtest_signals_gpu = None



@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.Series
    trade_details: Optional[pd.DataFrame] = None


def calculate_tod_spread(
    timestamps: pd.DatetimeIndex,
    base_spread_bps: float,
    tod_multipliers: Optional[Dict[int, float]] = None
) -> pd.Series:
    """Calculate time-of-day dependent spread.
    
    Args:
        timestamps: DatetimeIndex of bars
        base_spread_bps: Base spread in basis points
        tod_multipliers: Dict mapping hour -> multiplier (e.g., {0: 1.5, 9: 0.8})
                        Higher multiplier = wider spread during that hour
                        Default uses typical FX/crypto patterns
    
    Returns:
        Series of spread in bps for each timestamp
    """
    if tod_multipliers is None:
        # Default multipliers based on typical market patterns
        # Wider spreads during low liquidity hours (Asian/early European)
        # Tighter spreads during peak London/NY overlap
        tod_multipliers = {
            0: 1.4,   # Midnight - low liquidity
            1: 1.5,   # Early Asian
            2: 1.5,
            3: 1.4,
            4: 1.3,
            5: 1.2,
            6: 1.1,
            7: 1.0,   # London open
            8: 0.9,
            9: 0.85,  # Peak liquidity
            10: 0.85,
            11: 0.85,
            12: 0.85,
            13: 0.9,  # NY open
            14: 0.85,
            15: 0.85,
            16: 0.9,
            17: 1.0,
            18: 1.1,  # London close
            19: 1.2,
            20: 1.3,
            21: 1.3,
            22: 1.4,
            23: 1.4,
        }
    
    hours = timestamps.hour
    multipliers = pd.Series(hours.map(lambda h: tod_multipliers.get(h, 1.0)), index=timestamps)
    return base_spread_bps * multipliers


def calculate_volume_slippage(
    volumes: pd.Series,
    position_changes: pd.Series,
    base_slippage_bps: float = 0.5,
    volume_impact_coeff: float = 0.1
) -> pd.Series:
    """Calculate volume-dependent slippage.
    
    Slippage increases when trade size is large relative to market volume.
    
    Args:
        volumes: Series of bar volumes
        position_changes: Series of position size changes (absolute values)
        base_slippage_bps: Minimum slippage in basis points
        volume_impact_coeff: Coefficient controlling volume impact
                            Higher = more slippage for given volume ratio
    
    Returns:
        Series of slippage in bps for each bar
    """
    if volumes.empty or position_changes.empty:
        return pd.Series(base_slippage_bps, index=position_changes.index)
    
    # Normalize volumes to avoid extreme values
    avg_volume = volumes.rolling(20, min_periods=1).mean()
    avg_volume = avg_volume.replace(0, volumes.mean() if volumes.mean() > 0 else 1.0)
    
    # Volume ratio: higher when trading during low volume
    volume_ratio = 1.0 / (volumes / avg_volume).clip(lower=0.1)
    
    # Slippage scales with volume ratio
    slippage = base_slippage_bps * (1.0 + volume_impact_coeff * (volume_ratio - 1.0))
    
    return slippage.reindex(position_changes.index).fillna(base_slippage_bps)


def calculate_market_impact(
    prices: pd.Series,
    position_changes: pd.Series,
    volumes: pd.Series,
    weight: float = 1.0,
    leverage: float = 1.0,
    participation_rate: float = 0.05,
    impact_exponent: float = 0.6
) -> pd.Series:
    """Calculate market impact for large orders using square-root impact model.
    
    Based on academic research (Almgren, Chriss) showing market impact scales
    with square root of trade size relative to volume.
    
    Args:
        prices: Series of prices
        position_changes: Series of absolute position changes
        volumes: Series of bar volumes
        weight: Position weight (fraction of capital)
        leverage: Leverage multiplier
        participation_rate: Expected participation rate in market volume
        impact_exponent: Power law exponent (typically 0.5-0.7)
    
    Returns:
        Series of market impact cost in bps
    """
    if volumes.empty or position_changes.empty:
        return pd.Series(0.0, index=position_changes.index)
    
    # Estimate trade size as fraction of volume
    # Assume capital base and convert position change to notional volume
    avg_volume = volumes.rolling(20, min_periods=1).mean()
    avg_volume = avg_volume.replace(0, volumes.mean() if volumes.mean() > 0 else 1.0)
    
    # Trade size relative to average volume
    trade_size_ratio = (position_changes * weight * leverage) / avg_volume.reindex(position_changes.index)
    trade_size_ratio = trade_size_ratio.fillna(0.0).clip(upper=1.0)  # Cap at 100% of volume
    
    # Square-root impact: cost ~ (trade_size / volume) ^ exponent
    # Scale by participation rate expectation
    impact_bps = 100.0 * (trade_size_ratio / participation_rate) ** impact_exponent
    
    # Additional volatility-based scaling
    volatility = prices.pct_change().rolling(20, min_periods=1).std().fillna(0.01)
    volatility_scalar = (volatility / volatility.mean()).clip(0.5, 3.0)
    
    impact_bps = impact_bps * volatility_scalar.reindex(position_changes.index).fillna(1.0)
    
    return impact_bps.fillna(0.0)


def calculate_funding_rate(
    timestamps: pd.DatetimeIndex,
    positions: pd.Series,
    funding_rate_bps: float = 1.0,
    funding_interval_hours: int = 8
) -> pd.Series:
    """Calculate funding rate costs for perpetual swap contracts.
    
    Perpetual swaps charge/pay funding rates periodically (typically every 8 hours).
    Funding rates are paid by longs to shorts (positive rate) or vice versa (negative rate).
    
    Args:
        timestamps: DatetimeIndex of bars
        positions: Series of positions (-1 to 1, where sign indicates long/short)
        funding_rate_bps: Funding rate in basis points per interval
                         Positive = longs pay shorts, Negative = shorts pay longs
        funding_interval_hours: Hours between funding payments (typically 8 for crypto)
    
    Returns:
        Series of funding costs (positive = cost, negative = rebate) in bps
    """
    funding_cost = pd.Series(0.0, index=timestamps)
    
    if funding_rate_bps == 0.0:
        return funding_cost
    
    # Identify funding payment times (e.g., 00:00, 08:00, 16:00 UTC)
    hours = timestamps.hour
    
    # Find bars where funding is paid
    funding_hours = set(range(0, 24, funding_interval_hours))
    is_funding_time = hours.isin(funding_hours)
    
    # Also check if hour changed to funding hour (to avoid double-counting)
    prev_hours = pd.Series(timestamps.hour).shift(1).fillna(-1)
    prev_hours.index = timestamps
    hour_changed_to_funding = is_funding_time & (hours != prev_hours)
    
    # Apply funding rate to positions held at funding time
    # Positive position (long) pays positive funding rate
    # Negative position (short) receives positive funding rate
    funding_cost = hour_changed_to_funding * positions * funding_rate_bps
    
    return funding_cost


def calculate_dynamic_funding_rate(
    timestamps: pd.DatetimeIndex,
    positions: pd.Series,
    prices: pd.Series,
    index_prices: Optional[pd.Series] = None,
    base_funding_bps: float = 1.0,
    funding_interval_hours: int = 8,
    premium_sensitivity: float = 0.1
) -> pd.Series:
    """Calculate dynamic funding rate based on perpetual-spot premium.
    
    More realistic model where funding rate adjusts based on market conditions.
    When perpetual trades at premium to spot, funding rate increases to discourage longs.
    
    Args:
        timestamps: DatetimeIndex of bars
        positions: Series of positions
        prices: Series of perpetual contract prices
        index_prices: Series of spot/index prices (if None, uses simple price momentum)
        base_funding_bps: Base funding rate in bps
        funding_interval_hours: Hours between funding payments
        premium_sensitivity: How much funding responds to premium (0.1 = 10% premium -> 0.1 bps change)
    
    Returns:
        Series of funding costs in bps
    """
    # Calculate premium/discount
    if index_prices is not None:
        premium = (prices - index_prices) / index_prices
    else:
        # Fallback: use price momentum as proxy for premium
        # Rising prices often correlate with positive funding
        momentum = prices.pct_change(periods=funding_interval_hours).fillna(0.0)
        premium = momentum
    
    # Dynamic funding rate
    dynamic_funding_bps = base_funding_bps + premium_sensitivity * premium * 10000
    
    # Identify funding times
    hours = timestamps.hour
    funding_hours = set(range(0, 24, funding_interval_hours))
    is_funding_time = hours.isin(funding_hours)
    
    prev_hours = pd.Series(timestamps.hour).shift(1).fillna(-1)
    prev_hours.index = timestamps
    hour_changed_to_funding = is_funding_time & (hours != prev_hours)
    
    # Apply funding
    funding_cost = hour_changed_to_funding * positions * dynamic_funding_bps
    
    return funding_cost


def backtest_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    fee_bps: float = 1.0,
    slippage_bps: float = 0.0,
    weight: float = 1.0,
    stop_loss_atr: float = None,
    take_profit_atr: float = None,
    spread_bps: float = 0.0,
    leverage: float = 1.0,  # Forex leverage (1x = crypto, 50x = typical Forex)
    swap_long: float = 0.0,
    swap_short: float = 0.0,
    funding_bps_long: float = 0.0,
    funding_bps_short: float = 0.0,
    pip_value: float = 0.0001,
    impact_coeff: float = 0.0,
    use_tod_spread: bool = False,
    tod_multipliers: Optional[Dict[int, float]] = None,
    use_volume_slippage: bool = False,
    volume_impact_coeff: float = 0.1,
    use_market_impact: bool = False,
    participation_rate: float = 0.05,
    impact_exponent: float = 0.6,
    use_dynamic_funding: bool = False,
    funding_interval_hours: int = 8,
    funding_rate_bps: float = 1.0,
    index_prices: Optional[pd.Series] = None,
    premium_sensitivity: float = 0.1,
) -> BacktestResult:
    """Backtest long/flat signals on Close prices with fees in basis points per trade.
    Args:
        df: OHLCV DataFrame with 'Close', 'High', 'Low', optionally 'Volume'.
        signals: Series of {0,1,-1} same index as df. 1=long, -1=short, 0=flat.
        fee_bps: Fee per change in position (entry/exit) in basis points.
        slippage_bps: Base slippage in basis points (if use_volume_slippage=False).
        weight: Fraction of capital allocated when in position (0..).
        stop_loss_atr: Stop loss distance in ATR multiples (e.g., 2.0).
        take_profit_atr: Take profit distance in ATR multiples (e.g., 3.0).
        spread_bps: Base bid-ask spread in basis points (if use_tod_spread=False).
        leverage: Leverage multiplier (1.0 = no leverage, 50.0 = 50x Forex leverage).
        swap_long: Swap cost for long positions in pips per day (negative = cost).
        swap_short: Swap cost for short positions in pips per day.
        funding_bps_long: Legacy funding cost for long positions in bps (superseded by use_dynamic_funding).
        funding_bps_short: Legacy funding cost for short positions in bps.
        pip_value: Value of 1 pip in quote currency (e.g., 0.0001 for EURUSD).
        impact_coeff: Legacy simple impact coefficient (superseded by use_market_impact).
        use_tod_spread: If True, use time-of-day dependent spread model.
        tod_multipliers: Custom time-of-day multipliers for spread (hour -> multiplier).
        use_volume_slippage: If True, calculate volume-dependent slippage.
        volume_impact_coeff: Coefficient for volume slippage calculation.
        use_market_impact: If True, use square-root market impact model for large orders.
        participation_rate: Expected participation rate in market volume.
        impact_exponent: Power law exponent for market impact (0.5-0.7).
        use_dynamic_funding: If True, use dynamic funding rate based on premium.
        funding_interval_hours: Hours between funding payments (typically 8).
        funding_rate_bps: Base funding rate in bps for perpetual swaps.
        index_prices: Spot/index prices for dynamic funding calculation.
        premium_sensitivity: Sensitivity of funding rate to premium.
    Returns:
        BacktestResult with returns and equity curve (starting at 1.0).
    """
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column")
    w = max(0.0, float(weight))
    lev = max(1.0, float(leverage))

    px = df["Close"].astype(float)
    # Allow float signals for position sizing (e.g. 0.5, -0.2)
    sig = signals.reindex(px.index).fillna(0).astype(float)
    sig = sig.clip(-1.0, 1.0)  # Ensure valid range [-1, 1]

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
    pos = sig.shift(1).fillna(0.0).astype(float)
    
    # SL/TP Logic: Modify position based on intraday price movements
    if stop_loss_atr or take_profit_atr:
        # Convert to numpy for speed
        px_arr = px.values
        pos_arr = pos.values
        atr_arr = atr.values if atr is not None else np.zeros_like(px_arr)
        
        entry_price = 0.0
        sl_price = -1.0
        tp_price = float('inf')
        
        # We need to iterate because exit depends on entry state
        # Using numpy iteration is faster than pandas iloc
        n = len(px_arr)
        for i in range(1, n):
            curr_pos = pos_arr[i]
            prev_pos = pos_arr[i-1]
            
            if curr_pos == 1 and prev_pos == 0:
                # New entry
                entry_price = px_arr[i-1] # Entry at previous close (signal generation) or open? 
                # Standard assumption: Signal at close, trade at close (or next open). 
                # Here we use previous close as entry reference for SL/TP calculation.
                
                if stop_loss_atr:
                    sl_price = entry_price - stop_loss_atr * atr_arr[i-1]
                else:
                    sl_price = -float('inf')
                    
                if take_profit_atr:
                    tp_price = entry_price + take_profit_atr * atr_arr[i-1]
                else:
                    tp_price = float('inf')
                    
            elif curr_pos == 1:
                # Check SL hit
                if stop_loss_atr and px_arr[i] <= sl_price:
                    pos_arr[i] = 0  # Force exit
                # Check TP hit
                elif take_profit_atr and px_arr[i] >= tp_price:
                    pos_arr[i] = 0  # Force exit
        
        # Update pos series from modified array
        pos = pd.Series(pos_arr, index=px.index)

    pos_change = pos.diff().abs().fillna(pos.abs())
    
    # Fee cost (always applied)
    fee = pos_change * w * lev * (fee_bps / 10000.0)
    
    # Enhanced Spread Cost with Time-of-Day modeling
    if use_tod_spread and isinstance(px.index, pd.DatetimeIndex):
        spread_bps_series = calculate_tod_spread(px.index, spread_bps, tod_multipliers)
        spread_cost = pos_change * w * lev * (spread_bps_series / 10000.0)
    else:
        spread_cost = pos_change * w * lev * (spread_bps / 10000.0) if spread_bps > 0 else 0.0
    
    # Enhanced Slippage Cost with Volume dependency
    if use_volume_slippage and "Volume" in df.columns:
        volumes = df["Volume"].astype(float)
        slippage_bps_series = calculate_volume_slippage(
            volumes, pos_change, slippage_bps, volume_impact_coeff
        )
        slippage_cost = pos_change * w * lev * (slippage_bps_series / 10000.0)
    else:
        slippage_cost = pos_change * w * lev * (slippage_bps / 10000.0) if slippage_bps > 0 else 0.0

    # Swap Cost Calculation (Overnight holding)
    # Identify days where position was held overnight (23:59 -> 00:00)
    # Simplified: Apply swap if position held at end of bar and bar crosses day boundary
    # For 1h/4h data, this approximates daily rollover
    swap_cost = pd.Series(0.0, index=px.index)
    if (swap_long != 0.0 or swap_short != 0.0) or (funding_bps_long != 0.0 or funding_bps_short != 0.0):
        # Detect day change
        # Create a series of day values, shift it, and compare
        day_series = pd.Series(px.index.day, index=px.index)
        day_shifted = day_series.shift(1)
        # Fill NaN with same day (first row won't trigger day change)
        day_shifted = day_shifted.fillna(day_series.iloc[0])
        day_change = day_series != day_shifted
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
        f_long = -1 * (funding_bps_long / 10000.0)
        f_short = -1 * (funding_bps_short / 10000.0)
        
        swap_impact = pd.Series(0.0, index=px.index)
        swap_impact += (pos > 0) * day_change * s_long * w * lev
        swap_impact += (pos < 0) * day_change * s_short * w * lev
        # Funding impact (crypto perpetuals)
        swap_impact += (pos > 0) * day_change * f_long * w * lev
        swap_impact += (pos < 0) * day_change * f_short * w * lev
        
        # swap_impact is the % return impact. Negative = loss, Positive = gain.
        # We add it to strat_ret.
    else:
        swap_impact = 0.0

    # Enhanced Market Impact model for large orders
    if use_market_impact and "Volume" in df.columns:
        volumes = df["Volume"].astype(float)
        impact_bps_series = calculate_market_impact(
            px, pos_change, volumes, w, lev, participation_rate, impact_exponent
        )
        impact_cost = pos_change * w * lev * (impact_bps_series / 10000.0)
    elif impact_coeff > 0.0:
        # Legacy simple market impact model
        vol = ret.rolling(20).std().fillna(0.0)
        impact_cost = pos_change * w * lev * impact_coeff * vol
    else:
        impact_cost = 0.0
    
    # Enhanced Funding Rate for perpetual swaps
    if use_dynamic_funding and isinstance(px.index, pd.DatetimeIndex):
        funding_cost_bps = calculate_dynamic_funding_rate(
            px.index, pos, px, index_prices, funding_rate_bps,
            funding_interval_hours, premium_sensitivity
        )
        funding_cost = w * lev * (funding_cost_bps / 10000.0)
    elif funding_rate_bps != 0.0 and isinstance(px.index, pd.DatetimeIndex):
        # Simple funding rate model
        funding_cost_bps = calculate_funding_rate(
            px.index, pos, funding_rate_bps, funding_interval_hours
        )
        funding_cost = w * lev * (funding_cost_bps / 10000.0)
    else:
        funding_cost = 0.0
    
    # Combine all cost components
    # Note: swap_impact is already in return % terms from legacy code
    strat_ret = (pos * w * lev) * ret - fee - spread_cost - slippage_cost - impact_cost - funding_cost + swap_impact
    equity = (1.0 + strat_ret).cumprod()

    trades = pos_change
    try:
        entries = (pos_change > 0) & (pos > 0)
        exits = (pos_change > 0) & (pos == 0)
        td_rows = []
        for i, ts in enumerate(px.index):
            if entries.iloc[i]:
                td_rows.append({
                    "ts": ts,
                    "side": "BUY" if pos.iloc[i] > 0 else "SELL",
                    "delta_units": float(pos_change.iloc[i] * w * lev),
                    "price": float(px.shift(1).iloc[i] if i > 0 else px.iloc[i]),
                })
            elif exits.iloc[i]:
                td_rows.append({
                    "ts": ts,
                    "side": "SELL" if pos.shift(1).iloc[i] > 0 else "BUY",
                    "delta_units": float(pos_change.iloc[i] * w * lev),
                    "price": float(px.shift(1).iloc[i] if i > 0 else px.iloc[i]),
                })
        trade_details = pd.DataFrame(td_rows)
    except Exception:
        trade_details = None

    return BacktestResult(equity_curve=equity, returns=strat_ret, positions=pos, trades=trades, trade_details=trade_details)


def summarize_performance(result: BacktestResult, freq: str = "1h") -> dict:
    from .metrics import sharpe as _sharpe, sortino as _sortino, max_drawdown as _mdd, win_rate as _wr, profit_factor as _pf
    s = float(_sharpe(result.returns, freq=freq))
    so = float(_sortino(result.returns, freq=freq))
    dd = float(_mdd(result.equity_curve))
    wr = float(_wr(result.returns))
    pf = float(_pf(result.returns))
    return {
        "sharpe": s,
        "sortino": so,
        "max_drawdown": dd,
        "win_rate": wr,
        "profit_factor": pf,
    }


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


def auto_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    use_gpu: bool = True,
    **kwargs
) -> BacktestResult:
    """Backtest automatique avec sélection GPU/CPU.
    
    Wrapper intelligent qui choisit automatiquement GPU si disponible,
    sinon fallback sur CPU. Permet de maximiser performance sans
    se soucier de l'environnement d'exécution.
    
    Args:
        df: OHLCV DataFrame
        signals: Series de signaux {-1, 0, 1}
        use_gpu: Si True, utilise GPU si disponible (default: True)
        **kwargs: Arguments passés à backtest_signals(_gpu)
        
    Returns:
        BacktestResult
        
    Example:
        >>> # Utilise GPU si disponible, sinon CPU automatiquement
        >>> result = auto_backtest(df, signals, fee_bps=1.0)
    """
    # Détermine si on utilise GPU
    should_use_gpu = use_gpu and GPU_AVAILABLE
    
    if should_use_gpu:
        LOGGER.debug("Using GPU-accelerated backtest")
        return backtest_signals_gpu(df=df, signals=signals, **kwargs)
    else:
        if use_gpu and not GPU_AVAILABLE:
            LOGGER.warning(
                "GPU backtest demandé mais non disponible. "
                "Fallback sur CPU backtest."
            )
        return backtest_signals(df=df, signals=signals, **kwargs)

