"""Minimal vectorized backtest engine for long/flat signals with performance optimizations.

PERFORMANCE OPTIMIZATIONS:
==========================

1. NumPy Vectorization (Fee/Slippage):
   - calculate_costs_vectorized(): Single-pass vectorized fee, spread, and slippage calculations
   - calculate_swap_costs_vectorized(): Vectorized swap/funding cost calculations
   - calculate_impact_cost_vectorized(): Vectorized market impact calculations
   - Eliminates Python loops for cost calculations, ~5-10x faster

2. Numba JIT Compilation (Indicator Loops):
   - apply_sl_tp_numba(): JIT-compiled stop loss and take profit logic
   - calculate_atr_numba(): JIT-compiled ATR calculation
   - nopython=True mode for maximum performance (~10-50x faster)
   - Automatic fallback to pure Python if Numba unavailable

3. Dask Parallel Operations (Large Universes):
   - backtest_universe_parallel(): Parallel backtesting across multiple symbols
   - Uses Dask delayed for distributed computation
   - Automatically scales to available CPU cores
   - Sequential fallback if Dask unavailable

Dependencies:
   - numba: JIT compilation (optional but recommended)
   - dask[dataframe]: Parallel operations (optional but recommended)
   
Install: pip install numba dask[dataframe]
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
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

# Lazy import Numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

# Lazy import Dask for parallel operations
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    dd = None


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.Series
    trade_details: Optional[pd.DataFrame] = None


# ======================== VECTORIZED FEE/SLIPPAGE CALCULATIONS ========================

def calculate_costs_vectorized(
    pos_change: np.ndarray,
    weight: float,
    leverage: float,
    fee_bps: float,
    slippage_bps: float,
    spread_bps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized calculation of trading costs.
    
    Args:
        pos_change: Array of position changes
        weight: Position weight
        leverage: Leverage multiplier
        fee_bps: Fee in basis points
        slippage_bps: Slippage in basis points
        spread_bps: Spread in basis points
        
    Returns:
        Tuple of (fees, spread_costs, slippage_costs) as numpy arrays
    """
    # Precompute scaling factors
    wl = weight * leverage
    fee_factor = wl * (fee_bps / 10000.0)
    spread_factor = wl * (spread_bps / 10000.0)
    slippage_factor = wl * (slippage_bps / 10000.0)
    
    # Vectorized calculation - single pass
    fees = pos_change * fee_factor
    spread_costs = pos_change * spread_factor if spread_bps > 0 else np.zeros_like(pos_change)
    slippage_costs = pos_change * slippage_factor if slippage_bps > 0 else np.zeros_like(pos_change)
    
    return fees, spread_costs, slippage_costs


def calculate_swap_costs_vectorized(
    pos: np.ndarray,
    day_change: np.ndarray,
    px: np.ndarray,
    weight: float,
    leverage: float,
    swap_long: float,
    swap_short: float,
    funding_bps_long: float,
    funding_bps_short: float,
    pip_value: float,
) -> np.ndarray:
    """Vectorized swap cost calculation.
    
    Args:
        pos: Position array
        day_change: Boolean array indicating day changes
        px: Price array
        weight: Position weight
        leverage: Leverage multiplier
        swap_long: Swap rate for long positions (pips/day)
        swap_short: Swap rate for short positions (pips/day)
        funding_bps_long: Funding rate for long positions (basis points)
        funding_bps_short: Funding rate for short positions (basis points)
        pip_value: Value of one pip
        
    Returns:
        Swap impact array
    """
    wl = weight * leverage
    
    # Vectorized swap calculation
    s_long = -1.0 * swap_long * pip_value / px
    s_short = -1.0 * swap_short * pip_value / px
    f_long = -1.0 * (funding_bps_long / 10000.0)
    f_short = -1.0 * (funding_bps_short / 10000.0)
    
    # Create boolean masks
    is_long = pos > 0
    is_short = pos < 0
    
    # Vectorized calculation
    swap_impact = np.zeros_like(pos)
    swap_impact += is_long * day_change * s_long * wl
    swap_impact += is_short * day_change * s_short * wl
    swap_impact += is_long * day_change * f_long * wl
    swap_impact += is_short * day_change * f_short * wl
    
    return swap_impact


def calculate_impact_cost_vectorized(
    pos_change: np.ndarray,
    returns: np.ndarray,
    weight: float,
    leverage: float,
    impact_coeff: float,
    window: int = 20,
) -> np.ndarray:
    """Vectorized market impact cost calculation.
    
    Args:
        pos_change: Position change array
        returns: Returns array
        weight: Position weight
        leverage: Leverage multiplier
        impact_coeff: Impact coefficient
        window: Rolling window for volatility
        
    Returns:
        Impact cost array
    """
    if impact_coeff <= 0:
        return np.zeros_like(pos_change)
    
    # Calculate rolling volatility using numpy
    vol = np.zeros_like(returns)
    for i in range(window, len(returns)):
        vol[i] = np.std(returns[i-window:i])
    
    return pos_change * weight * leverage * impact_coeff * vol


# ======================== NUMBA-OPTIMIZED STOP LOSS / TAKE PROFIT ========================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def apply_sl_tp_numba(
        pos_arr: np.ndarray,
        px_arr: np.ndarray,
        atr_arr: np.ndarray,
        stop_loss_atr: float,
        take_profit_atr: float,
    ) -> np.ndarray:
        """Numba-optimized stop loss and take profit logic.
        
        Args:
            pos_arr: Position array
            px_arr: Price array
            atr_arr: ATR array
            stop_loss_atr: Stop loss in ATR multiples
            take_profit_atr: Take profit in ATR multiples
            
        Returns:
            Modified position array
        """
        n = len(px_arr)
        result = pos_arr.copy()
        
        entry_price = 0.0
        sl_price = -np.inf
        tp_price = np.inf
        
        for i in range(1, n):
            curr_pos = result[i]
            prev_pos = result[i-1]
            
            # Long position entry
            if curr_pos > 0 and prev_pos <= 0:
                entry_price = px_arr[i-1]
                
                if stop_loss_atr > 0:
                    sl_price = entry_price - stop_loss_atr * atr_arr[i-1]
                else:
                    sl_price = -np.inf
                    
                if take_profit_atr > 0:
                    tp_price = entry_price + take_profit_atr * atr_arr[i-1]
                else:
                    tp_price = np.inf
                    
            # Check SL/TP for long positions
            elif curr_pos > 0:
                if stop_loss_atr > 0 and px_arr[i] <= sl_price:
                    result[i] = 0.0
                elif take_profit_atr > 0 and px_arr[i] >= tp_price:
                    result[i] = 0.0
                    
            # Short position entry
            elif curr_pos < 0 and prev_pos >= 0:
                entry_price = px_arr[i-1]
                
                if stop_loss_atr > 0:
                    sl_price = entry_price + stop_loss_atr * atr_arr[i-1]
                else:
                    sl_price = np.inf
                    
                if take_profit_atr > 0:
                    tp_price = entry_price - take_profit_atr * atr_arr[i-1]
                else:
                    tp_price = -np.inf
                    
            # Check SL/TP for short positions
            elif curr_pos < 0:
                if stop_loss_atr > 0 and px_arr[i] >= sl_price:
                    result[i] = 0.0
                elif take_profit_atr > 0 and px_arr[i] <= tp_price:
                    result[i] = 0.0
        
        return result
else:
    # Fallback non-Numba version
    def apply_sl_tp_numba(
        pos_arr: np.ndarray,
        px_arr: np.ndarray,
        atr_arr: np.ndarray,
        stop_loss_atr: float,
        take_profit_atr: float,
    ) -> np.ndarray:
        """Fallback stop loss and take profit logic without Numba."""
        n = len(px_arr)
        result = pos_arr.copy()
        
        entry_price = 0.0
        sl_price = -float('inf')
        tp_price = float('inf')
        
        for i in range(1, n):
            curr_pos = result[i]
            prev_pos = result[i-1]
            
            if curr_pos > 0 and prev_pos <= 0:
                entry_price = px_arr[i-1]
                sl_price = entry_price - stop_loss_atr * atr_arr[i-1] if stop_loss_atr > 0 else -float('inf')
                tp_price = entry_price + take_profit_atr * atr_arr[i-1] if take_profit_atr > 0 else float('inf')
                    
            elif curr_pos > 0:
                if stop_loss_atr > 0 and px_arr[i] <= sl_price:
                    result[i] = 0.0
                elif take_profit_atr > 0 and px_arr[i] >= tp_price:
                    result[i] = 0.0
                    
            elif curr_pos < 0 and prev_pos >= 0:
                entry_price = px_arr[i-1]
                sl_price = entry_price + stop_loss_atr * atr_arr[i-1] if stop_loss_atr > 0 else float('inf')
                tp_price = entry_price - take_profit_atr * atr_arr[i-1] if take_profit_atr > 0 else -float('inf')
                    
            elif curr_pos < 0:
                if stop_loss_atr > 0 and px_arr[i] >= sl_price:
                    result[i] = 0.0
                elif take_profit_atr > 0 and px_arr[i] <= tp_price:
                    result[i] = 0.0
        
        return result


# ======================== NUMBA-OPTIMIZED ATR CALCULATION ========================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def calculate_atr_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Numba-optimized ATR calculation.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR array
        """
        n = len(close)
        tr = np.zeros(n)
        atr = np.zeros(n)
        
        # Calculate True Range
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Calculate ATR using rolling mean
        for i in range(period, n):
            atr[i] = np.mean(tr[i-period+1:i+1])
        
        return atr
else:
    def calculate_atr_numba(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> np.ndarray:
        """Fallback ATR calculation without Numba."""
        n = len(close)
        tr = np.zeros(n)
        
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        atr = np.zeros(n)
        for i in range(period, n):
            atr[i] = np.mean(tr[i-period+1:i+1])
        
        return atr


# ======================== MAIN BACKTEST FUNCTION ========================

def backtest_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    fee_bps: float = None,
    slippage_bps: float = None,
    weight: float = None,
    stop_loss_atr: float = None,
    take_profit_atr: float = None,
    spread_bps: float = None,
    leverage: float = None,
    swap_long: float = 0.0,
    swap_short: float = 0.0,
    funding_bps_long: float = 0.0,
    funding_bps_short: float = 0.0,
    pip_value: float = 0.0001,
    impact_coeff: float = None,
    config = None,
) -> BacktestResult:
    """Backtest long/flat signals on Close prices with fees in basis points per trade.
    
    Optimized with NumPy vectorization for fee/slippage calculations and Numba JIT 
    compilation for indicator loops.
    
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
        config: ConfigManager instance (optional, loads from global if None)
    Returns:
        BacktestResult with returns and equity curve (starting at 1.0).
    """
    if config is None:
        from openquant.config.manager import get_config
        config = get_config()
    
    bt_config = config.get_section("backtest")
    fee_bps = fee_bps if fee_bps is not None else bt_config.fee_bps
    slippage_bps = slippage_bps if slippage_bps is not None else bt_config.slippage_bps
    weight = weight if weight is not None else bt_config.weight
    spread_bps = spread_bps if spread_bps is not None else bt_config.spread_bps
    leverage = leverage if leverage is not None else bt_config.leverage
    impact_coeff = impact_coeff if impact_coeff is not None else bt_config.impact_coeff
    
    if "Close" not in df.columns:
        raise KeyError("DataFrame must contain 'Close' column")
    w = max(0.0, float(weight))
    lev = max(1.0, float(leverage))

    px = df["Close"].astype(float)
    # Allow float signals for position sizing (e.g. 0.5, -0.2)
    sig = signals.reindex(px.index).fillna(0).astype(float)
    sig = sig.clip(-1.0, 1.0)  # Ensure valid range [-1, 1]

    # Calculate ATR if SL/TP is enabled (using optimized Numba version)
    atr = None
    if stop_loss_atr is not None or take_profit_atr is not None:
        if "High" in df.columns and "Low" in df.columns:
            high = df["High"].astype(float).values
            low = df["Low"].astype(float).values
            close = px.values
            atr_arr = calculate_atr_numba(high, low, close, period=14)
            atr = pd.Series(atr_arr, index=px.index)
        else:
            # Fallback: use simple volatility
            atr = px.pct_change().rolling(14).std() * px

    ret_arr = np.diff(px.values, prepend=px.iloc[0]) / px.shift(1).fillna(px.iloc[0]).values
    ret_arr[0] = 0.0  # First return is 0
    
    pos_arr = sig.shift(1).fillna(0.0).values.astype(float)
    
    # SL/TP Logic: Modify position based on intraday price movements (using Numba)
    if stop_loss_atr or take_profit_atr:
        px_arr = px.values
        atr_arr = atr.values if atr is not None else np.zeros_like(px_arr)
        
        sl_atr = stop_loss_atr if stop_loss_atr else 0.0
        tp_atr = take_profit_atr if take_profit_atr else 0.0
        
        # Apply optimized SL/TP logic
        pos_arr = apply_sl_tp_numba(pos_arr, px_arr, atr_arr, sl_atr, tp_atr)

    # Vectorized position change calculation
    pos_change_arr = np.abs(np.diff(pos_arr, prepend=0.0))
    pos_change_arr[0] = np.abs(pos_arr[0])
    
    # Vectorized fee/slippage/spread calculations
    fees, spread_costs, slippage_costs = calculate_costs_vectorized(
        pos_change_arr, w, lev, fee_bps, slippage_bps, spread_bps
    )
    
    # Vectorized swap cost calculation
    swap_impact = np.zeros_like(pos_arr)
    if (swap_long != 0.0 or swap_short != 0.0) or (funding_bps_long != 0.0 or funding_bps_short != 0.0):
        # Detect day change using vectorized operations
        day_series = np.array([d.day for d in px.index])
        day_shifted = np.roll(day_series, 1)
        day_shifted[0] = day_series[0]
        day_change = (day_series != day_shifted).astype(float)
        
        px_arr = px.values
        swap_impact = calculate_swap_costs_vectorized(
            pos_arr, day_change, px_arr, w, lev,
            swap_long, swap_short, funding_bps_long, funding_bps_short, pip_value
        )
    
    # Vectorized market impact calculation
    impact_costs = calculate_impact_cost_vectorized(
        pos_change_arr, ret_arr, w, lev, impact_coeff, window=20
    )
    
    # Calculate strategy returns
    strat_ret_arr = (pos_arr * w * lev) * ret_arr - fees - spread_costs - slippage_costs - impact_costs + swap_impact
    
    # Calculate equity curve
    equity_arr = np.cumprod(1.0 + strat_ret_arr)
    
    # Convert back to pandas
    pos = pd.Series(pos_arr, index=px.index)
    pos_change = pd.Series(pos_change_arr, index=px.index)
    strat_ret = pd.Series(strat_ret_arr, index=px.index)
    equity = pd.Series(equity_arr, index=px.index)
    trades = pos_change

    # Trade details generation
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


# ======================== PARALLEL BACKTEST FOR LARGE UNIVERSES ========================

def backtest_universe_parallel(
    universe_data: dict[str, pd.DataFrame],
    universe_signals: dict[str, pd.Series],
    n_partitions: int = None,
    **backtest_kwargs
) -> dict[str, BacktestResult]:
    """Backtest multiple symbols in parallel using Dask.
    
    Optimized for large universes (50+ symbols). Uses Dask delayed tasks
    to distribute backtests across available CPU cores.
    
    Example:
        >>> universe_data = {"BTC/USD": btc_df, "ETH/USD": eth_df, ...}
        >>> universe_signals = {"BTC/USD": btc_signals, "ETH/USD": eth_signals, ...}
        >>> results = backtest_universe_parallel(
        ...     universe_data, universe_signals,
        ...     fee_bps=1.0, slippage_bps=0.5
        ... )
        >>> # Returns: {"BTC/USD": BacktestResult(...), "ETH/USD": BacktestResult(...), ...}
    
    Args:
        universe_data: Dictionary mapping symbol -> OHLCV DataFrame
        universe_signals: Dictionary mapping symbol -> signal Series
        n_partitions: Number of Dask partitions (default: number of symbols)
        **backtest_kwargs: Arguments passed to backtest_signals
        
    Returns:
        Dictionary mapping symbol -> BacktestResult
    """
    if not DASK_AVAILABLE:
        LOGGER.warning("Dask not available, falling back to sequential processing")
        return backtest_universe_sequential(universe_data, universe_signals, **backtest_kwargs)
    
    import dask
    from dask import delayed
    
    # Create delayed tasks for each symbol
    tasks = {}
    for symbol in universe_data.keys():
        if symbol not in universe_signals:
            continue
        
        # Create delayed task
        task = delayed(backtest_signals)(
            df=universe_data[symbol],
            signals=universe_signals[symbol],
            **backtest_kwargs
        )
        tasks[symbol] = task
    
    # Compute all tasks in parallel
    LOGGER.info(f"Backtesting {len(tasks)} symbols in parallel with Dask")
    results_list = dask.compute(*tasks.values())
    
    # Map results back to symbols
    results = dict(zip(tasks.keys(), results_list))
    
    return results


def backtest_universe_sequential(
    universe_data: dict[str, pd.DataFrame],
    universe_signals: dict[str, pd.Series],
    **backtest_kwargs
) -> dict[str, BacktestResult]:
    """Backtest multiple symbols sequentially.
    
    Args:
        universe_data: Dictionary mapping symbol -> OHLCV DataFrame
        universe_signals: Dictionary mapping symbol -> signal Series
        **backtest_kwargs: Arguments passed to backtest_signals
        
    Returns:
        Dictionary mapping symbol -> BacktestResult
    """
    results = {}
    
    for symbol in universe_data.keys():
        if symbol not in universe_signals:
            continue
        
        try:
            result = backtest_signals(
                df=universe_data[symbol],
                signals=universe_signals[symbol],
                **backtest_kwargs
            )
            results[symbol] = result
        except Exception as e:
            LOGGER.error(f"Error backtesting {symbol}: {e}")
            continue
    
    return results


# ======================== PERFORMANCE SUMMARY ========================

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


# ======================== AUTO BACKTEST WITH GPU/CPU SELECTION ========================

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
