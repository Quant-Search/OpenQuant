"""Multi-Timeframe Filter for Regime Detection and Signal Validation.

Provides utilities to check multi-timeframe confirmation and regime filtering
for trading strategies. Ensures that signals are aligned with higher timeframe trends
and market regimes before entry.
"""
from typing import Dict, Any, Optional, Callable, List
import pandas as pd
import numpy as np


def check_mtf_confirmation(
    symbol: str,
    timeframe: str,
    signal_direction: int,
    fetch_func: Callable[[str, str], pd.DataFrame],
) -> bool:
    """Check if higher timeframes confirm the signal.
    
    Args:
        symbol: Symbol to check
        timeframe: Current timeframe (e.g., "1h")
        signal_direction: 1 for long, -1 for short, 0 for flat
        fetch_func: Function (symbol, tf) -> pd.DataFrame (OHLCV)
        
    Returns:
        True if higher TFs confirm, False otherwise
    """
    if signal_direction == 0:
        return True
        
    tf_hierarchy = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
    
    try:
        current_idx = tf_hierarchy.index(timeframe)
    except ValueError:
        return True
        
    higher_tfs = tf_hierarchy[current_idx + 1 : current_idx + 3]
    
    for htf in higher_tfs:
        try:
            df_htf = fetch_func(symbol, htf)
            if df_htf.empty:
                continue
                
            close = df_htf["Close"]
            sma = close.rolling(50).mean()
            current_price = close.iloc[-1]
            current_sma = sma.iloc[-1]
            
            if pd.isna(current_sma):
                continue
                
            if signal_direction == 1 and current_price < current_sma:
                return False
            elif signal_direction == -1 and current_price > current_sma:
                return False
                
        except Exception:
            continue
            
    return True


def check_regime_filter(
    df: pd.DataFrame,
    regime_type: str = 'trend',
    min_regime_strength: float = 0.5,
) -> pd.Series:
    """Filter signals based on market regime detection.
    
    Identifies market regimes (trending, ranging, volatile) and returns
    a boolean mask indicating favorable conditions.
    
    Args:
        df: OHLCV DataFrame
        regime_type: Type of regime to favor: 'trend', 'range', 'volatile', or 'any'
        min_regime_strength: Minimum regime strength (0-1) to pass filter
        
    Returns:
        Boolean Series indicating where regime is favorable
    """
    if df.empty or len(df) < 50:
        return pd.Series(True, index=df.index)
        
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    trend_strength = calculate_trend_strength(close)
    volatility = calculate_volatility_regime(close, high, low)
    range_bound = calculate_range_regime(close)
    
    if regime_type == 'trend':
        regime_score = trend_strength
    elif regime_type == 'range':
        regime_score = range_bound
    elif regime_type == 'volatile':
        regime_score = volatility
    elif regime_type == 'any':
        return pd.Series(True, index=df.index)
    else:
        regime_score = trend_strength
        
    return regime_score >= min_regime_strength


def calculate_trend_strength(close: pd.Series, window: int = 50) -> pd.Series:
    """Calculate trend strength using ADX-like logic.
    
    Args:
        close: Close price series
        window: Lookback window
        
    Returns:
        Series of trend strength values (0-1)
    """
    if len(close) < window:
        return pd.Series(0.5, index=close.index)
        
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    
    distance_from_sma = (close - sma).abs() / std
    distance_from_sma = distance_from_sma.fillna(0).clip(0, 3) / 3
    
    price_momentum = close.pct_change(window // 5).fillna(0).abs()
    momentum_normalized = price_momentum.clip(0, 0.2) / 0.2
    
    trend_strength = (distance_from_sma * 0.6 + momentum_normalized * 0.4)
    
    return trend_strength.fillna(0.5)


def calculate_volatility_regime(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Calculate volatility regime indicator.
    
    High values indicate high volatility regime.
    
    Args:
        close: Close price series
        high: High price series
        low: Low price series
        window: Lookback window
        
    Returns:
        Series of volatility regime values (0-1)
    """
    if len(close) < window:
        return pd.Series(0.5, index=close.index)
        
    returns = close.pct_change().fillna(0)
    realized_vol = returns.rolling(window).std()
    
    atr = calculate_atr(high, low, close, window)
    atr_pct = atr / close
    
    vol_percentile = realized_vol.rolling(window * 3).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
        raw=False
    ).fillna(0.5)
    
    atr_percentile = atr_pct.rolling(window * 3).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
        raw=False
    ).fillna(0.5)
    
    volatility_regime = (vol_percentile * 0.5 + atr_percentile * 0.5)
    
    return volatility_regime.fillna(0.5)


def calculate_range_regime(close: pd.Series, window: int = 50) -> pd.Series:
    """Calculate range-bound regime indicator.
    
    High values indicate range-bound market.
    
    Args:
        close: Close price series
        window: Lookback window
        
    Returns:
        Series of range regime values (0-1)
    """
    if len(close) < window:
        return pd.Series(0.5, index=close.index)
        
    rolling_max = close.rolling(window).max()
    rolling_min = close.rolling(window).min()
    rolling_range = rolling_max - rolling_min
    
    position_in_range = (close - rolling_min) / rolling_range
    position_in_range = position_in_range.fillna(0.5)
    
    touches_upper = ((close - rolling_max).abs() / rolling_range < 0.05).rolling(window // 5).sum()
    touches_lower = ((close - rolling_min).abs() / rolling_range < 0.05).rolling(window // 5).sum()
    
    range_activity = (touches_upper + touches_lower) / (window // 5)
    range_activity = range_activity.clip(0, 1)
    
    returns = close.pct_change(window // 5).fillna(0).abs()
    low_directional_move = (returns < 0.05).rolling(window // 5).mean()
    
    range_regime = (range_activity * 0.4 + low_directional_move * 0.6)
    
    return range_regime.fillna(0.5)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: ATR window
        
    Returns:
        ATR series
    """
    if len(close) < 2:
        return pd.Series(0, index=close.index)
        
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    
    return atr.fillna(0)


def check_multi_regime_alignment(
    symbol: str,
    timeframes: List[str],
    fetch_func: Callable[[str, str], pd.DataFrame],
    regime_type: str = 'trend',
) -> Dict[str, bool]:
    """Check regime alignment across multiple timeframes.
    
    Args:
        symbol: Symbol to check
        timeframes: List of timeframes to check
        fetch_func: Function to fetch OHLCV data
        regime_type: Type of regime to check
        
    Returns:
        Dict mapping timeframe to regime confirmation status
    """
    results = {}
    
    for tf in timeframes:
        try:
            df = fetch_func(symbol, tf)
            if df.empty or len(df) < 50:
                results[tf] = True
                continue
                
            regime_mask = check_regime_filter(df, regime_type=regime_type)
            current_regime = regime_mask.iloc[-1] if not regime_mask.empty else True
            results[tf] = bool(current_regime)
            
        except Exception:
            results[tf] = True
            
    return results


def get_regime_score(
    df: pd.DataFrame,
    signal_direction: int,
) -> float:
    """Calculate a regime score indicating how favorable conditions are for the signal.
    
    Args:
        df: OHLCV DataFrame
        signal_direction: 1 for long, -1 for short, 0 for flat
        
    Returns:
        Regime score between 0 (unfavorable) and 1 (favorable)
    """
    if df.empty or len(df) < 50 or signal_direction == 0:
        return 0.5
        
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    trend_strength = calculate_trend_strength(close).iloc[-1]
    volatility = calculate_volatility_regime(close, high, low).iloc[-1]
    
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    current_price = close.iloc[-1]
    
    if pd.isna(sma20) or pd.isna(sma50):
        return 0.5
        
    if signal_direction == 1:
        trend_alignment = 1.0 if (current_price > sma20 > sma50) else 0.3
        recent_momentum = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] if len(close) >= 10 else 0
        momentum_score = min(max(recent_momentum * 10, 0), 1)
        
        score = (trend_alignment * 0.4 + trend_strength * 0.3 + momentum_score * 0.3)
        
    elif signal_direction == -1:
        trend_alignment = 1.0 if (current_price < sma20 < sma50) else 0.3
        recent_momentum = (close.iloc[-10] - close.iloc[-1]) / close.iloc[-10] if len(close) >= 10 else 0
        momentum_score = min(max(recent_momentum * 10, 0), 1)
        
        score = (trend_alignment * 0.4 + trend_strength * 0.3 + momentum_score * 0.3)
        
    else:
        score = 0.5
        
    return float(np.clip(score, 0, 1))
