from typing import Dict, Any, Optional, Callable
import pandas as pd

def check_mtf_confirmation(
    symbol: str,
    timeframe: str,
    signal_direction: int,  # 1 = long, -1 = short, 0 = flat
    fetch_func: Callable[[str, str], pd.DataFrame],  # Function to fetch OHLCV for (symbol, tf)
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
        return True  # No need to confirm flat
        
    # Timeframe hierarchy
    tf_hierarchy = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
    
    try:
        current_idx = tf_hierarchy.index(timeframe)
    except ValueError:
        # Unknown timeframe, skip MTF check
        return True
        
    # Check next 2 higher timeframes
    higher_tfs = tf_hierarchy[current_idx + 1 : current_idx + 3]
    
    for htf in higher_tfs:
        try:
            df_htf = fetch_func(symbol, htf)
            if df_htf.empty:
                continue
                
            # Simple MTF check: Is price above/below SMA 50?
            close = df_htf["Close"]
            sma = close.rolling(50).mean()
            current_price = close.iloc[-1]
            current_sma = sma.iloc[-1]
            
            if pd.isna(current_sma):
                continue
                
            # For Long: Price should be above SMA on higher TF
            if signal_direction == 1 and current_price < current_sma:
                return False
            # For Short: Price should be below SMA on higher TF
            elif signal_direction == -1 and current_price > current_sma:
                return False
                
        except Exception:
            # If we can't fetch higher TF, don't block the signal
            continue
            
    return True
