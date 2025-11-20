from typing import Dict, Any, Optional
import pandas as pd
import pandas_ta as ta
from ..base import BaseStrategy

class PandasTAStrategy(BaseStrategy):
    """Universal wrapper for pandas-ta indicators.
    
    Can wrap almost any indicator from pandas-ta library.
    
    Modes:
    - 'threshold': Signal when indicator crosses a threshold (e.g. RSI < 30).
    - 'crossover': Signal when indicator crosses another indicator or value.
    - 'signal_col': Use a specific column from the indicator result as the signal.
    """
    
    def __init__(self, kind: str, params: Dict[str, Any] = None, mode: str = 'threshold', 
                 buy_at: float = None, sell_at: float = None, 
                 cross_target: str = None):
        self.kind = kind
        self.params = params or {}
        self.mode = mode
        self.buy_at = buy_at
        self.sell_at = sell_at
        self.cross_target = cross_target
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series()
            
        # Run pandas-ta strategy
        # We use the lower level API: df.ta.rsi(...)
        
        try:
            # Dynamically call the indicator method
            method = getattr(df.ta, self.kind)
            result = method(**self.params)
        except Exception as e:
            # Fallback or log error? For now, return empty
            return pd.Series(0, index=df.index)
            
        if result is None:
            return pd.Series(0, index=df.index)

        # Handle multi-column results (e.g. MACD returns MACD, Histogram, Signal)
        if isinstance(result, pd.DataFrame):
            # Default to the first column if not specified
            # Usually the main line is first
            main_col = result.iloc[:, 0]
        else:
            main_col = result
            
        signals = pd.Series(0, index=df.index)
        
        if self.mode == 'threshold':
            if self.buy_at is not None:
                # Buy when crossing UP above buy_at? Or simply being below?
                # Standard mean reversion: Buy if < buy_at (Oversold)
                # Standard trend: Buy if > buy_at (Breakout)
                # Let's assume Mean Reversion for bounded indicators (RSI, STOCH)
                # And Trend for unbounded? 
                # To be safe, we'll use the standard "Oversold/Overbought" logic for now.
                # Buy if < buy_at
                signals[main_col < self.buy_at] = 1
                
            if self.sell_at is not None:
                # Sell if > sell_at
                signals[main_col > self.sell_at] = -1
                
        # TODO: Implement 'crossover' and other modes as needed
        
        return signals

def make_pta_strategy(name: str, **kwargs):
    """Factory to create a PandasTAStrategy from a simple name like 'pta_rsi'."""
    # Remove 'pta_' prefix
    kind = name.replace('pta_', '')
    
    # Extract known params
    buy_at = kwargs.pop('buy_at', None)
    sell_at = kwargs.pop('sell_at', None)
    
    # Default thresholds for common indicators
    if buy_at is None and sell_at is None:
        if kind == 'rsi':
            buy_at, sell_at = 30, 70
        elif kind == 'stoch':
            buy_at, sell_at = 20, 80
        elif kind == 'cci':
            buy_at, sell_at = -100, 100
            
    return PandasTAStrategy(kind, params=kwargs, mode='threshold', buy_at=buy_at, sell_at=sell_at)
