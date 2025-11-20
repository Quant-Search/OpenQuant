"""
Strategy Mixer.
Combines multiple strategies into a single portfolio strategy using weighted voting.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from openquant.strategies.registry import make_strategy

class StrategyMixer:
    """
    Combines signals from multiple strategies.
    """
    def __init__(self, strategies: List[Any], weights: List[float] = None):
        """
        Args:
            strategies: List of instantiated strategy objects (must have generate_signals method).
            weights: List of weights for each strategy. If None, equal weights are used.
        """
        self.strategies = strategies
        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            # Normalize weights
            s = sum(weights)
            if s == 0:
                self.weights = [1.0 / len(strategies)] * len(strategies)
            else:
                self.weights = [w / s for w in weights]
                
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate combined signals.
        Signal = sign(Weighted Sum of individual signals)
        """
        if not self.strategies:
            return pd.Series(0, index=df.index)
            
        # Collect signals
        combined_signal = pd.Series(0.0, index=df.index)
        
        for i, strat in enumerate(self.strategies):
            try:
                sig = strat.generate_signals(df)
                # Ensure signal is -1, 0, 1
                sig = sig.clip(-1, 1).fillna(0)
                combined_signal += sig * self.weights[i]
            except Exception as e:
                # Log error but continue?
                print(f"Strategy {i} failed: {e}")
                
        # Thresholding
        # If weighted sum > 0.3 -> Long (1)
        # If weighted sum < -0.3 -> Short (-1)
        # Else -> Flat (0)
        # The threshold depends on how aggressive we want to be.
        # 0.0 means any positive consensus is a buy.
        threshold = 0.2 
        
        final_signal = pd.Series(0, index=df.index)
        final_signal[combined_signal > threshold] = 1
        final_signal[combined_signal < -threshold] = -1
        
        return final_signal

    def optimize_weights(self, df: pd.DataFrame):
        """
        Find optimal weights based on historical performance (Sharpe).
        Simple Monte Carlo or Scipy Optimize.
        """
        # Placeholder for future implementation
        pass
