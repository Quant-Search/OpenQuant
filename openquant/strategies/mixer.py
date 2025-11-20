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
        Uses scipy.optimize to maximize Portfolio Sharpe Ratio.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            print("scipy not installed, skipping optimization.")
            return

        # 1. Get returns for each strategy
        returns_list = []
        valid_indices = []
        
        # Calculate market returns once
        market_ret = df['Close'].pct_change().fillna(0)
        
        for i, strat in enumerate(self.strategies):
            try:
                # Generate signals
                sig = strat.generate_signals(df)
                # Ensure signal is -1, 0, 1
                sig = sig.clip(-1, 1).fillna(0)
                
                # Calculate strategy returns (lagged signal * return)
                # Signal at t determines position at t+1
                strat_ret = sig.shift(1).fillna(0) * market_ret
                
                returns_list.append(strat_ret)
                valid_indices.append(i)
            except Exception as e:
                print(f"Strategy {i} optimization failed: {e}")
        
        if not returns_list:
            return

        # 2. Define Objective Function (Negative Sharpe)
        # Align all returns to same index (should be already)
        returns_df = pd.DataFrame(returns_list).T.fillna(0)
        
        def neg_sharpe(weights):
            # Portfolio Return
            port_ret = (returns_df * weights).sum(axis=1)
            
            mean_ret = port_ret.mean()
            std_ret = port_ret.std()
            
            if std_ret == 0:
                return 0.0
            
            # Maximize Sharpe => Minimize Negative Sharpe
            return - (mean_ret / std_ret)

        # 3. Optimize
        n = len(returns_list)
        # Constraints: Sum(weights) = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Bounds: 0 <= weight <= 1 (Long only weights, no shorting strategies themselves)
        bounds = tuple((0, 1) for _ in range(n))
        initial_weights = [1./n] * n
        
        try:
            result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                # Map back to full list of strategies
                new_weights = [0.0] * len(self.strategies)
                for idx, weight in zip(valid_indices, result.x):
                    new_weights[idx] = max(0.0, float(weight)) # Ensure no negative dust
                
                # Normalize again just in case
                s = sum(new_weights)
                if s > 0:
                    self.weights = [w/s for w in new_weights]
                
                print(f"Optimized Weights: {[f'{w:.2f}' for w in self.weights]}")
            else:
                print(f"Optimization failed: {result.message}")
        except Exception as e:
            print(f"Optimization error: {e}")
