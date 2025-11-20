"""
Portfolio Rebalancing Module.
Monitors asset allocation drift and generates rebalancing orders.
"""
from typing import Dict, List, Tuple, Any
import pandas as pd

class Rebalancer:
    """
    Manages portfolio rebalancing logic.
    """
    def __init__(self, drift_threshold: float = 0.1):
        """
        Args:
            drift_threshold: Allowable deviation from target weight (e.g., 0.1 = 10%).
                             If target is 0.5, rebalance if current < 0.45 or > 0.55?
                             Or relative: |curr - target| / target > 0.1?
                             Let's use absolute deviation for simplicity first, or relative.
                             Common practice: Absolute deviation bands (e.g. +/- 5%).
        """
        self.drift_threshold = drift_threshold

    def check_drift(self, 
                   current_holdings: Dict[str, float], 
                   prices: Dict[str, float], 
                   target_weights: Dict[str, float],
                   cash: float) -> List[str]:
        """
        Check if any asset has drifted beyond the threshold.
        Returns list of symbols that need rebalancing.
        """
        # 1. Calculate Total Portfolio Value
        pos_val = sum(current_holdings.get(sym, 0.0) * prices.get(sym, 0.0) for sym in current_holdings)
        total_equity = cash + pos_val
        
        if total_equity <= 0:
            return []

        drifted_symbols = []
        
        # 2. Check each target
        all_symbols = set(current_holdings.keys()) | set(target_weights.keys())
        
        for sym in all_symbols:
            target = target_weights.get(sym, 0.0)
            
            # Current Value
            units = current_holdings.get(sym, 0.0)
            price = prices.get(sym, 0.0)
            curr_val = units * price
            curr_weight = curr_val / total_equity
            
            # Deviation
            deviation = abs(curr_weight - target)
            
            # Check threshold (Absolute deviation)
            # If target is 0.2, and threshold is 0.05, we rebalance if weight < 0.15 or > 0.25
            if deviation > self.drift_threshold:
                drifted_symbols.append(sym)
                
        return drifted_symbols

    def generate_rebalancing_orders(self, 
                                   current_holdings: Dict[str, float], 
                                   prices: Dict[str, float], 
                                   target_weights: Dict[str, float], 
                                   cash: float) -> List[Dict[str, Any]]:
        """
        Generate orders to restore target weights.
        """
        pos_val = sum(current_holdings.get(sym, 0.0) * prices.get(sym, 0.0) for sym in current_holdings)
        total_equity = cash + pos_val
        
        if total_equity <= 0:
            return []
            
        orders = []
        all_symbols = set(current_holdings.keys()) | set(target_weights.keys())
        
        for sym in all_symbols:
            target = target_weights.get(sym, 0.0)
            price = prices.get(sym, 0.0)
            
            if price <= 0:
                continue
                
            # Target Value
            target_val = total_equity * target
            
            # Current Value
            curr_units = current_holdings.get(sym, 0.0)
            curr_val = curr_units * price
            
            # Delta
            delta_val = target_val - curr_val
            delta_units = delta_val / price
            
            if abs(delta_units) > 0:
                orders.append({
                    "symbol": sym,
                    "delta_units": delta_units,
                    "current_units": curr_units,
                    "target_units": curr_units + delta_units,
                    "price": price
                })
                
        return orders
