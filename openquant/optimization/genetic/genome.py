from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import random
import pandas as pd
import numpy as np
from openquant.quant.microstructure import vpin
from openquant.quant.volatility import parkinson_volatility

# --- Gene Definitions ---

class Gene(ABC):
    """Base class for all genes."""
    
    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> Union[pd.Series, float]:
        """Evaluate the gene on the given DataFrame."""
        pass

    @abstractmethod
    def mutate(self) -> None:
        """Randomly mutate the gene's parameters."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

class ValueGene(Gene):
    """Represents a constant value (e.g., 0.5, 2.0)."""
    def __init__(self, value: float, min_val: float = -100.0, max_val: float = 100.0):
        self.value = value
        self.min_val = min_val
        self.max_val = max_val

    def evaluate(self, df: pd.DataFrame) -> float:
        return self.value

    def mutate(self) -> None:
        # Mutate by small random amount or reset
        if random.random() < 0.5:
            self.value += random.uniform(-0.1 * self.value, 0.1 * self.value)
        else:
            self.value = random.uniform(self.min_val, self.max_val)
        self.value = max(self.min_val, min(self.max_val, self.value))

    def __str__(self) -> str:
        return f"{self.value:.2f}"

class QuantGene(Gene):
    """Represents a Quantitative Metric (Vectorized)."""
    def __init__(self, metric: str, params: Dict[str, Any]):
        self.metric = metric
        self.params = params
        
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        
        if self.metric == 'efficiency_ratio':
            # Kaufman Efficiency Ratio (Fractal Efficiency)
            # Proxy for Hurst / Trendiness
            # ER = |Change| / Sum(|Changes|)
            period = int(self.params.get('period', 20))
            change = close.diff(period).abs()
            volatility = close.diff().abs().rolling(period).sum()
            er = change / volatility
            return er.fillna(0.0)
            
        elif self.metric == 'vpin':
            # Flow Toxicity
            # We use the vpin function but we need to ensure it returns a Series aligned with df
            # The vpin function in microstructure.py returns a Series aligned to buckets or bars?
            # My implementation returns aligned to bars (shifted).
            window = int(self.params.get('window', 50))
            # Estimate bucket volume dynamically if not passed?
            # The vpin function handles it.
            return vpin(df, window_buckets=window).fillna(0.5)
            
        elif self.metric == 'volatility':
            # Parkinson Volatility
            window = int(self.params.get('window', 20))
            return parkinson_volatility(df['High'], df['Low'], window=window).fillna(0.0)
            
        elif self.metric == 'zscore':
            # Rolling Z-Score (Mean Reversion)
            # (Close - Mean) / Std
            window = int(self.params.get('window', 20))
            roll = close.rolling(window)
            z = (close - roll.mean()) / roll.std()
            return z.fillna(0.0)
            
        elif self.metric == 'return':
            # Simple momentum
            period = int(self.params.get('period', 1))
            return close.pct_change(period).fillna(0.0)

        return pd.Series(0, index=df.index)

    def mutate(self) -> None:
        # Mutate parameters
        if 'period' in self.params:
            current = self.params['period']
            change = int(random.choice([-1, 1]) * max(1, current * 0.1))
            self.params['period'] = max(2, current + change)
        if 'window' in self.params:
            current = self.params['window']
            change = int(random.choice([-1, 1]) * max(1, current * 0.1))
            self.params['window'] = max(5, current + change)

    def __str__(self) -> str:
        param_str = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.metric}({param_str})"

class ComparisonGene(Gene):
    """Represents a comparison (e.g., Metric < Value)."""
    def __init__(self, left: Gene, operator: str, right: Gene):
        self.left = left
        self.operator = operator # '>', '<', '>=', '<='
        self.right = right

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l_val = self.left.evaluate(df)
        r_val = self.right.evaluate(df)
        
        if self.operator == '>':
            return l_val > r_val
        elif self.operator == '<':
            return l_val < r_val
        elif self.operator == '>=':
            return l_val >= r_val
        elif self.operator == '<=':
            return l_val <= r_val
        return pd.Series(False, index=df.index)

    def mutate(self) -> None:
        r = random.random()
        if r < 0.33:
            self.operator = random.choice(['>', '<', '>=', '<='])
        elif r < 0.66:
            self.left.mutate()
        else:
            self.right.mutate()

    def __str__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"

class LogicGene(Gene):
    """Represents a logical operation (e.g., Cond1 AND Cond2)."""
    def __init__(self, left: Gene, operator: str, right: Gene):
        self.left = left
        self.operator = operator # 'AND', 'OR'
        self.right = right

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        l_val = self.left.evaluate(df)
        r_val = self.right.evaluate(df)
        
        if self.operator == 'AND':
            return l_val & r_val
        elif self.operator == 'OR':
            return l_val | r_val
        return pd.Series(False, index=df.index)

    def mutate(self) -> None:
        if random.random() < 0.33:
            self.operator = 'OR' if self.operator == 'AND' else 'AND'
        elif random.random() < 0.66:
            self.left.mutate()
        else:
            self.right.mutate()

    def __str__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"

# --- Genome Definition ---

class Genome:
    """Represents a complete strategy genome."""
    def __init__(self, entry_condition: Gene):
        self.entry_condition = entry_condition
        self.fitness: float = 0.0

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals: 1 where entry_condition is True, 0 otherwise."""
        signals = self.entry_condition.evaluate(df)
        return signals.astype(int)

    def mutate(self) -> None:
        self.entry_condition.mutate()

    def crossover(self, other: Genome) -> Genome:
        """Create a new genome by combining this one with another."""
        import copy
        # Start with a copy of self
        child_entry = copy.deepcopy(self.entry_condition)
        
        # Attempt to swap subtrees if compatible
        # We assume the standard structure: LogicGene(Left, Op, Right)
        if isinstance(child_entry, LogicGene) and isinstance(other.entry_condition, LogicGene):
            # 50% chance to swap Left (Regime), 50% to swap Right (Signal)
            if random.random() < 0.5:
                child_entry.left = copy.deepcopy(other.entry_condition.left)
            else:
                child_entry.right = copy.deepcopy(other.entry_condition.right)
        
        return Genome(child_entry)

    def __str__(self) -> str:
        return f"ENTRY: {self.entry_condition}"

# --- Factory ---

def generate_random_genome() -> Genome:
    """Generate a random Quant genome."""
    # Template: (Regime Check) AND (Signal)
    
    # 1. Regime Check: Efficiency Ratio (Trend) or VPIN (Toxicity)
    # e.g. ER > 0.3 (Trending) OR VPIN < 0.3 (Safe)
    
    metric_type = random.choice(['efficiency_ratio', 'vpin', 'volatility'])
    if metric_type == 'efficiency_ratio':
        # Trendiness
        metric = QuantGene('efficiency_ratio', {'period': random.randint(10, 50)})
        val = ValueGene(random.uniform(0.2, 0.6), 0.0, 1.0)
        op = random.choice(['>', '<']) # > means Trend, < means Range
        cond1 = ComparisonGene(metric, op, val)
    elif metric_type == 'vpin':
        # Toxicity
        metric = QuantGene('vpin', {'window': random.randint(20, 100)})
        val = ValueGene(random.uniform(0.2, 0.4), 0.0, 1.0)
        cond1 = ComparisonGene(metric, '<', val) # Usually avoid toxic flow
    else:
        # Volatility
        metric = QuantGene('volatility', {'window': random.randint(10, 30)})
        val = ValueGene(random.uniform(0.005, 0.02), 0.0, 0.1)
        cond1 = ComparisonGene(metric, '>', val) # High vol?
        
    # 2. Signal: Z-Score (Mean Reversion) or Return (Momentum)
    signal_type = random.choice(['zscore', 'return'])
    if signal_type == 'zscore':
        metric = QuantGene('zscore', {'window': random.randint(10, 50)})
        val = ValueGene(random.uniform(1.5, 3.0), 0.0, 5.0)
        # Z < -2 (Buy Dip) or Z > 2 (Short Spike - but we only do Longs here? 
        # If we only do Longs (1), we want Z < -Val (Buy Dip)
        cond2 = ComparisonGene(metric, '<', ValueGene(-1 * val.value, -5.0, 0.0))
    else:
        # Momentum
        metric = QuantGene('return', {'period': random.randint(1, 10)})
        val = ValueGene(0.0, -0.1, 0.1)
        cond2 = ComparisonGene(metric, '>', val) # Ret > 0
        
    # Combine
    root = LogicGene(cond1, 'AND', cond2)
    
    return Genome(root)
