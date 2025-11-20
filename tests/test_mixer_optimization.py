"""
Test for Strategy Mixer Optimization.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from openquant.strategies.mixer import StrategyMixer

class MockStrategy:
    def __init__(self, signals):
        self.signals = signals
        
    def generate_signals(self, df):
        return self.signals

def test_mixer_optimization():
    print("\n--- Testing Mixer Optimization ---")
    
    # 1. Generate Synthetic Data
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="1d")
    # Market goes up steadily
    price = np.linspace(100, 110, n) 
    # Add some noise
    price += np.random.normal(0, 0.1, n)
    
    df = pd.DataFrame({"Close": price}, index=dates)
    
    # 2. Create Strategies
    # Strat A: Always Long (Perfect for this market)
    sig_a = pd.Series(1, index=dates)
    strat_a = MockStrategy(sig_a)
    
    # Strat B: Always Short (Terrible for this market)
    sig_b = pd.Series(-1, index=dates)
    strat_b = MockStrategy(sig_b)
    
    # Strat C: Random
    np.random.seed(42)
    sig_c = pd.Series(np.random.choice([-1, 1], n), index=dates)
    strat_c = MockStrategy(sig_c)
    
    # 3. Initialize Mixer with Equal Weights
    mixer = StrategyMixer([strat_a, strat_b, strat_c])
    print(f"Initial Weights: {mixer.weights}")
    
    # 4. Optimize
    print("Optimizing...")
    mixer.optimize_weights(df)
    
    # 5. Verify Results
    w = mixer.weights
    print(f"Final Weights: {[f'{x:.2f}' for x in w]}")
    
    # Strat A (Long) should have highest weight
    # Strat B (Short) should have lowest weight (near 0)
    assert w[0] > w[1], "Good strategy should have higher weight than bad one"
    assert w[0] > w[2], "Good strategy should have higher weight than random one"
    assert w[1] < 0.1, "Bad strategy should have very low weight"
    
    print("\n✅ Mixer Optimization Test Passed!")

if __name__ == "__main__":
    test_mixer_optimization()
