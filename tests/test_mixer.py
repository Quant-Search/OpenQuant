"""
Test for Strategy Mixer.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from openquant.strategies.mixer import StrategyMixer

class MockStrategy:
    def __init__(self, signal_val):
        self.signal_val = signal_val
        
    def generate_signals(self, df):
        return pd.Series(self.signal_val, index=df.index)

def test_mixer():
    print("\n--- Testing Strategy Mixer ---")
    
    dates = pd.date_range("2024-01-01", periods=5, freq="1d")
    df = pd.DataFrame({"Close": [100]*5}, index=dates)
    
    # Case 1: Consensus Long
    # Strat A (0.5): 1
    # Strat B (0.5): 1
    # Result: 1.0 -> 1
    s1 = MockStrategy(1)
    s2 = MockStrategy(1)
    mixer = StrategyMixer([s1, s2], weights=[0.5, 0.5])
    sig = mixer.generate_signals(df)
    print(f"Consensus Long: {sig.iloc[0]}")
    assert sig.iloc[0] == 1
    
    # Case 2: Conflict (Neutral)
    # Strat A (0.5): 1
    # Strat B (0.5): -1
    # Result: 0.0 -> 0
    s1 = MockStrategy(1)
    s2 = MockStrategy(-1)
    mixer = StrategyMixer([s1, s2], weights=[0.5, 0.5])
    sig = mixer.generate_signals(df)
    print(f"Conflict (Neutral): {sig.iloc[0]}")
    assert sig.iloc[0] == 0
    
    # Case 3: Weighted Dominance
    # Strat A (0.8): 1
    # Strat B (0.2): -1
    # Result: 0.8 - 0.2 = 0.6 -> 1
    s1 = MockStrategy(1)
    s2 = MockStrategy(-1)
    mixer = StrategyMixer([s1, s2], weights=[0.8, 0.2])
    sig = mixer.generate_signals(df)
    print(f"Weighted Dominance: {sig.iloc[0]}")
    assert sig.iloc[0] == 1
    
    print("\n✅ Strategy Mixer Test Passed!")

if __name__ == "__main__":
    test_mixer()
