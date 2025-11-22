import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from openquant.strategies.mixer import StrategyMixer

class MockStrategy:
    def __init__(self, signals):
        self.signals = signals
        
    def generate_signals(self, df):
        return pd.Series(self.signals, index=df.index)

class TestMixer(unittest.TestCase):
    def test_weighted_voting(self):
        # 3 data points
        idx = pd.date_range("2025-01-01", periods=3)
        df = pd.DataFrame({"Close": [100, 101, 102]}, index=idx)
        
        # Strat 1: Buy, Buy, Buy
        s1 = MockStrategy([1, 1, 1])
        # Strat 2: Sell, Sell, Sell
        s2 = MockStrategy([-1, -1, -1])
        # Strat 3: Neutral, Buy, Sell
        s3 = MockStrategy([0, 1, -1])
        
        # Case 1: Equal Weights (1/3 each)
        # P1: 1 - 1 + 0 = 0 -> Neutral
        # P2: 1 - 1 + 1 = 1 -> Buy (0.33 > 0.2)
        # P3: 1 - 1 - 1 = -1 -> Sell (-0.33 < -0.2)
        
        mixer = StrategyMixer([s1, s2, s3]) # weights=None -> Equal
        sigs = mixer.generate_signals(df)
        
        self.assertEqual(sigs.iloc[0], 0)
        self.assertEqual(sigs.iloc[1], 1)
        self.assertEqual(sigs.iloc[2], -1)
        
        # Case 2: Weighted (S1 has 0.8, S2 0.1, S3 0.1)
        # P1: 0.8*1 + 0.1*-1 + 0.1*0 = 0.7 -> Buy
        mixer_w = StrategyMixer([s1, s2, s3], weights=[0.8, 0.1, 0.1])
        sigs_w = mixer_w.generate_signals(df)
        self.assertEqual(sigs_w.iloc[0], 1)

if __name__ == "__main__":
    unittest.main()
