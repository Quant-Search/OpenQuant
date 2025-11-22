"""
Tests for MLStrategy.
"""
import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from openquant.strategies.ml_strategy import MLStrategy

class TestMLStrategy(unittest.TestCase):
    def setUp(self):
        """Create synthetic OHLCV data"""
        np.random.seed(42)
        n = 600
        dates = pd.date_range('2024-01-01', periods=n, freq='1h')
        
        # Synthetic price with slight upward drift
        ret = np.random.randn(n) * 0.01 + 0.0001
        price = 100 * np.exp(np.cumsum(ret))
        
        self.df = pd.DataFrame({
            'Open': price * (1 + np.random.randn(n) * 0.001),
            'High': price * (1 + np.abs(np.random.randn(n)) * 0.005),
            'Low': price * (1 - np.abs(np.random.randn(n)) * 0.005),
            'Close': price,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=dates)
        
    def test_feature_generation(self):
        """Test that features are generated correctly"""
        strategy = MLStrategy(lookback=100, retrain_interval=50)
        features = strategy._generate_features(self.df)
        
        # Check that features exist
        self.assertFalse(features.empty)
        
        # Check that there are no NaNs after dropna
        self.assertEqual(features.isna().sum().sum(), 0)
        
        # Check feature columns
        expected_cols = ['ret_lag1', 'ret_lag2', 'ret_lag5', 'vol_20', 'vol_50', 
                        'efficiency_ratio', 'vol_imb']
        for col in expected_cols:
            self.assertIn(col, features.columns)
            
    def test_signal_generation(self):
        """Test that signals are generated"""
        strategy = MLStrategy(
            model=RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42),
            lookback=200,
            retrain_interval=50,
            probability_threshold=0.55
        )
        
        signals = strategy.generate_signals(self.df)
        
        # Check that signals are generated
        self.assertEqual(len(signals), len(self.df))
        
        # Check that signals are in {-1, 0, 1}
        unique_sigs = signals.unique()
        for sig in unique_sigs:
            self.assertIn(sig, [-1, 0, 1])
            
        # Check that we have some non-zero signals (if the model learned anything)
        # This may fail if the synthetic data is too random, but should generally work
        self.assertTrue((signals != 0).sum() > 0, "No signals generated - model may need tuning")
        
    def test_no_lookahead_bias(self):
        """Verify that features at time T do not depend on data > T"""
        strategy = MLStrategy(lookback=100)
        
        # Take a subset of data
        df_subset = self.df.iloc[:300].copy()
        features_subset = strategy._generate_features(df_subset)
        
        # Take full data
        features_full = strategy._generate_features(self.df)
        
        # Features computed on subset should match features computed on full data
        # for the overlapping period
        common_index = features_subset.index.intersection(features_full.index)
        
        for col in features_subset.columns:
            subset_vals = features_subset.loc[common_index, col]
            full_vals = features_full.loc[common_index, col]
            
            # Allow small numerical errors
            np.testing.assert_allclose(
                subset_vals.values, 
                full_vals.values, 
                rtol=1e-5,
                err_msg=f"Feature {col} has look-ahead bias"
            )

if __name__ == "__main__":
    unittest.main()
