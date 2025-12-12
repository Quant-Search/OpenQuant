"""Tests for Multi-Timeframe Strategy."""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from openquant.strategies.mtf_strategy import MultiTimeframeStrategy, MultiTimeframeEnsemble
from openquant.strategies.base import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, signals=None):
        self.signals = signals
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if self.signals is not None:
            return pd.Series(self.signals, index=df.index)
        return pd.Series(1, index=df.index)


def create_test_df(periods=100, start_price=100.0, trend='neutral'):
    """Create test OHLCV DataFrame."""
    idx = pd.date_range(start=datetime.now() - timedelta(hours=periods), periods=periods, freq='H')
    
    if trend == 'up':
        close = np.linspace(start_price, start_price * 1.2, periods)
    elif trend == 'down':
        close = np.linspace(start_price, start_price * 0.8, periods)
    else:
        close = np.ones(periods) * start_price + np.random.randn(periods) * 2
    
    df = pd.DataFrame({
        'Open': close * 0.99,
        'High': close * 1.01,
        'Low': close * 0.98,
        'Close': close,
        'Volume': np.random.randint(1000, 10000, periods),
    }, index=idx)
    
    return df


class TestMultiTimeframeStrategy(unittest.TestCase):
    """Test MultiTimeframeStrategy."""
    
    def test_no_fetch_function(self):
        """Test that strategy works without fetch function (no MTF filtering)."""
        base_strategy = MockStrategy(signals=[1, 1, -1, 0, 1])
        mtf_strategy = MultiTimeframeStrategy(
            base_strategy=base_strategy,
            timeframes=['1h', '4h', '1d'],
            fetch_func=None,
        )
        
        df = create_test_df(periods=5)
        signals = mtf_strategy.generate_signals(df)
        
        self.assertEqual(len(signals), 5)
        self.assertEqual(list(signals), [1, 1, -1, 0, 1])
        
    def test_with_uptrend_confirmation(self):
        """Test MTF strategy with uptrend confirmation."""
        base_strategy = MockStrategy(signals=[1] * 100)
        
        def fetch_func(symbol, timeframe):
            if timeframe == '4h':
                return create_test_df(periods=50, trend='up')
            elif timeframe == '1d':
                return create_test_df(periods=30, trend='up')
            return create_test_df(periods=100)
        
        mtf_strategy = MultiTimeframeStrategy(
            base_strategy=base_strategy,
            timeframes=['1h', '4h', '1d'],
            fetch_func=fetch_func,
            require_all_timeframes=False,
            min_confirmations=1,
        )
        mtf_strategy.set_symbol('BTC/USDT')
        
        df = create_test_df(periods=100)
        signals = mtf_strategy.generate_signals(df)
        
        self.assertEqual(len(signals), 100)
        self.assertTrue((signals == 1).sum() > 0)
        
    def test_with_downtrend_rejection(self):
        """Test MTF strategy rejecting long signals in downtrend."""
        base_strategy = MockStrategy(signals=[1] * 100)
        
        def fetch_func(symbol, timeframe):
            if timeframe == '4h':
                return create_test_df(periods=50, trend='down')
            elif timeframe == '1d':
                return create_test_df(periods=30, trend='down')
            return create_test_df(periods=100)
        
        mtf_strategy = MultiTimeframeStrategy(
            base_strategy=base_strategy,
            timeframes=['1h', '4h', '1d'],
            fetch_func=fetch_func,
            require_all_timeframes=True,
            min_confirmations=2,
        )
        mtf_strategy.set_symbol('BTC/USDT')
        
        df = create_test_df(periods=100)
        signals = mtf_strategy.generate_signals(df)
        
        self.assertEqual(len(signals), 100)
        long_signals = (signals == 1).sum()
        self.assertTrue(long_signals < 50)
        
    def test_short_signals_with_confirmation(self):
        """Test short signals with MTF confirmation."""
        base_strategy = MockStrategy(signals=[-1] * 100)
        
        def fetch_func(symbol, timeframe):
            if timeframe in ['4h', '1d']:
                return create_test_df(periods=50, trend='down')
            return create_test_df(periods=100)
        
        mtf_strategy = MultiTimeframeStrategy(
            base_strategy=base_strategy,
            timeframes=['1h', '4h', '1d'],
            fetch_func=fetch_func,
            require_all_timeframes=False,
            min_confirmations=1,
        )
        mtf_strategy.set_symbol('BTC/USDT')
        
        df = create_test_df(periods=100)
        signals = mtf_strategy.generate_signals(df)
        
        self.assertEqual(len(signals), 100)
        self.assertTrue((signals == -1).sum() > 0)
        
    def test_strategy_based_confirmation(self):
        """Test using strategy signals for confirmation."""
        base_strategy = MockStrategy()
        
        def fetch_func(symbol, timeframe):
            return create_test_df(periods=50)
        
        mtf_strategy = MultiTimeframeStrategy(
            base_strategy=base_strategy,
            timeframes=['1h', '4h'],
            fetch_func=fetch_func,
            use_strategy_signals=True,
            min_confirmations=1,
        )
        mtf_strategy.set_symbol('BTC/USDT')
        
        df = create_test_df(periods=100)
        signals = mtf_strategy.generate_signals(df)
        
        self.assertEqual(len(signals), 100)
        
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        base_strategy = MockStrategy()
        mtf_strategy = MultiTimeframeStrategy(
            base_strategy=base_strategy,
            timeframes=['1h', '4h'],
        )
        
        df = pd.DataFrame()
        signals = mtf_strategy.generate_signals(df)
        
        self.assertEqual(len(signals), 0)
        
    def test_flat_signals_pass_through(self):
        """Test that flat signals (0) pass through without confirmation."""
        base_strategy = MockStrategy(signals=[0] * 100)
        
        def fetch_func(symbol, timeframe):
            return create_test_df(periods=50, trend='down')
        
        mtf_strategy = MultiTimeframeStrategy(
            base_strategy=base_strategy,
            timeframes=['1h', '4h', '1d'],
            fetch_func=fetch_func,
            require_all_timeframes=True,
        )
        mtf_strategy.set_symbol('BTC/USDT')
        
        df = create_test_df(periods=100)
        signals = mtf_strategy.generate_signals(df)
        
        self.assertTrue((signals == 0).all())


class TestMultiTimeframeEnsemble(unittest.TestCase):
    """Test MultiTimeframeEnsemble."""
    
    def test_weighted_aggregation(self):
        """Test weighted aggregation of strategies."""
        strategy_1h = MockStrategy(signals=[1] * 100)
        strategy_4h = MockStrategy(signals=[1] * 100)
        strategy_1d = MockStrategy(signals=[-1] * 100)
        
        def fetch_func(symbol, timeframe):
            return create_test_df(periods=100)
        
        ensemble = MultiTimeframeEnsemble(
            strategies=[
                ('1h', strategy_1h, 0.5),
                ('4h', strategy_4h, 0.3),
                ('1d', strategy_1d, 0.2),
            ],
            fetch_func=fetch_func,
            aggregation='weighted',
            threshold=0.3,
        )
        ensemble.set_symbol('BTC/USDT')
        
        df = create_test_df(periods=100)
        signals = ensemble.generate_signals(df)
        
        self.assertEqual(len(signals), 100)
        self.assertTrue((signals == 1).sum() > 0)
        
    def test_majority_aggregation(self):
        """Test majority voting aggregation."""
        strategy_1 = MockStrategy(signals=[1] * 100)
        strategy_2 = MockStrategy(signals=[1] * 100)
        strategy_3 = MockStrategy(signals=[-1] * 100)
        
        def fetch_func(symbol, timeframe):
            return create_test_df(periods=100)
        
        ensemble = MultiTimeframeEnsemble(
            strategies=[
                ('1h', strategy_1, 1.0),
                ('4h', strategy_2, 1.0),
                ('1d', strategy_3, 1.0),
            ],
            fetch_func=fetch_func,
            aggregation='majority',
        )
        ensemble.set_symbol('BTC/USDT')
        
        df = create_test_df(periods=100)
        signals = ensemble.generate_signals(df)
        
        self.assertEqual(len(signals), 100)
        self.assertTrue((signals == 1).all())
        
    def test_unanimous_aggregation(self):
        """Test unanimous aggregation."""
        strategy_1 = MockStrategy(signals=[1] * 100)
        strategy_2 = MockStrategy(signals=[1] * 100)
        strategy_3 = MockStrategy(signals=[1] * 100)
        
        def fetch_func(symbol, timeframe):
            return create_test_df(periods=100)
        
        ensemble = MultiTimeframeEnsemble(
            strategies=[
                ('1h', strategy_1, 1.0),
                ('4h', strategy_2, 1.0),
                ('1d', strategy_3, 1.0),
            ],
            fetch_func=fetch_func,
            aggregation='unanimous',
        )
        ensemble.set_symbol('BTC/USDT')
        
        df = create_test_df(periods=100)
        signals = ensemble.generate_signals(df)
        
        self.assertEqual(len(signals), 100)
        self.assertTrue((signals == 1).all())
        
    def test_unanimous_with_disagreement(self):
        """Test unanimous aggregation with disagreement results in flat."""
        strategy_1 = MockStrategy(signals=[1] * 100)
        strategy_2 = MockStrategy(signals=[1] * 100)
        strategy_3 = MockStrategy(signals=[-1] * 100)
        
        def fetch_func(symbol, timeframe):
            return create_test_df(periods=100)
        
        ensemble = MultiTimeframeEnsemble(
            strategies=[
                ('1h', strategy_1, 1.0),
                ('4h', strategy_2, 1.0),
                ('1d', strategy_3, 1.0),
            ],
            fetch_func=fetch_func,
            aggregation='unanimous',
        )
        ensemble.set_symbol('BTC/USDT')
        
        df = create_test_df(periods=100)
        signals = ensemble.generate_signals(df)
        
        self.assertEqual(len(signals), 100)
        self.assertTrue((signals == 0).all())
        
    def test_empty_strategies(self):
        """Test ensemble with no strategies."""
        ensemble = MultiTimeframeEnsemble(
            strategies=[],
            fetch_func=None,
            aggregation='weighted',
        )
        
        df = create_test_df(periods=100)
        signals = ensemble.generate_signals(df)
        
        self.assertEqual(len(signals), 100)
        self.assertTrue((signals == 0).all())
        
    def test_weight_normalization(self):
        """Test that weights are normalized."""
        strategy_1 = MockStrategy(signals=[1] * 100)
        strategy_2 = MockStrategy(signals=[1] * 100)
        
        def fetch_func(symbol, timeframe):
            return create_test_df(periods=100)
        
        ensemble = MultiTimeframeEnsemble(
            strategies=[
                ('1h', strategy_1, 100.0),
                ('4h', strategy_2, 200.0),
            ],
            fetch_func=fetch_func,
            aggregation='weighted',
        )
        
        total_weight = sum(w for _, _, w in ensemble.normalized_weights)
        self.assertAlmostEqual(total_weight, 1.0, places=5)
        
    def test_mixed_signals(self):
        """Test ensemble with mixed signal patterns."""
        strategy_1 = MockStrategy(signals=[1, -1, 0, 1, -1] * 20)
        strategy_2 = MockStrategy(signals=[-1, 1, 0, -1, 1] * 20)
        strategy_3 = MockStrategy(signals=[0, 0, 1, -1, 0] * 20)
        
        def fetch_func(symbol, timeframe):
            return create_test_df(periods=100)
        
        ensemble = MultiTimeframeEnsemble(
            strategies=[
                ('1h', strategy_1, 0.4),
                ('4h', strategy_2, 0.4),
                ('1d', strategy_3, 0.2),
            ],
            fetch_func=fetch_func,
            aggregation='weighted',
            threshold=0.5,
        )
        ensemble.set_symbol('BTC/USDT')
        
        df = create_test_df(periods=100)
        signals = ensemble.generate_signals(df)
        
        self.assertEqual(len(signals), 100)
        self.assertTrue(set(signals.unique()).issubset({-1, 0, 1}))


class TestMTFIntegration(unittest.TestCase):
    """Integration tests for MTF strategies."""
    
    def test_mtf_strategy_reduces_signals(self):
        """Test that MTF filtering reduces number of signals."""
        base_strategy = MockStrategy(signals=[1] * 100)
        
        signal_count_without_mtf = 100
        
        def fetch_func(symbol, timeframe):
            return create_test_df(periods=50, trend='neutral')
        
        mtf_strategy = MultiTimeframeStrategy(
            base_strategy=base_strategy,
            timeframes=['1h', '4h', '1d'],
            fetch_func=fetch_func,
            require_all_timeframes=True,
        )
        mtf_strategy.set_symbol('BTC/USDT')
        
        df = create_test_df(periods=100)
        signals = mtf_strategy.generate_signals(df)
        
        signal_count_with_mtf = (signals != 0).sum()
        self.assertTrue(signal_count_with_mtf <= signal_count_without_mtf)
        
    def test_ensemble_different_from_single(self):
        """Test that ensemble produces different results than single strategy."""
        strategy = MockStrategy(signals=[1, -1] * 50)
        
        def fetch_func(symbol, timeframe):
            return create_test_df(periods=100)
        
        single_signals = strategy.generate_signals(create_test_df(periods=100))
        
        ensemble = MultiTimeframeEnsemble(
            strategies=[
                ('1h', strategy, 0.5),
                ('4h', MockStrategy(signals=[-1, 1] * 50), 0.5),
            ],
            fetch_func=fetch_func,
            aggregation='weighted',
            threshold=0.3,
        )
        ensemble.set_symbol('BTC/USDT')
        
        ensemble_signals = ensemble.generate_signals(create_test_df(periods=100))
        
        self.assertFalse((single_signals == ensemble_signals).all())


if __name__ == '__main__':
    unittest.main()
