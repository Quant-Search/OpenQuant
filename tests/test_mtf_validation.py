"""Tests for Multi-Timeframe Validation and Regime Filtering."""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openquant.validation.mtf_filter import (
    check_mtf_confirmation,
    check_regime_filter,
    calculate_trend_strength,
    calculate_volatility_regime,
    calculate_range_regime,
    calculate_atr,
    check_multi_regime_alignment,
    get_regime_score,
)


def create_test_df(periods=100, start_price=100.0, trend='neutral', volatility='normal'):
    """Create test OHLCV DataFrame."""
    idx = pd.date_range(
        start=datetime.now() - timedelta(hours=periods),
        periods=periods,
        freq='H'
    )
    
    if trend == 'up':
        close = np.linspace(start_price, start_price * 1.3, periods)
    elif trend == 'down':
        close = np.linspace(start_price, start_price * 0.7, periods)
    elif trend == 'sideways':
        close = np.ones(periods) * start_price + np.sin(np.linspace(0, 4*np.pi, periods)) * 2
    else:
        close = np.ones(periods) * start_price + np.random.randn(periods) * 2
    
    if volatility == 'high':
        noise = np.random.randn(periods) * 5
    elif volatility == 'low':
        noise = np.random.randn(periods) * 0.5
    else:
        noise = np.random.randn(periods) * 2
    
    close = close + noise
    
    df = pd.DataFrame({
        'Open': close * 0.995,
        'High': close * 1.005 + np.abs(np.random.randn(periods) * 0.5),
        'Low': close * 0.995 - np.abs(np.random.randn(periods) * 0.5),
        'Close': close,
        'Volume': np.random.randint(1000, 10000, periods),
    }, index=idx)
    
    return df


class TestMTFConfirmation(unittest.TestCase):
    """Test MTF confirmation checks."""
    
    def test_flat_signal_always_confirms(self):
        """Test that flat signals (0) always confirm."""
        def fetch_func(symbol, timeframe):
            return create_test_df()
        
        result = check_mtf_confirmation('BTC/USDT', '1h', 0, fetch_func)
        self.assertTrue(result)
        
    def test_long_signal_with_uptrend(self):
        """Test long signal with uptrending higher timeframes."""
        def fetch_func(symbol, timeframe):
            if timeframe in ['4h', '1d']:
                return create_test_df(trend='up')
            return create_test_df()
        
        result = check_mtf_confirmation('BTC/USDT', '1h', 1, fetch_func)
        self.assertTrue(result)
        
    def test_long_signal_with_downtrend(self):
        """Test long signal rejection with downtrending higher timeframes."""
        def fetch_func(symbol, timeframe):
            if timeframe in ['4h', '1d']:
                return create_test_df(trend='down')
            return create_test_df()
        
        result = check_mtf_confirmation('BTC/USDT', '1h', 1, fetch_func)
        self.assertFalse(result)
        
    def test_short_signal_with_downtrend(self):
        """Test short signal with downtrending higher timeframes."""
        def fetch_func(symbol, timeframe):
            if timeframe in ['4h', '1d']:
                return create_test_df(trend='down')
            return create_test_df()
        
        result = check_mtf_confirmation('BTC/USDT', '1h', -1, fetch_func)
        self.assertTrue(result)
        
    def test_short_signal_with_uptrend(self):
        """Test short signal rejection with uptrending higher timeframes."""
        def fetch_func(symbol, timeframe):
            if timeframe in ['4h', '1d']:
                return create_test_df(trend='up')
            return create_test_df()
        
        result = check_mtf_confirmation('BTC/USDT', '1h', -1, fetch_func)
        self.assertFalse(result)
        
    def test_unknown_timeframe(self):
        """Test handling of unknown timeframe."""
        def fetch_func(symbol, timeframe):
            return create_test_df()
        
        result = check_mtf_confirmation('BTC/USDT', 'unknown', 1, fetch_func)
        self.assertTrue(result)
        
    def test_empty_higher_timeframe_data(self):
        """Test handling of empty higher timeframe data."""
        def fetch_func(symbol, timeframe):
            if timeframe in ['4h', '1d']:
                return pd.DataFrame()
            return create_test_df()
        
        result = check_mtf_confirmation('BTC/USDT', '1h', 1, fetch_func)
        self.assertTrue(result)


class TestRegimeFilters(unittest.TestCase):
    """Test regime detection and filtering."""
    
    def test_trend_regime_filter(self):
        """Test trend regime detection."""
        df = create_test_df(trend='up')
        regime_mask = check_regime_filter(df, regime_type='trend', min_regime_strength=0.3)
        
        self.assertEqual(len(regime_mask), len(df))
        self.assertTrue(regime_mask.any())
        
    def test_range_regime_filter(self):
        """Test range regime detection."""
        df = create_test_df(trend='sideways')
        regime_mask = check_regime_filter(df, regime_type='range', min_regime_strength=0.3)
        
        self.assertEqual(len(regime_mask), len(df))
        self.assertTrue(isinstance(regime_mask, pd.Series))
        
    def test_volatile_regime_filter(self):
        """Test volatile regime detection."""
        df = create_test_df(volatility='high')
        regime_mask = check_regime_filter(df, regime_type='volatile', min_regime_strength=0.3)
        
        self.assertEqual(len(regime_mask), len(df))
        self.assertTrue(isinstance(regime_mask, pd.Series))
        
    def test_any_regime_always_passes(self):
        """Test 'any' regime type always returns True."""
        df = create_test_df()
        regime_mask = check_regime_filter(df, regime_type='any')
        
        self.assertTrue(regime_mask.all())
        
    def test_empty_dataframe(self):
        """Test regime filter with empty DataFrame."""
        df = pd.DataFrame()
        regime_mask = check_regime_filter(df)
        
        self.assertEqual(len(regime_mask), 0)
        
    def test_short_dataframe(self):
        """Test regime filter with short DataFrame."""
        df = create_test_df(periods=30)
        regime_mask = check_regime_filter(df)
        
        self.assertEqual(len(regime_mask), 30)
        self.assertTrue(regime_mask.all())


class TestTrendStrength(unittest.TestCase):
    """Test trend strength calculation."""
    
    def test_uptrend_strength(self):
        """Test trend strength for uptrend."""
        df = create_test_df(trend='up')
        trend_strength = calculate_trend_strength(df['Close'])
        
        self.assertEqual(len(trend_strength), len(df))
        self.assertTrue((trend_strength >= 0).all())
        self.assertTrue((trend_strength <= 1).all())
        self.assertTrue(trend_strength.iloc[-20:].mean() > 0.3)
        
    def test_downtrend_strength(self):
        """Test trend strength for downtrend."""
        df = create_test_df(trend='down')
        trend_strength = calculate_trend_strength(df['Close'])
        
        self.assertEqual(len(trend_strength), len(df))
        self.assertTrue((trend_strength >= 0).all())
        self.assertTrue((trend_strength <= 1).all())
        
    def test_neutral_trend_strength(self):
        """Test trend strength for neutral/sideways market."""
        df = create_test_df(trend='sideways')
        trend_strength = calculate_trend_strength(df['Close'])
        
        self.assertTrue((trend_strength >= 0).all())
        self.assertTrue((trend_strength <= 1).all())
        
    def test_short_series(self):
        """Test trend strength with short series."""
        df = create_test_df(periods=30)
        trend_strength = calculate_trend_strength(df['Close'])
        
        self.assertEqual(len(trend_strength), 30)
        self.assertTrue((trend_strength == 0.5).all())


class TestVolatilityRegime(unittest.TestCase):
    """Test volatility regime calculation."""
    
    def test_high_volatility(self):
        """Test volatility regime for high volatility."""
        df = create_test_df(volatility='high')
        vol_regime = calculate_volatility_regime(df['Close'], df['High'], df['Low'])
        
        self.assertEqual(len(vol_regime), len(df))
        self.assertTrue((vol_regime >= 0).all())
        self.assertTrue((vol_regime <= 1).all())
        
    def test_low_volatility(self):
        """Test volatility regime for low volatility."""
        df = create_test_df(volatility='low')
        vol_regime = calculate_volatility_regime(df['Close'], df['High'], df['Low'])
        
        self.assertEqual(len(vol_regime), len(df))
        self.assertTrue((vol_regime >= 0).all())
        self.assertTrue((vol_regime <= 1).all())
        
    def test_short_series(self):
        """Test volatility regime with short series."""
        df = create_test_df(periods=15)
        vol_regime = calculate_volatility_regime(df['Close'], df['High'], df['Low'])
        
        self.assertEqual(len(vol_regime), 15)


class TestRangeRegime(unittest.TestCase):
    """Test range regime calculation."""
    
    def test_sideways_range(self):
        """Test range regime for sideways market."""
        df = create_test_df(trend='sideways')
        range_regime = calculate_range_regime(df['Close'])
        
        self.assertEqual(len(range_regime), len(df))
        self.assertTrue((range_regime >= 0).all())
        self.assertTrue((range_regime <= 1).all())
        
    def test_trending_range(self):
        """Test range regime for trending market."""
        df = create_test_df(trend='up')
        range_regime = calculate_range_regime(df['Close'])
        
        self.assertEqual(len(range_regime), len(df))
        self.assertTrue((range_regime >= 0).all())
        self.assertTrue((range_regime <= 1).all())
        
    def test_short_series(self):
        """Test range regime with short series."""
        df = create_test_df(periods=30)
        range_regime = calculate_range_regime(df['Close'])
        
        self.assertEqual(len(range_regime), 30)


class TestATR(unittest.TestCase):
    """Test ATR calculation."""
    
    def test_atr_calculation(self):
        """Test ATR calculation."""
        df = create_test_df()
        atr = calculate_atr(df['High'], df['Low'], df['Close'])
        
        self.assertEqual(len(atr), len(df))
        self.assertTrue((atr >= 0).all())
        
    def test_atr_with_gaps(self):
        """Test ATR with price gaps."""
        df = create_test_df()
        df.loc[df.index[50], 'Open'] = df.loc[df.index[50], 'Close'] * 1.05
        atr = calculate_atr(df['High'], df['Low'], df['Close'])
        
        self.assertTrue((atr >= 0).all())
        
    def test_atr_short_series(self):
        """Test ATR with short series."""
        df = create_test_df(periods=5)
        atr = calculate_atr(df['High'], df['Low'], df['Close'])
        
        self.assertEqual(len(atr), 5)


class TestMultiRegimeAlignment(unittest.TestCase):
    """Test multi-regime alignment checks."""
    
    def test_all_timeframes_aligned(self):
        """Test when all timeframes are in same regime."""
        def fetch_func(symbol, timeframe):
            return create_test_df(trend='up')
        
        results = check_multi_regime_alignment(
            'BTC/USDT',
            ['1h', '4h', '1d'],
            fetch_func,
            regime_type='trend'
        )
        
        self.assertEqual(len(results), 3)
        self.assertIn('1h', results)
        self.assertIn('4h', results)
        self.assertIn('1d', results)
        
    def test_mixed_regime_alignment(self):
        """Test when timeframes are in different regimes."""
        def fetch_func(symbol, timeframe):
            if timeframe == '1h':
                return create_test_df(trend='up')
            else:
                return create_test_df(trend='down')
        
        results = check_multi_regime_alignment(
            'BTC/USDT',
            ['1h', '4h', '1d'],
            fetch_func,
            regime_type='trend'
        )
        
        self.assertEqual(len(results), 3)
        
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        def fetch_func(symbol, timeframe):
            return pd.DataFrame()
        
        results = check_multi_regime_alignment(
            'BTC/USDT',
            ['1h', '4h'],
            fetch_func,
        )
        
        self.assertTrue(all(results.values()))


class TestRegimeScore(unittest.TestCase):
    """Test regime score calculation."""
    
    def test_long_signal_uptrend_score(self):
        """Test regime score for long signal in uptrend."""
        df = create_test_df(trend='up')
        score = get_regime_score(df, signal_direction=1)
        
        self.assertTrue(0 <= score <= 1)
        self.assertTrue(score > 0.5)
        
    def test_long_signal_downtrend_score(self):
        """Test regime score for long signal in downtrend."""
        df = create_test_df(trend='down')
        score = get_regime_score(df, signal_direction=1)
        
        self.assertTrue(0 <= score <= 1)
        self.assertTrue(score < 0.7)
        
    def test_short_signal_downtrend_score(self):
        """Test regime score for short signal in downtrend."""
        df = create_test_df(trend='down')
        score = get_regime_score(df, signal_direction=-1)
        
        self.assertTrue(0 <= score <= 1)
        self.assertTrue(score > 0.5)
        
    def test_short_signal_uptrend_score(self):
        """Test regime score for short signal in uptrend."""
        df = create_test_df(trend='up')
        score = get_regime_score(df, signal_direction=-1)
        
        self.assertTrue(0 <= score <= 1)
        self.assertTrue(score < 0.7)
        
    def test_flat_signal_score(self):
        """Test regime score for flat signal."""
        df = create_test_df()
        score = get_regime_score(df, signal_direction=0)
        
        self.assertEqual(score, 0.5)
        
    def test_empty_dataframe_score(self):
        """Test regime score with empty DataFrame."""
        df = pd.DataFrame()
        score = get_regime_score(df, signal_direction=1)
        
        self.assertEqual(score, 0.5)
        
    def test_short_dataframe_score(self):
        """Test regime score with short DataFrame."""
        df = create_test_df(periods=30)
        score = get_regime_score(df, signal_direction=1)
        
        self.assertEqual(score, 0.5)


if __name__ == '__main__':
    unittest.main()
