"""
Unit Tests for Strategy Module
==============================
Tests for KalmanStrategy and signal generation.
"""
import pytest
import pandas as pd
import numpy as np
from robot.strategy import KalmanStrategy, BaseStrategy


class TestKalmanStrategyInit:
    """Test KalmanStrategy initialization."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        strategy = KalmanStrategy()
        assert strategy.Q == 1e-5
        assert strategy.R == 1e-3
        assert strategy.threshold == 1.5
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        strategy = KalmanStrategy(
            process_noise=0.01,
            measurement_noise=0.1,
            threshold=2.0
        )
        assert strategy.Q == 0.01
        assert strategy.R == 0.1
        assert strategy.threshold == 2.0
    
    def test_implements_base_strategy(self):
        """Test that KalmanStrategy implements BaseStrategy."""
        strategy = KalmanStrategy()
        assert isinstance(strategy, BaseStrategy)


class TestKalmanFilter:
    """Test Kalman filter calculations."""
    
    def test_kalman_filter_output_shapes(self, sample_ohlcv_data):
        """Test that Kalman filter returns correct shapes."""
        strategy = KalmanStrategy()
        prices = sample_ohlcv_data['Close'].values
        estimates, deviations = strategy._kalman_filter(prices)
        
        assert len(estimates) == len(prices)
        assert len(deviations) == len(prices)
    
    def test_kalman_filter_estimates_follow_prices(self, sample_ohlcv_data):
        """Test that estimates track prices reasonably."""
        strategy = KalmanStrategy()
        prices = sample_ohlcv_data['Close'].values
        estimates, _ = strategy._kalman_filter(prices)
        
        # Correlation should be very high
        correlation = np.corrcoef(prices, estimates)[0, 1]
        assert correlation > 0.95
    
    def test_kalman_filter_deviations_bounded(self, sample_ohlcv_data):
        """Test that deviations are reasonably bounded."""
        strategy = KalmanStrategy()
        prices = sample_ohlcv_data['Close'].values
        _, deviations = strategy._kalman_filter(prices)
        
        # Deviations should be small relative to price
        max_deviation_pct = np.max(np.abs(deviations)) / np.mean(prices)
        assert max_deviation_pct < 0.1  # Less than 10%


class TestSignalGeneration:
    """Test trading signal generation."""
    
    def test_signals_shape(self, sample_ohlcv_data):
        """Test that signals have correct shape."""
        strategy = KalmanStrategy()
        signals = strategy.generate_signals(sample_ohlcv_data)
        
        assert len(signals) == len(sample_ohlcv_data)
        assert isinstance(signals, pd.Series)
    
    def test_signal_values(self, sample_ohlcv_data):
        """Test that signals are in {-1, 0, 1}."""
        strategy = KalmanStrategy()
        signals = strategy.generate_signals(sample_ohlcv_data)
        
        unique_signals = set(signals.unique())
        assert unique_signals.issubset({-1, 0, 1})
    
    def test_insufficient_data_returns_zeros(self, small_ohlcv_data):
        """Test that insufficient data returns all zeros."""
        strategy = KalmanStrategy()
        signals = strategy.generate_signals(small_ohlcv_data)
        
        assert all(signals == 0)
    
    def test_empty_data_returns_empty(self, empty_dataframe):
        """Test that empty data returns empty series."""
        strategy = KalmanStrategy()
        signals = strategy.generate_signals(empty_dataframe)
        
        assert len(signals) == 0
    
    def test_higher_threshold_fewer_signals(self, sample_ohlcv_data):
        """Test that higher threshold produces fewer signals."""
        strategy_low = KalmanStrategy(threshold=1.0)
        strategy_high = KalmanStrategy(threshold=3.0)
        
        signals_low = strategy_low.generate_signals(sample_ohlcv_data)
        signals_high = strategy_high.generate_signals(sample_ohlcv_data)
        
        active_low = (signals_low != 0).sum()
        active_high = (signals_high != 0).sum()
        
        assert active_high <= active_low
    
    def test_mean_reverting_data_generates_signals(self, mean_reverting_data):
        """Test that mean-reverting data generates signals."""
        strategy = KalmanStrategy(threshold=1.5)
        signals = strategy.generate_signals(mean_reverting_data)
        
        # Should have both long and short signals
        assert 1 in signals.values
        assert -1 in signals.values


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_constant_prices(self):
        """Test behavior with constant prices."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        df = pd.DataFrame({
            'Open': [1.10] * 100,
            'High': [1.10] * 100,
            'Low': [1.10] * 100,
            'Close': [1.10] * 100,
            'Volume': [1000] * 100
        }, index=dates)
        
        strategy = KalmanStrategy()
        signals = strategy.generate_signals(df)
        
        # With constant prices, should have no signals (z-score = 0)
        assert all(signals == 0)
    
    def test_extreme_process_noise(self, sample_ohlcv_data):
        """Test with extreme process noise values."""
        strategy = KalmanStrategy(process_noise=1e-10)
        signals = strategy.generate_signals(sample_ohlcv_data)
        assert len(signals) == len(sample_ohlcv_data)
        
        strategy = KalmanStrategy(process_noise=1.0)
        signals = strategy.generate_signals(sample_ohlcv_data)
        assert len(signals) == len(sample_ohlcv_data)

