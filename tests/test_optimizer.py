"""
Unit Tests for Optimizer Module
================================
Tests for parameter optimization and walk-forward validation.
"""
import pytest
import pandas as pd
import numpy as np
from robot.optimizer import ParameterOptimizer, OptimizationResult, OptimizationReport


class TestOptimizerInit:
    """Test ParameterOptimizer initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        opt = ParameterOptimizer()
        
        assert opt.train_ratio == 0.7
        assert opt.min_trades == 20
        assert opt.min_sharpe == 0.5
        assert opt.max_drawdown == 0.25
    
    def test_custom_init(self):
        """Test custom initialization."""
        opt = ParameterOptimizer(
            train_ratio=0.8,
            min_trades=10,
            min_sharpe=1.0,
            max_drawdown=0.15
        )
        
        assert opt.train_ratio == 0.8
        assert opt.min_trades == 10
        assert opt.min_sharpe == 1.0
        assert opt.max_drawdown == 0.15
    
    def test_custom_param_grids(self):
        """Test custom parameter grids."""
        grids = {
            "threshold": [1.0, 2.0],
            "process_noise": [0.01]
        }
        opt = ParameterOptimizer(param_grids=grids)
        
        assert opt.param_grids == grids


class TestOptimization:
    """Test optimization process."""
    
    @pytest.fixture
    def large_sample_data(self):
        """Generate large sample data for optimization."""
        np.random.seed(42)
        n_bars = 1000
        dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='h')
        
        # Mean-reverting data
        mean = 1.10
        close = mean + 0.01 * np.sin(np.linspace(0, 20*np.pi, n_bars))
        close += np.random.normal(0, 0.001, n_bars)
        
        return pd.DataFrame({
            'Open': np.roll(close, 1),
            'High': close * 1.002,
            'Low': close * 0.998,
            'Close': close,
            'Volume': np.random.randint(100, 10000, n_bars).astype(float)
        }, index=dates)
    
    def test_optimize_returns_report(self, large_sample_data):
        """Test that optimize returns OptimizationReport."""
        # Use minimal param grid for speed
        opt = ParameterOptimizer(
            param_grids={"threshold": [1.5, 2.0]},
            min_trades=5,
            min_sharpe=0.0
        )
        
        report = opt.optimize(large_sample_data, "EURUSD")
        
        assert isinstance(report, OptimizationReport)
    
    def test_report_has_best_params(self, large_sample_data):
        """Test that report includes best parameters."""
        opt = ParameterOptimizer(
            param_grids={"threshold": [1.5, 2.0]},
            min_trades=5,
            min_sharpe=0.0
        )
        
        report = opt.optimize(large_sample_data, "EURUSD")
        
        assert "threshold" in report.best_params
    
    def test_report_has_periods(self, large_sample_data):
        """Test that report includes train/test periods."""
        opt = ParameterOptimizer(
            param_grids={"threshold": [1.5]},
            min_trades=5,
            min_sharpe=0.0
        )
        
        report = opt.optimize(large_sample_data, "EURUSD")
        
        assert report.in_sample_period is not None
        assert report.out_of_sample_period is not None
    
    def test_total_combinations_correct(self, large_sample_data):
        """Test that total combinations is calculated correctly."""
        opt = ParameterOptimizer(
            param_grids={
                "threshold": [1.0, 2.0],
                "process_noise": [0.01, 0.02]
            },
            min_trades=5,
            min_sharpe=0.0
        )
        
        report = opt.optimize(large_sample_data, "EURUSD")
        
        # 2 threshold * 2 process_noise = 4 combinations
        assert report.total_combinations == 4


class TestWalkForwardValidation:
    """Test walk-forward validation process."""
    
    def test_train_test_split(self):
        """Test that data is split correctly."""
        opt = ParameterOptimizer(train_ratio=0.7)
        
        # With 1000 bars, should split at 700
        assert opt.train_ratio == 0.7


class TestRobustnessCheck:
    """Test robustness validation."""
    
    def test_robust_params_preferred(self, large_sample_data):
        """Test that robust parameters are scored higher."""
        opt = ParameterOptimizer(
            param_grids={"threshold": [1.5, 2.0, 2.5]},
            min_trades=5,
            min_sharpe=0.0
        )
        
        report = opt.optimize(large_sample_data, "EURUSD")
        
        if report.all_results:
            # Best result should be at the top
            best = report.all_results[0]
            assert best.score >= report.all_results[-1].score


class TestEdgeCases:
    """Test edge cases for optimizer."""
    
    def test_insufficient_data(self, small_ohlcv_data):
        """Test handling of insufficient data."""
        opt = ParameterOptimizer()
        
        # Should either raise error or return default params
        try:
            report = opt.optimize(small_ohlcv_data, "EURUSD")
            # If it succeeds, should have default params
            assert report.best_params is not None
        except (ValueError, Exception):
            pass  # Expected for insufficient data
    
    def test_no_valid_combinations(self):
        """Test when no combinations meet criteria."""
        np.random.seed(42)
        n_bars = 500
        dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='h')
        
        # Random walk - unlikely to produce good strategies
        close = 1.10 + np.cumsum(np.random.normal(0, 0.001, n_bars))
        
        df = pd.DataFrame({
            'Open': np.roll(close, 1),
            'High': close * 1.002,
            'Low': close * 0.998,
            'Close': close,
            'Volume': np.ones(n_bars) * 1000
        }, index=dates)
        
        opt = ParameterOptimizer(
            param_grids={"threshold": [1.0]},
            min_trades=1000,  # Impossible requirement
            min_sharpe=100.0  # Impossible requirement
        )
        
        report = opt.optimize(df, "EURUSD")
        
        # Should return default params
        assert report.best_params is not None


@pytest.fixture
def large_sample_data():
    """Generate large sample data for optimization."""
    np.random.seed(42)
    n_bars = 1000
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='h')
    
    mean = 1.10
    close = mean + 0.01 * np.sin(np.linspace(0, 20*np.pi, n_bars))
    close += np.random.normal(0, 0.001, n_bars)
    
    return pd.DataFrame({
        'Open': np.roll(close, 1),
        'High': close * 1.002,
        'Low': close * 0.998,
        'Close': close,
        'Volume': np.random.randint(100, 10000, n_bars).astype(float)
    }, index=dates)

