"""Unit tests for Monte Carlo robustness testing module."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from openquant.evaluation.monte_carlo import (
    PathDependentBootstrap,
    ParameterPerturbation,
    RegimeShiftSimulator,
    MonteCarloConfig,
    MonteCarloResult,
    run_path_dependent_mc,
    run_parameter_perturbation_mc,
    run_regime_shift_mc,
    run_comprehensive_mc,
    evaluate_robustness,
)
from openquant.strategies.base import BaseStrategy


class SimpleTestStrategy(BaseStrategy):
    """Simple strategy for testing."""
    
    def __init__(self, window: int = 20, threshold: float = 0.01):
        self.window = window
        self.threshold = threshold
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        ma = close.rolling(window=self.window).mean()
        
        signals = pd.Series(0, index=df.index)
        signals[close > ma * (1 + self.threshold)] = 1
        signals[close < ma * (1 - self.threshold)] = -1
        
        return signals


@pytest.fixture
def sample_df():
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2020-01-01', periods=n, freq='1D')
    
    returns = np.random.randn(n) * 0.01 + 0.0002
    close = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Open': close * (1 + np.random.randn(n) * 0.002),
        'High': close * (1 + np.abs(np.random.randn(n) * 0.01)),
        'Low': close * (1 - np.abs(np.random.randn(n) * 0.01)),
        'Close': close,
        'Volume': np.random.randint(1000000, 5000000, n),
    }, index=dates)
    
    return df


@pytest.fixture
def strategy_factory():
    """Factory function for creating test strategies."""
    def factory(window=20, threshold=0.01):
        return SimpleTestStrategy(window=window, threshold=threshold)
    return factory


def test_path_dependent_bootstrap(sample_df):
    """Test path-dependent bootstrap."""
    bootstrap = PathDependentBootstrap(block_size=10)
    
    resampled_df = bootstrap.resample_prices(sample_df, seed=42)
    
    assert len(resampled_df) == len(sample_df)
    assert set(resampled_df.columns) == set(sample_df.columns)
    assert not resampled_df['Close'].equals(sample_df['Close'])
    assert resampled_df['Close'].iloc[0] == sample_df['Close'].iloc[0]


def test_parameter_perturbation():
    """Test parameter perturbation."""
    perturbator = ParameterPerturbation(perturbation_pct=0.1)
    
    params = {"window": 20, "threshold": 0.01, "name": "test"}
    perturbed = perturbator.perturb_params(params, seed=42)
    
    assert "window" in perturbed
    assert "threshold" in perturbed
    assert perturbed["name"] == "test"
    assert perturbed["window"] != 20
    assert perturbed["threshold"] != 0.01
    assert isinstance(perturbed["window"], int)
    assert isinstance(perturbed["threshold"], float)


def test_regime_shift_simulator(sample_df):
    """Test regime shift simulation."""
    simulator = RegimeShiftSimulator(
        volatility_multipliers=[0.5, 2.0],
        trend_multipliers=[0.0, 1.0]
    )
    
    df_low_vol = simulator.apply_volatility_regime(sample_df, 0.5)
    df_high_vol = simulator.apply_volatility_regime(sample_df, 2.0)
    
    returns_orig = sample_df['Close'].pct_change().dropna()
    returns_low_vol = df_low_vol['Close'].pct_change().dropna()
    returns_high_vol = df_high_vol['Close'].pct_change().dropna()
    
    assert returns_low_vol.std() < returns_orig.std()
    assert returns_high_vol.std() > returns_orig.std()
    
    df_trend = simulator.apply_trend_regime(sample_df, 1.0)
    assert df_trend['Close'].iloc[-1] > df_trend['Close'].iloc[0]


def test_monte_carlo_config():
    """Test MonteCarloConfig initialization."""
    config = MonteCarloConfig()
    
    assert config.n_simulations == 500
    assert config.block_size == 20
    assert config.confidence_level == 0.95
    assert len(config.regime_shift_volatility_multipliers) == 4
    
    custom_config = MonteCarloConfig(
        n_simulations=100,
        block_size=30,
        regime_shift_volatility_multipliers=[1.0, 2.0]
    )
    assert custom_config.n_simulations == 100
    assert custom_config.block_size == 30
    assert len(custom_config.regime_shift_volatility_multipliers) == 2


def test_monte_carlo_result():
    """Test MonteCarloResult."""
    simulations = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    result = MonteCarloResult(
        metric="sharpe",
        mean=np.mean(simulations),
        median=np.median(simulations),
        std=np.std(simulations),
        percentile_5=np.percentile(simulations, 5),
        percentile_95=np.percentile(simulations, 95),
        min=np.min(simulations),
        max=np.max(simulations),
        simulations=simulations
    )
    
    assert result.mean == 1.5
    assert result.median == 1.5
    assert result.is_robust(threshold=0.0)
    assert not result.is_robust(threshold=1.0)


def test_run_path_dependent_mc(sample_df, strategy_factory):
    """Test path-dependent Monte Carlo."""
    config = MonteCarloConfig(n_simulations=10, block_size=20)
    params = {"window": 20, "threshold": 0.01}
    
    results = run_path_dependent_mc(
        sample_df, strategy_factory, params, config, fee_bps=2.0
    )
    
    assert "sharpe" in results
    assert "sortino" in results
    assert "max_drawdown" in results
    
    assert len(results["sharpe"].simulations) <= 10
    assert results["sharpe"].mean is not None
    assert results["sharpe"].percentile_5 is not None


def test_run_parameter_perturbation_mc(sample_df, strategy_factory):
    """Test parameter perturbation Monte Carlo."""
    config = MonteCarloConfig(n_simulations=10, param_perturbation_pct=0.1)
    params = {"window": 20, "threshold": 0.01}
    
    results = run_parameter_perturbation_mc(
        sample_df, strategy_factory, params, config, fee_bps=2.0
    )
    
    assert "sharpe" in results
    assert len(results["sharpe"].simulations) <= 10


def test_run_regime_shift_mc(sample_df, strategy_factory):
    """Test regime shift Monte Carlo."""
    config = MonteCarloConfig(
        regime_shift_volatility_multipliers=[0.5, 2.0],
        regime_shift_trend_multipliers=[0.0, 1.0]
    )
    params = {"window": 20, "threshold": 0.01}
    
    results = run_regime_shift_mc(
        sample_df, strategy_factory, params, config, fee_bps=2.0
    )
    
    assert len(results) > 0
    
    for regime_name, regime_results in results.items():
        assert "sharpe" in regime_results
        assert "sortino" in regime_results
        assert "max_drawdown" in regime_results


def test_run_comprehensive_mc(sample_df, strategy_factory):
    """Test comprehensive Monte Carlo analysis."""
    config = MonteCarloConfig(
        n_simulations=5,
        regime_shift_volatility_multipliers=[1.0, 2.0],
        regime_shift_trend_multipliers=[0.0]
    )
    params = {"window": 20, "threshold": 0.01}
    
    results = run_comprehensive_mc(
        sample_df, strategy_factory, params, config, fee_bps=2.0
    )
    
    assert "path_dependent" in results
    assert "parameter_perturbation" in results
    assert "regime_shift" in results
    assert "overall_summary" in results
    
    summary = results["overall_summary"]
    assert "sharpe_mean" in summary
    assert "sharpe_5th_percentile" in summary
    assert "total_simulations" in summary


def test_evaluate_robustness():
    """Test robustness evaluation."""
    mc_results = {
        "path_dependent": {
            "sharpe": MonteCarloResult(
                metric="sharpe",
                mean=1.0,
                median=1.0,
                std=0.2,
                percentile_5=0.6,
                percentile_95=1.4,
                min=0.5,
                max=1.5,
                simulations=[1.0] * 10
            )
        },
        "parameter_perturbation": {
            "sharpe": MonteCarloResult(
                metric="sharpe",
                mean=0.9,
                median=0.9,
                std=0.3,
                percentile_5=0.4,
                percentile_95=1.4,
                min=0.3,
                max=1.5,
                simulations=[0.9] * 10
            )
        },
        "regime_shift": {
            "vol_1.0x": {"sharpe": MonteCarloResult(
                metric="sharpe", mean=0.8, median=0.8, std=0.0,
                percentile_5=0.8, percentile_95=0.8, min=0.8, max=0.8,
                simulations=[0.8]
            )},
            "vol_2.0x": {"sharpe": MonteCarloResult(
                metric="sharpe", mean=0.5, median=0.5, std=0.0,
                percentile_5=0.5, percentile_95=0.5, min=0.5, max=0.5,
                simulations=[0.5]
            )}
        }
    }
    
    evaluation = evaluate_robustness(mc_results)
    
    assert "robustness_score" in evaluation
    assert "rating" in evaluation
    assert "details" in evaluation
    assert 0.0 <= evaluation["robustness_score"] <= 1.0
    assert evaluation["rating"] in ["HIGHLY_ROBUST", "ROBUST", "MODERATE", "FRAGILE"]


def test_selective_mc_runs(sample_df, strategy_factory):
    """Test running individual MC methods."""
    config = MonteCarloConfig(n_simulations=5)
    params = {"window": 20}
    
    results_only_path = run_comprehensive_mc(
        sample_df, strategy_factory, params, config,
        run_path_dependent=True,
        run_param_perturbation=False,
        run_regime_shift=False
    )
    
    assert "path_dependent" in results_only_path
    assert "parameter_perturbation" not in results_only_path
    assert "regime_shift" not in results_only_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
