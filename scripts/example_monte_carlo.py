"""Example script demonstrating Monte Carlo robustness testing.

This script shows how to use the Monte Carlo simulation features for strategy robustness testing.
It can be run standalone or integrated into a WFO pipeline.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openquant.evaluation import (
    run_comprehensive_mc,
    MonteCarloConfig,
    evaluate_robustness,
    walk_forward_evaluate,
    WFOSpec,
)
from openquant.strategies.base import BaseStrategy


class SimpleMovingAverageStrategy(BaseStrategy):
    """Example MA crossover strategy for demonstration."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, threshold: float = 0.0):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.threshold = threshold
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df['Close']
        
        fast_ma = close.rolling(window=self.fast_period).mean()
        slow_ma = close.rolling(window=self.slow_period).mean()
        
        diff = (fast_ma - slow_ma) / slow_ma
        
        signals = pd.Series(0, index=df.index)
        signals[diff > self.threshold] = 1
        signals[diff < -self.threshold] = -1
        
        return signals


def generate_sample_data(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    
    dates = pd.date_range(start='2020-01-01', periods=n_bars, freq='1D')
    
    returns = np.random.randn(n_bars) * 0.02 + 0.0005
    close = 100 * np.exp(np.cumsum(returns))
    
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.01))
    open_ = close * (1 + np.random.randn(n_bars) * 0.005)
    volume = np.random.randint(1000000, 10000000, n_bars)
    
    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)
    
    return df


def example_standalone_mc():
    """Example 1: Standalone Monte Carlo analysis."""
    print("=" * 80)
    print("Example 1: Standalone Monte Carlo Robustness Testing")
    print("=" * 80)
    
    df = generate_sample_data(n_bars=500)
    
    def strategy_factory(fast_period=10, slow_period=30, threshold=0.0):
        return SimpleMovingAverageStrategy(fast_period, slow_period, threshold)
    
    params = {"fast_period": 10, "slow_period": 30, "threshold": 0.01}
    
    mc_config = MonteCarloConfig(
        n_simulations=50,
        block_size=20,
        param_perturbation_pct=0.15
    )
    
    print("\nRunning comprehensive Monte Carlo analysis...")
    mc_results = run_comprehensive_mc(
        df=df,
        strategy_factory=strategy_factory,
        params=params,
        config=mc_config,
        fee_bps=2.0,
        weight=1.0
    )
    
    print("\n--- Path-Dependent Results ---")
    if "path_dependent" in mc_results and "sharpe" in mc_results["path_dependent"]:
        sharpe_result = mc_results["path_dependent"]["sharpe"]
        print(f"Mean Sharpe: {sharpe_result.mean:.2f}")
        print(f"Median Sharpe: {sharpe_result.median:.2f}")
        print(f"5th Percentile: {sharpe_result.percentile_5:.2f}")
        print(f"95th Percentile: {sharpe_result.percentile_95:.2f}")
        print(f"Std Dev: {sharpe_result.std:.2f}")
    
    print("\n--- Parameter Perturbation Results ---")
    if "parameter_perturbation" in mc_results and "sharpe" in mc_results["parameter_perturbation"]:
        sharpe_result = mc_results["parameter_perturbation"]["sharpe"]
        print(f"Mean Sharpe: {sharpe_result.mean:.2f}")
        print(f"5th Percentile: {sharpe_result.percentile_5:.2f}")
        print(f"95th Percentile: {sharpe_result.percentile_95:.2f}")
    
    print("\n--- Regime Shift Results ---")
    if "regime_shift" in mc_results:
        print(f"Tested {len(mc_results['regime_shift'])} regimes")
        for regime_name, regime_results in list(mc_results["regime_shift"].items())[:5]:
            if "sharpe" in regime_results:
                print(f"  {regime_name}: Sharpe = {regime_results['sharpe'].mean:.2f}")
    
    print("\n--- Overall Summary ---")
    if "overall_summary" in mc_results:
        summary = mc_results["overall_summary"]
        print(f"Mean Sharpe: {summary['sharpe_mean']:.2f}")
        print(f"5th Percentile: {summary['sharpe_5th_percentile']:.2f}")
        print(f"95th Percentile: {summary['sharpe_95th_percentile']:.2f}")
        print(f"Is Robust: {summary['is_robust']}")
        print(f"Total Simulations: {summary['total_simulations']}")
    
    print("\n--- Robustness Evaluation ---")
    robustness_eval = evaluate_robustness(mc_results)
    print(f"Robustness Score: {robustness_eval['robustness_score']:.2f}")
    print(f"Rating: {robustness_eval['rating']}")
    print(f"Details: {robustness_eval['details']}")


def example_wfo_with_mc():
    """Example 2: Walk-Forward Optimization with Monte Carlo."""
    print("\n" + "=" * 80)
    print("Example 2: Walk-Forward Optimization with Monte Carlo")
    print("=" * 80)
    
    df = generate_sample_data(n_bars=800)
    
    def strategy_factory(fast_period=10, slow_period=30):
        return SimpleMovingAverageStrategy(fast_period, slow_period)
    
    param_grid = {
        "fast_period": [5, 10, 15],
        "slow_period": [20, 30, 40],
    }
    
    wfo_spec = WFOSpec(
        n_splits=4,
        train_frac=0.7,
        use_monte_carlo=True,
        mc_n_simulations=30,
        mc_block_size=15,
        mc_param_perturbation_pct=0.1
    )
    
    print("\nRunning WFO with Monte Carlo robustness testing...")
    wfo_results = walk_forward_evaluate(
        df=df,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        fee_bps=2.0,
        weight=1.0,
        wfo=wfo_spec
    )
    
    print("\n--- WFO Results ---")
    print(f"Mean Test Sharpe: {wfo_results['mean_test_sharpe']:.2f}")
    print(f"Test Sharpes per split: {[f'{s:.2f}' for s in wfo_results['test_sharpes']]}")
    print(f"Best params per split: {wfo_results['best_params_per_split']}")
    
    if "robustness_evaluation" in wfo_results:
        print("\n--- Monte Carlo Robustness Evaluation ---")
        robustness = wfo_results["robustness_evaluation"]
        print(f"Robustness Score: {robustness['robustness_score']:.2f}")
        print(f"Rating: {robustness['rating']}")
        print(f"Details: {robustness['details']}")


if __name__ == "__main__":
    example_standalone_mc()
    example_wfo_with_mc()
    
    print("\n" + "=" * 80)
    print("Monte Carlo examples completed!")
    print("=" * 80)
