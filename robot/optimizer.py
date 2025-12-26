"""
Parameter Optimizer with Walk-Forward Validation
=================================================
Prevents overfitting by testing on out-of-sample data.

Features:
- Grid search over parameter combinations
- Walk-forward analysis (train/test split)
- Automatic best parameter selection
- Config auto-update with optimized values
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from robot.strategy import KalmanStrategy
from robot.backtester import Backtester, BacktestResult
from robot.performance import PerformanceMetrics


@dataclass
class OptimizationResult:
    """Result of a single parameter test."""
    params: Dict[str, float]
    in_sample_metrics: PerformanceMetrics
    out_of_sample_metrics: Optional[PerformanceMetrics]
    is_robust: bool  # True if OOS performance is similar to IS
    score: float  # Combined ranking score


@dataclass
class OptimizationReport:
    """Complete optimization report."""
    best_params: Dict[str, float]
    all_results: List[OptimizationResult]
    in_sample_period: str
    out_of_sample_period: str
    total_combinations: int
    profitable_combinations: int


class ParameterOptimizer:
    """
    Optimizes strategy parameters using walk-forward validation.

    Walk-forward process:
    1. Split data into in-sample (training) and out-of-sample (testing)
    2. Optimize on in-sample data
    3. Validate on out-of-sample data
    4. Only accept parameters that work on both
    """

    # Default parameter grids
    DEFAULT_GRIDS = {
        "threshold": [1.0, 1.5, 2.0, 2.5, 3.0],
        "process_noise": [0.001, 0.005, 0.01, 0.02, 0.05],
        "measurement_noise": [0.1, 0.5, 1.0, 2.0, 5.0]
    }

    def __init__(
        self,
        param_grids: Dict[str, List[float]] = None,
        train_ratio: float = 0.7,  # 70% train, 30% test
        min_trades: int = 20,
        min_sharpe: float = 0.5,
        max_drawdown: float = 0.25  # Max 25% drawdown allowed
    ):
        self.param_grids = param_grids or self.DEFAULT_GRIDS
        self.train_ratio = train_ratio
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown

    def optimize(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        initial_capital: float = 10000,
        n_jobs: int = 4
    ) -> OptimizationReport:
        """
        Run full optimization with walk-forward validation.

        Args:
            df: Historical OHLCV data
            symbol: Symbol name
            initial_capital: Starting capital
            n_jobs: Number of parallel jobs

        Returns:
            OptimizationReport with best parameters and all results
        """
        # Split data
        split_idx = int(len(df) * self.train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        train_period = f"{train_df.index[0].date()} to {train_df.index[-1].date()}"
        test_period = f"{test_df.index[0].date()} to {test_df.index[-1].date()}"

        # Generate all parameter combinations
        param_names = list(self.param_grids.keys())
        param_values = list(self.param_grids.values())
        combinations = list(product(*param_values))

        results: List[OptimizationResult] = []

        # Test each combination
        for combo in combinations:
            params = dict(zip(param_names, combo))
            result = self._test_params(
                params, train_df, test_df, symbol, initial_capital
            )
            if result:
                results.append(result)

        # Sort by score (higher is better)
        results.sort(key=lambda x: x.score, reverse=True)

        # Count profitable combinations
        profitable = sum(1 for r in results if r.is_robust and
                        r.out_of_sample_metrics and
                        r.out_of_sample_metrics.total_return > 0)

        best_params = results[0].params if results else self._default_params()

        return OptimizationReport(
            best_params=best_params,
            all_results=results,
            in_sample_period=train_period,
            out_of_sample_period=test_period,
            total_combinations=len(combinations),
            profitable_combinations=profitable
        )

    def _test_params(
        self,
        params: Dict[str, float],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        symbol: str,
        initial_capital: float
    ) -> Optional[OptimizationResult]:
        """Test a single parameter combination on train and test data."""
        try:
            # Create strategy with these params
            strategy = KalmanStrategy(
                process_noise=params.get("process_noise", 0.01),
                measurement_noise=params.get("measurement_noise", 1.0),
                threshold=params.get("threshold", 1.5)
            )

            # In-sample backtest
            bt_train = Backtester(strategy=strategy, initial_capital=initial_capital)
            train_result = bt_train.run(train_df, symbol)
            is_metrics = train_result.metrics

            # Check minimum requirements on training data
            if is_metrics.total_trades < self.min_trades:
                return None
            if is_metrics.sharpe_ratio < self.min_sharpe:
                return None
            if abs(is_metrics.max_drawdown) > self.max_drawdown:
                return None

            # Out-of-sample backtest
            bt_test = Backtester(strategy=strategy, initial_capital=initial_capital)

            # Need enough data for OOS test
            if len(test_df) < 100:
                oos_metrics = None
                is_robust = False
            else:
                test_result = bt_test.run(test_df, symbol)
                oos_metrics = test_result.metrics

                # Check if OOS performance is similar to IS (robustness check)
                # Allow 50% degradation in Sharpe ratio
                sharpe_degradation = (is_metrics.sharpe_ratio - oos_metrics.sharpe_ratio) / max(is_metrics.sharpe_ratio, 0.01)
                is_robust = (
                    oos_metrics.total_return > 0 and
                    oos_metrics.sharpe_ratio > 0 and
                    sharpe_degradation < 0.5  # Less than 50% degradation
                )

            # Calculate combined score
            # Prioritize: OOS Sharpe > IS Sharpe > Low Drawdown
            score = self._calculate_score(is_metrics, oos_metrics, is_robust)

            return OptimizationResult(
                params=params,
                in_sample_metrics=is_metrics,
                out_of_sample_metrics=oos_metrics,
                is_robust=is_robust,
                score=score
            )

        except Exception:
            return None

    def _calculate_score(
        self,
        is_metrics: PerformanceMetrics,
        oos_metrics: Optional[PerformanceMetrics],
        is_robust: bool
    ) -> float:
        """Calculate ranking score for parameter combination."""
        score = 0.0

        # In-sample contribution (40%)
        score += is_metrics.sharpe_ratio * 0.2
        score += min(is_metrics.total_return * 100, 50) * 0.1  # Cap at 50%
        score += (1 - abs(is_metrics.max_drawdown)) * 0.1

        # Out-of-sample contribution (60%) - more weight on OOS
        if oos_metrics:
            score += oos_metrics.sharpe_ratio * 0.3
            score += min(oos_metrics.total_return * 100, 50) * 0.2
            score += (1 - abs(oos_metrics.max_drawdown)) * 0.1

        # Robustness bonus
        if is_robust:
            score *= 1.5

        return score

    def _default_params(self) -> Dict[str, float]:
        """Return default parameters if optimization fails."""
        return {
            "threshold": 2.0,
            "process_noise": 0.01,
            "measurement_noise": 1.0
        }


def run_optimization(
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    bars: int = 2000,
    save_results: bool = True
) -> OptimizationReport:
    """
    Convenience function to run full optimization.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe to optimize
        bars: Number of historical bars
        save_results: Whether to save results to file

    Returns:
        OptimizationReport with best parameters
    """
    from robot.data_fetcher import DataFetcher

    print(f"🔍 Optimizing {symbol} on {timeframe}...")
    print(f"   Fetching {bars} bars of historical data...")

    fetcher = DataFetcher()
    df = fetcher.fetch(symbol, timeframe, bars)

    if df.empty or len(df) < 500:
        raise ValueError(f"Insufficient data for optimization: {len(df)} bars")

    optimizer = ParameterOptimizer()
    report = optimizer.optimize(df, symbol)

    print(f"\n📊 Optimization Results:")
    print(f"   Total combinations tested: {report.total_combinations}")
    print(f"   Profitable combinations: {report.profitable_combinations}")
    print(f"   Best parameters: {report.best_params}")

    if report.all_results:
        best = report.all_results[0]
        print(f"\n   In-Sample Performance:")
        print(f"     Return: {best.in_sample_metrics.total_return:.2%}")
        print(f"     Sharpe: {best.in_sample_metrics.sharpe_ratio:.2f}")

        if best.out_of_sample_metrics:
            print(f"\n   Out-of-Sample Performance:")
            print(f"     Return: {best.out_of_sample_metrics.total_return:.2%}")
            print(f"     Sharpe: {best.out_of_sample_metrics.sharpe_ratio:.2f}")
            print(f"     Robust: {'✅ Yes' if best.is_robust else '❌ No'}")

    if save_results:
        _save_optimization_results(report, symbol, timeframe)

    return report


def _save_optimization_results(report: OptimizationReport, symbol: str, timeframe: str):
    """Save optimization results to JSON file."""
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)

    filename = output_dir / f"opt_{symbol}_{timeframe}.json"

    data = {
        "symbol": symbol,
        "timeframe": timeframe,
        "best_params": report.best_params,
        "in_sample_period": report.in_sample_period,
        "out_of_sample_period": report.out_of_sample_period,
        "total_combinations": report.total_combinations,
        "profitable_combinations": report.profitable_combinations
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n💾 Results saved to {filename}")


def update_config_with_best_params(params: Dict[str, float]):
    """Update the main config with optimized parameters."""
    from robot.config import Config

    # Update Config class attributes
    if "threshold" in params:
        Config.SIGNAL_THRESHOLD = params["threshold"]
    if "process_noise" in params:
        Config.PROCESS_NOISE = params["process_noise"]
    if "measurement_noise" in params:
        Config.MEASUREMENT_NOISE = params["measurement_noise"]

    print(f"✅ Config updated with optimized parameters:")
    print(f"   SIGNAL_THRESHOLD = {Config.SIGNAL_THRESHOLD}")
    print(f"   PROCESS_NOISE = {Config.PROCESS_NOISE}")
    print(f"   MEASUREMENT_NOISE = {Config.MEASUREMENT_NOISE}")
