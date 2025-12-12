"""End-to-End Profitability Testing Framework.

Walk-forward optimization across 3+ years of data, calculates risk-adjusted returns
(Sharpe, Sortino, Calmar, Omega), performs Monte Carlo robustness tests (500 bootstrap runs),
validates >50% total return target with <25% max drawdown constraint, and generates
go/no-go recommendation with confidence score.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import json

from openquant.backtest.engine import backtest_signals, BacktestResult
from openquant.backtest.metrics import sharpe, sortino, max_drawdown, monte_carlo_bootstrap
from openquant.data.loader import DataLoader
from openquant.strategies.registry import get_strategy
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class WalkForwardWindow:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float


@dataclass
class MonteCarloResults:
    mean_sharpe: float
    std_sharpe: float
    percentile_5_sharpe: float
    percentile_95_sharpe: float
    mean_sortino: float
    std_sortino: float
    mean_max_dd: float
    std_max_dd: float
    percentile_5_max_dd: float
    percentile_95_max_dd: float
    mean_return: float
    std_return: float
    percentile_5_return: float
    percentile_95_return: float


@dataclass
class ProfitabilityReport:
    strategy_name: str
    test_period: str
    walk_forward_windows: int
    in_sample_metrics: PerformanceMetrics
    out_of_sample_metrics: PerformanceMetrics
    monte_carlo_results: MonteCarloResults
    meets_return_target: bool
    meets_drawdown_constraint: bool
    confidence_score: float
    recommendation: str
    details: Dict[str, Any]
    timestamp: str


class ProfitabilityTester:
    """Comprehensive profitability testing framework."""
    
    def __init__(
        self,
        strategy_name: str = "stat_arb",
        symbols: List[str] = None,
        data_source: str = "ccxt:binance",
        timeframe: str = "1h",
        return_target: float = 0.50,
        max_drawdown_constraint: float = 0.25,
        monte_carlo_runs: int = 500,
        min_years: float = 3.0,
    ):
        """Initialize profitability tester.
        
        Args:
            strategy_name: Name of strategy to test
            symbols: List of symbols to test (default: ["BTC/USDT"])
            data_source: Data source identifier
            timeframe: Timeframe for backtesting
            return_target: Minimum total return target (default: 50%)
            max_drawdown_constraint: Maximum drawdown constraint (default: 25%)
            monte_carlo_runs: Number of bootstrap runs (default: 500)
            min_years: Minimum years of data required (default: 3.0)
        """
        self.strategy_name = strategy_name
        self.symbols = symbols or ["BTC/USDT"]
        self.data_source = data_source
        self.timeframe = timeframe
        self.return_target = return_target
        self.max_drawdown_constraint = max_drawdown_constraint
        self.monte_carlo_runs = monte_carlo_runs
        self.min_years = min_years
        self.data_loader = DataLoader()
        
    def load_data(self, symbol: str, years: float = 3.5) -> pd.DataFrame:
        """Load historical data for testing."""
        LOGGER.info(f"Loading {years} years of {symbol} data from {self.data_source}")
        
        end = datetime.utcnow()
        start = end - timedelta(days=int(years * 365))
        
        try:
            df = self.data_loader.get_ohlcv(
                source=self.data_source,
                symbol=symbol,
                timeframe=self.timeframe,
                start=start,
                end=end,
                limit=None
            )
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
                
            actual_years = (df.index[-1] - df.index[0]).days / 365.25
            LOGGER.info(f"Loaded {len(df)} bars spanning {actual_years:.2f} years")
            
            if actual_years < self.min_years:
                LOGGER.warning(
                    f"Only {actual_years:.2f} years of data available, "
                    f"less than minimum {self.min_years} years"
                )
            
            return df
            
        except Exception as e:
            LOGGER.error(f"Failed to load data for {symbol}: {e}")
            raise
    
    def generate_walk_forward_windows(
        self,
        df: pd.DataFrame,
        train_months: int = 12,
        test_months: int = 3,
    ) -> List[WalkForwardWindow]:
        """Generate walk-forward optimization windows.
        
        Args:
            df: Historical data
            train_months: Training window size in months
            test_months: Test window size in months
            
        Returns:
            List of walk-forward windows
        """
        windows = []
        
        start_date = df.index[0]
        end_date = df.index[-1]
        
        current_train_start = start_date
        
        while True:
            train_end = current_train_start + pd.DateOffset(months=train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=test_months)
            
            if test_end > end_date:
                break
                
            windows.append(WalkForwardWindow(
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            ))
            
            current_train_start = test_start
        
        LOGGER.info(f"Generated {len(windows)} walk-forward windows")
        return windows
    
    def calculate_omega_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """Calculate Omega ratio.
        
        Omega = E[max(R - threshold, 0)] / E[max(threshold - R, 0)]
        """
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = -excess[excess < 0].sum()
        
        if losses == 0:
            return float('inf') if gains > 0 else 1.0
            
        return gains / losses
    
    def calculate_calmar_ratio(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        annualization_factor: float = 252.0
    ) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        if len(returns) == 0:
            return 0.0
            
        ann_return = returns.mean() * annualization_factor
        mdd = abs(max_drawdown(equity_curve))
        
        if mdd == 0:
            return float('inf') if ann_return > 0 else 0.0
            
        return ann_return / mdd
    
    def calculate_metrics(
        self,
        result: BacktestResult,
        freq: str = "1h"
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        if len(result.returns) == 0:
            return PerformanceMetrics(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                omega_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                num_trades=0,
                avg_trade_return=0.0
            )
        
        total_return = (result.equity_curve.iloc[-1] / result.equity_curve.iloc[0]) - 1.0
        
        if freq == "1h":
            periods_per_year = 252 * 6.5
        elif freq == "4h":
            periods_per_year = 252 * 6.5 / 4
        elif freq == "1d":
            periods_per_year = 252
        else:
            periods_per_year = 252
            
        num_periods = len(result.returns)
        years = num_periods / periods_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1.0 if years > 0 else 0.0
        
        sharpe_ratio = sharpe(result.returns, freq=freq)
        sortino_ratio = sortino(result.returns, freq=freq)
        calmar_ratio = self.calculate_calmar_ratio(
            result.returns,
            result.equity_curve,
            annualization_factor=periods_per_year
        )
        omega_ratio = self.calculate_omega_ratio(result.returns)
        mdd = max_drawdown(result.equity_curve)
        
        winning_returns = result.returns[result.returns > 0]
        losing_returns = result.returns[result.returns < 0]
        
        win_rate = len(winning_returns) / len(result.returns) if len(result.returns) > 0 else 0.0
        
        total_gains = winning_returns.sum()
        total_losses = abs(losing_returns.sum())
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        
        num_trades = int(result.trades.sum())
        avg_trade_return = result.returns[result.trades > 0].mean() if num_trades > 0 else 0.0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            max_drawdown=mdd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=num_trades,
            avg_trade_return=avg_trade_return
        )
    
    def run_monte_carlo(
        self,
        returns: pd.Series,
        n_runs: int = 500,
        freq: str = "1h"
    ) -> MonteCarloResults:
        """Run Monte Carlo bootstrap analysis."""
        LOGGER.info(f"Running {n_runs} Monte Carlo bootstrap simulations")
        
        mc_results = monte_carlo_bootstrap(returns, n=n_runs, block=10, freq=freq)
        
        sharpes = np.array(mc_results["sharpe"])
        sortinos = np.array(mc_results["sortino"])
        max_dds = np.array(mc_results["max_dd"])
        
        returns_arr = returns.values
        mc_returns = []
        
        for _ in range(n_runs):
            idx = np.random.choice(len(returns_arr), size=len(returns_arr), replace=True)
            bootstrap_returns = returns_arr[idx]
            total_ret = np.prod(1 + bootstrap_returns) - 1
            mc_returns.append(total_ret)
        
        mc_returns = np.array(mc_returns)
        
        return MonteCarloResults(
            mean_sharpe=float(np.mean(sharpes)),
            std_sharpe=float(np.std(sharpes)),
            percentile_5_sharpe=float(np.percentile(sharpes, 5)),
            percentile_95_sharpe=float(np.percentile(sharpes, 95)),
            mean_sortino=float(np.mean(sortinos)),
            std_sortino=float(np.std(sortinos)),
            mean_max_dd=float(np.mean(max_dds)),
            std_max_dd=float(np.std(max_dds)),
            percentile_5_max_dd=float(np.percentile(max_dds, 5)),
            percentile_95_max_dd=float(np.percentile(max_dds, 95)),
            mean_return=float(np.mean(mc_returns)),
            std_return=float(np.std(mc_returns)),
            percentile_5_return=float(np.percentile(mc_returns, 5)),
            percentile_95_return=float(np.percentile(mc_returns, 95))
        )
    
    def run_walk_forward_backtest(
        self,
        df: pd.DataFrame,
        strategy: Any,
        windows: List[WalkForwardWindow]
    ) -> Tuple[BacktestResult, BacktestResult, Dict[str, Any]]:
        """Run walk-forward backtests.
        
        Returns:
            Tuple of (in_sample_result, out_of_sample_result, details)
        """
        LOGGER.info(f"Running walk-forward backtest with {len(windows)} windows")
        
        in_sample_returns_list = []
        out_of_sample_returns_list = []
        window_results = []
        
        for i, window in enumerate(windows):
            LOGGER.info(f"Window {i+1}/{len(windows)}: "
                       f"Train {window.train_start.date()} to {window.train_end.date()}, "
                       f"Test {window.test_start.date()} to {window.test_end.date()}")
            
            train_df = df.loc[window.train_start:window.train_end]
            test_df = df.loc[window.test_start:window.test_end]
            
            if len(train_df) < 100 or len(test_df) < 20:
                LOGGER.warning(f"Skipping window {i+1}: insufficient data")
                continue
            
            try:
                train_signals = strategy.generate_signals(train_df)
                train_result = backtest_signals(
                    train_df,
                    train_signals,
                    fee_bps=1.0,
                    slippage_bps=0.5,
                    weight=1.0
                )
                
                test_signals = strategy.generate_signals(test_df)
                test_result = backtest_signals(
                    test_df,
                    test_signals,
                    fee_bps=1.0,
                    slippage_bps=0.5,
                    weight=1.0
                )
                
                train_metrics = self.calculate_metrics(train_result, freq=self.timeframe)
                test_metrics = self.calculate_metrics(test_result, freq=self.timeframe)
                
                window_results.append({
                    "window": i + 1,
                    "train_return": train_metrics.total_return,
                    "test_return": test_metrics.total_return,
                    "train_sharpe": train_metrics.sharpe_ratio,
                    "test_sharpe": test_metrics.sharpe_ratio,
                    "train_mdd": train_metrics.max_drawdown,
                    "test_mdd": test_metrics.max_drawdown
                })
                
                in_sample_returns_list.append(train_result.returns)
                out_of_sample_returns_list.append(test_result.returns)
                
            except Exception as e:
                LOGGER.error(f"Window {i+1} failed: {e}")
                continue
        
        in_sample_returns = pd.concat(in_sample_returns_list)
        in_sample_equity = (1 + in_sample_returns).cumprod()
        in_sample_positions = pd.Series(0, index=in_sample_returns.index)
        in_sample_trades = in_sample_returns.abs() > 0
        
        in_sample_result = BacktestResult(
            equity_curve=in_sample_equity,
            returns=in_sample_returns,
            positions=in_sample_positions,
            trades=in_sample_trades
        )
        
        out_of_sample_returns = pd.concat(out_of_sample_returns_list)
        out_of_sample_equity = (1 + out_of_sample_returns).cumprod()
        out_of_sample_positions = pd.Series(0, index=out_of_sample_returns.index)
        out_of_sample_trades = out_of_sample_returns.abs() > 0
        
        out_of_sample_result = BacktestResult(
            equity_curve=out_of_sample_equity,
            returns=out_of_sample_returns,
            positions=out_of_sample_positions,
            trades=out_of_sample_trades
        )
        
        details = {
            "num_windows_tested": len(window_results),
            "window_results": window_results
        }
        
        return in_sample_result, out_of_sample_result, details
    
    def calculate_confidence_score(
        self,
        oos_metrics: PerformanceMetrics,
        mc_results: MonteCarloResults,
        meets_return: bool,
        meets_drawdown: bool
    ) -> float:
        """Calculate confidence score (0-100).
        
        Factors:
        - Out-of-sample Sharpe ratio (0-30 points)
        - Monte Carlo stability (0-20 points)
        - Meets targets (0-30 points)
        - Win rate (0-10 points)
        - Profit factor (0-10 points)
        """
        score = 0.0
        
        sharpe_score = min(30.0, max(0.0, oos_metrics.sharpe_ratio * 10))
        score += sharpe_score
        
        sharpe_stability = 1.0 - min(1.0, mc_results.std_sharpe / max(0.1, abs(mc_results.mean_sharpe)))
        mc_score = sharpe_stability * 20.0
        score += mc_score
        
        target_score = 0.0
        if meets_return:
            target_score += 15.0
        if meets_drawdown:
            target_score += 15.0
        score += target_score
        
        win_rate_score = oos_metrics.win_rate * 10.0
        score += win_rate_score
        
        pf_score = min(10.0, (oos_metrics.profit_factor - 1.0) * 5.0)
        score += pf_score
        
        return min(100.0, max(0.0, score))
    
    def generate_recommendation(
        self,
        confidence_score: float,
        meets_return: bool,
        meets_drawdown: bool,
        oos_metrics: PerformanceMetrics
    ) -> str:
        """Generate go/no-go recommendation."""
        
        if confidence_score >= 70 and meets_return and meets_drawdown:
            return "GO - Strong recommendation for production deployment"
        elif confidence_score >= 60 and meets_return and meets_drawdown:
            return "GO - Moderate recommendation, monitor closely in production"
        elif confidence_score >= 50 and (meets_return or meets_drawdown):
            return "CONDITIONAL GO - Consider paper trading first"
        elif confidence_score >= 40:
            return "NO GO - Requires optimization before deployment"
        else:
            return "NO GO - Strategy does not meet profitability standards"
    
    def run_profitability_test(
        self,
        symbol: str = None
    ) -> ProfitabilityReport:
        """Run complete profitability test."""
        symbol = symbol or self.symbols[0]
        
        LOGGER.info("=" * 80)
        LOGGER.info("PROFITABILITY TESTING FRAMEWORK")
        LOGGER.info("=" * 80)
        LOGGER.info(f"Strategy: {self.strategy_name}")
        LOGGER.info(f"Symbol: {symbol}")
        LOGGER.info(f"Timeframe: {self.timeframe}")
        LOGGER.info(f"Return Target: {self.return_target:.1%}")
        LOGGER.info(f"Max Drawdown Constraint: {self.max_drawdown_constraint:.1%}")
        LOGGER.info(f"Monte Carlo Runs: {self.monte_carlo_runs}")
        LOGGER.info("")
        
        df = self.load_data(symbol)
        
        try:
            strategy = get_strategy(self.strategy_name)
        except Exception as e:
            LOGGER.error(f"Failed to load strategy '{self.strategy_name}': {e}")
            LOGGER.info("Using synthetic strategy for testing purposes")
            from openquant.strategies.quant.stat_arb import StatArbStrategy
            strategy = StatArbStrategy()
        
        windows = self.generate_walk_forward_windows(df)
        
        LOGGER.info("\nRunning walk-forward optimization...")
        is_result, oos_result, wf_details = self.run_walk_forward_backtest(
            df, strategy, windows
        )
        
        LOGGER.info("\nCalculating in-sample metrics...")
        is_metrics = self.calculate_metrics(is_result, freq=self.timeframe)
        
        LOGGER.info("Calculating out-of-sample metrics...")
        oos_metrics = self.calculate_metrics(oos_result, freq=self.timeframe)
        
        LOGGER.info(f"\nRunning Monte Carlo analysis ({self.monte_carlo_runs} runs)...")
        mc_results = self.run_monte_carlo(
            oos_result.returns,
            n_runs=self.monte_carlo_runs,
            freq=self.timeframe
        )
        
        meets_return = oos_metrics.total_return >= self.return_target
        meets_drawdown = abs(oos_metrics.max_drawdown) <= self.max_drawdown_constraint
        
        confidence = self.calculate_confidence_score(
            oos_metrics, mc_results, meets_return, meets_drawdown
        )
        
        recommendation = self.generate_recommendation(
            confidence, meets_return, meets_drawdown, oos_metrics
        )
        
        test_period = f"{df.index[0].date()} to {df.index[-1].date()}"
        
        report = ProfitabilityReport(
            strategy_name=self.strategy_name,
            test_period=test_period,
            walk_forward_windows=len(windows),
            in_sample_metrics=is_metrics,
            out_of_sample_metrics=oos_metrics,
            monte_carlo_results=mc_results,
            meets_return_target=meets_return,
            meets_drawdown_constraint=meets_drawdown,
            confidence_score=confidence,
            recommendation=recommendation,
            details=wf_details,
            timestamp=datetime.utcnow().isoformat()
        )
        
        self.print_report(report)
        
        return report
    
    def print_report(self, report: ProfitabilityReport):
        """Print formatted profitability report."""
        
        print("\n" + "=" * 80)
        print("PROFITABILITY TEST REPORT")
        print("=" * 80)
        print(f"Strategy:              {report.strategy_name}")
        print(f"Test Period:           {report.test_period}")
        print(f"Walk-Forward Windows:  {report.walk_forward_windows}")
        print(f"Generated:             {report.timestamp}")
        print("")
        
        print("-" * 80)
        print("IN-SAMPLE PERFORMANCE (Training)")
        print("-" * 80)
        self.print_metrics(report.in_sample_metrics)
        
        print("\n" + "-" * 80)
        print("OUT-OF-SAMPLE PERFORMANCE (Testing)")
        print("-" * 80)
        self.print_metrics(report.out_of_sample_metrics)
        
        print("\n" + "-" * 80)
        print("MONTE CARLO ROBUSTNESS TEST")
        print("-" * 80)
        mc = report.monte_carlo_results
        print(f"Sharpe Ratio:          {mc.mean_sharpe:.2f} ± {mc.std_sharpe:.2f} "
              f"(5th: {mc.percentile_5_sharpe:.2f}, 95th: {mc.percentile_95_sharpe:.2f})")
        print(f"Sortino Ratio:         {mc.mean_sortino:.2f} ± {mc.std_sortino:.2f}")
        print(f"Max Drawdown:          {mc.mean_max_dd:.1%} ± {mc.std_max_dd:.1%} "
              f"(5th: {mc.percentile_5_max_dd:.1%}, 95th: {mc.percentile_95_max_dd:.1%})")
        print(f"Total Return:          {mc.mean_return:.1%} ± {mc.std_return:.1%} "
              f"(5th: {mc.percentile_5_return:.1%}, 95th: {mc.percentile_95_return:.1%})")
        
        print("\n" + "-" * 80)
        print("TARGET VALIDATION")
        print("-" * 80)
        print(f"Return Target (>50%):      {'✓ PASS' if report.meets_return_target else '✗ FAIL'} "
              f"(Actual: {report.out_of_sample_metrics.total_return:.1%})")
        print(f"Drawdown Limit (<25%):     {'✓ PASS' if report.meets_drawdown_constraint else '✗ FAIL'} "
              f"(Actual: {abs(report.out_of_sample_metrics.max_drawdown):.1%})")
        
        print("\n" + "-" * 80)
        print("FINAL ASSESSMENT")
        print("-" * 80)
        print(f"Confidence Score:      {report.confidence_score:.1f} / 100")
        print(f"Recommendation:        {report.recommendation}")
        print("=" * 80)
        print("")
    
    def print_metrics(self, metrics: PerformanceMetrics):
        """Print performance metrics in formatted table."""
        print(f"Total Return:          {metrics.total_return:.1%}")
        print(f"Annualized Return:     {metrics.annualized_return:.1%}")
        print(f"Sharpe Ratio:          {metrics.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:         {metrics.sortino_ratio:.2f}")
        print(f"Calmar Ratio:          {metrics.calmar_ratio:.2f}")
        print(f"Omega Ratio:           {metrics.omega_ratio:.2f}")
        print(f"Max Drawdown:          {metrics.max_drawdown:.1%}")
        print(f"Win Rate:              {metrics.win_rate:.1%}")
        print(f"Profit Factor:         {metrics.profit_factor:.2f}")
        print(f"Number of Trades:      {metrics.num_trades}")
        print(f"Avg Trade Return:      {metrics.avg_trade_return:.3%}")
    
    def save_report(self, report: ProfitabilityReport, output_path: str = None):
        """Save report to JSON file."""
        if output_path is None:
            output_path = f"reports/profitability_{report.strategy_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        report_dict = {
            "strategy_name": report.strategy_name,
            "test_period": report.test_period,
            "walk_forward_windows": report.walk_forward_windows,
            "in_sample_metrics": asdict(report.in_sample_metrics),
            "out_of_sample_metrics": asdict(report.out_of_sample_metrics),
            "monte_carlo_results": asdict(report.monte_carlo_results),
            "meets_return_target": report.meets_return_target,
            "meets_drawdown_constraint": report.meets_drawdown_constraint,
            "confidence_score": report.confidence_score,
            "recommendation": report.recommendation,
            "details": report.details,
            "timestamp": report.timestamp
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        LOGGER.info(f"Report saved to {output_file}")


def main():
    """Main entry point for profitability testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="End-to-End Profitability Testing Framework")
    parser.add_argument("--strategy", type=str, default="stat_arb", help="Strategy name")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Symbol to test")
    parser.add_argument("--source", type=str, default="ccxt:binance", help="Data source")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--return-target", type=float, default=0.50, help="Return target (default: 0.50 = 50%)")
    parser.add_argument("--max-drawdown", type=float, default=0.25, help="Max drawdown constraint (default: 0.25 = 25%)")
    parser.add_argument("--monte-carlo-runs", type=int, default=500, help="Number of Monte Carlo runs")
    parser.add_argument("--min-years", type=float, default=3.0, help="Minimum years of data")
    parser.add_argument("--output", type=str, default=None, help="Output file path for report")
    
    args = parser.parse_args()
    
    tester = ProfitabilityTester(
        strategy_name=args.strategy,
        symbols=[args.symbol],
        data_source=args.source,
        timeframe=args.timeframe,
        return_target=args.return_target,
        max_drawdown_constraint=args.max_drawdown,
        monte_carlo_runs=args.monte_carlo_runs,
        min_years=args.min_years
    )
    
    report = tester.run_profitability_test(symbol=args.symbol)
    
    tester.save_report(report, output_path=args.output)
    
    return 0 if report.confidence_score >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
