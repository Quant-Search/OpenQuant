"""Strategy Backtesting Comparison Report.

Generates comprehensive side-by-side comparisons of multiple strategies:
- Metrics tables (Sharpe, Sortino, Max DD, Win Rate, etc.)
- Equity curve overlays
- Drawdown overlays
- Statistical tests (t-test, Diebold-Mariano) for strategy selection
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from ..backtest.engine import BacktestResult
from ..backtest.metrics import (
    sharpe, sortino, max_drawdown, win_rate, profit_factor, cvar
)
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class StrategyComparisonResult:
    metrics_table: pd.DataFrame
    equity_curves: pd.DataFrame
    drawdown_curves: pd.DataFrame
    statistical_tests: Dict[str, Any]
    correlation_matrix: pd.DataFrame
    ranked_strategies: pd.DataFrame


def calculate_metrics(result: BacktestResult, freq: str = "1h") -> Dict[str, float]:
    """Calculate comprehensive performance metrics for a backtest result."""
    returns = result.returns.dropna()
    equity = result.equity_curve.dropna()
    
    if returns.empty or equity.empty:
        return {
            "Total Return (%)": 0.0,
            "CAGR (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Max Drawdown (%)": 0.0,
            "CVaR 95% (%)": 0.0,
            "Win Rate (%)": 0.0,
            "Profit Factor": 0.0,
            "Avg Win (%)": 0.0,
            "Avg Loss (%)": 0.0,
            "Total Trades": 0,
            "Final Equity": 1.0,
            "Calmar Ratio": 0.0,
            "Volatility (%)": 0.0,
            "Skewness": 0.0,
            "Kurtosis": 0.0,
        }
    
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    
    years = len(returns) / (252 * 6.5) if freq == "1h" else len(returns) / 252
    years = max(years, 1/252)
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100
    
    sharpe_val = sharpe(returns, freq=freq)
    sortino_val = sortino(returns, freq=freq)
    mdd = max_drawdown(equity) * 100
    cvar_val = cvar(returns, alpha=0.95) * 100
    wr = win_rate(returns) * 100
    pf = profit_factor(returns)
    
    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]
    avg_win = winning_trades.mean() * 100 if not winning_trades.empty else 0.0
    avg_loss = losing_trades.mean() * 100 if not losing_trades.empty else 0.0
    
    total_trades = int(result.trades.sum()) if hasattr(result, 'trades') else 0
    
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    
    volatility = returns.std() * np.sqrt(252 * 6.5 if freq == "1h" else 252) * 100
    
    try:
        skew = float(stats.skew(returns.dropna()))
        kurt = float(stats.kurtosis(returns.dropna()))
    except:
        skew = 0.0
        kurt = 0.0
    
    return {
        "Total Return (%)": round(total_return, 2),
        "CAGR (%)": round(cagr, 2),
        "Sharpe Ratio": round(sharpe_val, 2),
        "Sortino Ratio": round(sortino_val, 2),
        "Max Drawdown (%)": round(mdd, 2),
        "CVaR 95% (%)": round(cvar_val, 2),
        "Win Rate (%)": round(wr, 2),
        "Profit Factor": round(pf, 2),
        "Avg Win (%)": round(avg_win, 2),
        "Avg Loss (%)": round(avg_loss, 2),
        "Total Trades": total_trades,
        "Final Equity": round(equity.iloc[-1], 4),
        "Calmar Ratio": round(calmar, 2),
        "Volatility (%)": round(volatility, 2),
        "Skewness": round(skew, 2),
        "Kurtosis": round(kurt, 2),
    }


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Calculate drawdown series from equity curve."""
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak * 100
    return drawdown


def t_test_comparison(
    returns_a: pd.Series,
    returns_b: pd.Series
) -> Dict[str, float]:
    """Perform paired t-test comparing two strategies' returns.
    
    H0: Mean returns are equal
    H1: Mean returns are different
    """
    returns_a_clean = returns_a.dropna()
    returns_b_clean = returns_b.dropna()
    
    common_index = returns_a_clean.index.intersection(returns_b_clean.index)
    
    if len(common_index) < 2:
        return {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "mean_diff": 0.0,
            "significant": False
        }
    
    r_a = returns_a_clean.loc[common_index]
    r_b = returns_b_clean.loc[common_index]
    
    t_stat, p_val = stats.ttest_rel(r_a, r_b)
    mean_diff = r_a.mean() - r_b.mean()
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "mean_diff": float(mean_diff),
        "significant": p_val < 0.05
    }


def diebold_mariano_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    h: int = 1
) -> Dict[str, float]:
    """Diebold-Mariano test for predictive accuracy.
    
    Tests whether the forecast accuracy of two strategies differs significantly.
    H0: Two strategies have equal predictive accuracy
    H1: Strategies have different predictive accuracy
    
    Args:
        returns_a: Returns from strategy A
        returns_b: Returns from strategy B
        h: Forecast horizon (default: 1)
        
    Returns:
        Dictionary with DM statistic, p-value, and significance
    """
    returns_a_clean = returns_a.dropna()
    returns_b_clean = returns_b.dropna()
    
    common_index = returns_a_clean.index.intersection(returns_b_clean.index)
    
    if len(common_index) < 2:
        return {
            "dm_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "better_strategy": "none"
        }
    
    r_a = returns_a_clean.loc[common_index].values
    r_b = returns_b_clean.loc[common_index].values
    
    e_a = r_a ** 2
    e_b = r_b ** 2
    
    d = e_a - e_b
    
    mean_d = np.mean(d)
    
    def autocovariance(x, lag):
        n = len(x)
        x_mean = np.mean(x)
        if lag == 0:
            return np.sum((x - x_mean) ** 2) / n
        else:
            return np.sum((x[:-lag] - x_mean) * (x[lag:] - x_mean)) / n
    
    gamma_0 = autocovariance(d, 0)
    
    var_d = gamma_0
    for k in range(1, h):
        gamma_k = autocovariance(d, k)
        var_d += 2 * gamma_k
    
    var_d = var_d / len(d)
    
    if var_d <= 0:
        return {
            "dm_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "better_strategy": "none"
        }
    
    dm_stat = mean_d / np.sqrt(var_d)
    
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    better = "none"
    if p_value < 0.05:
        better = "strategy_a" if dm_stat > 0 else "strategy_b"
    
    return {
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "better_strategy": better
    }


def calculate_return_correlation(
    results: Dict[str, BacktestResult]
) -> pd.DataFrame:
    """Calculate correlation matrix of strategy returns."""
    returns_dict = {}
    
    for name, result in results.items():
        returns_dict[name] = result.returns
    
    returns_df = pd.DataFrame(returns_dict)
    correlation = returns_df.corr()
    
    return correlation


def rank_strategies(
    metrics_table: pd.DataFrame,
    ranking_method: str = "composite"
) -> pd.DataFrame:
    """Rank strategies based on performance metrics.
    
    Args:
        metrics_table: DataFrame with strategies as columns and metrics as rows
        ranking_method: Method for ranking
            - "composite": Weighted average of multiple metrics
            - "sharpe": Rank by Sharpe ratio only
            - "calmar": Rank by Calmar ratio only
            - "custom": Custom weighted scoring
            
    Returns:
        DataFrame with rankings and scores
    """
    if ranking_method == "sharpe":
        scores = metrics_table.loc["Sharpe Ratio"]
        ranks = scores.rank(ascending=False)
        
    elif ranking_method == "calmar":
        scores = metrics_table.loc["Calmar Ratio"]
        ranks = scores.rank(ascending=False)
        
    else:
        weights = {
            "Sharpe Ratio": 0.25,
            "Sortino Ratio": 0.15,
            "Calmar Ratio": 0.20,
            "Win Rate (%)": 0.10,
            "Profit Factor": 0.15,
            "Max Drawdown (%)": 0.15
        }
        
        normalized = pd.DataFrame()
        for metric, weight in weights.items():
            if metric in metrics_table.index:
                values = metrics_table.loc[metric]
                
                if metric == "Max Drawdown (%)":
                    norm_values = (values.max() - values) / (values.max() - values.min() + 1e-9)
                else:
                    norm_values = (values - values.min()) / (values.max() - values.min() + 1e-9)
                
                normalized[metric] = norm_values * weight
        
        scores = normalized.sum(axis=1)
        ranks = scores.rank(ascending=False)
    
    ranking_df = pd.DataFrame({
        "Rank": ranks.astype(int),
        "Score": scores.round(4)
    })
    
    ranking_df = ranking_df.sort_values("Rank")
    
    return ranking_df


def compare_strategies(
    results: Dict[str, BacktestResult],
    freq: str = "1h",
    ranking_method: str = "composite"
) -> StrategyComparisonResult:
    """Compare multiple strategy backtest results.
    
    Args:
        results: Dictionary mapping strategy names to BacktestResult objects
        freq: Data frequency for annualization ("1h", "4h", "1d")
        ranking_method: Method for ranking strategies
        
    Returns:
        StrategyComparisonResult with comprehensive comparison data
    """
    if not results:
        LOGGER.warning("No results provided for comparison")
        return StrategyComparisonResult(
            metrics_table=pd.DataFrame(),
            equity_curves=pd.DataFrame(),
            drawdown_curves=pd.DataFrame(),
            statistical_tests={},
            correlation_matrix=pd.DataFrame(),
            ranked_strategies=pd.DataFrame()
        )
    
    LOGGER.info(f"Comparing {len(results)} strategies...")
    
    metrics_dict = {}
    equity_dict = {}
    drawdown_dict = {}
    
    for name, result in results.items():
        metrics_dict[name] = calculate_metrics(result, freq=freq)
        equity_dict[name] = result.equity_curve
        drawdown_dict[name] = calculate_drawdown_series(result.equity_curve)
    
    metrics_table = pd.DataFrame(metrics_dict)
    
    equity_curves = pd.DataFrame(equity_dict)
    drawdown_curves = pd.DataFrame(drawdown_dict)
    
    statistical_tests = {}
    strategy_names = list(results.keys())
    
    for i, name_a in enumerate(strategy_names):
        for name_b in strategy_names[i+1:]:
            test_key = f"{name_a}_vs_{name_b}"
            
            t_test = t_test_comparison(
                results[name_a].returns,
                results[name_b].returns
            )
            
            dm_test = diebold_mariano_test(
                results[name_a].returns,
                results[name_b].returns
            )
            
            statistical_tests[test_key] = {
                "t_test": t_test,
                "diebold_mariano": dm_test
            }
    
    correlation_matrix = calculate_return_correlation(results)
    
    ranked_strategies = rank_strategies(metrics_table, ranking_method=ranking_method)
    
    LOGGER.info("Strategy comparison complete")
    
    return StrategyComparisonResult(
        metrics_table=metrics_table,
        equity_curves=equity_curves,
        drawdown_curves=drawdown_curves,
        statistical_tests=statistical_tests,
        correlation_matrix=correlation_matrix,
        ranked_strategies=ranked_strategies
    )


def generate_comparison_report(
    comparison: StrategyComparisonResult,
    output_path: Optional[Path] = None,
    include_plots: bool = True
) -> str:
    """Generate a formatted text report from comparison results.
    
    Args:
        comparison: StrategyComparisonResult object
        output_path: Optional path to save the report
        include_plots: Whether to generate plot files (requires matplotlib)
        
    Returns:
        Report text as string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("STRATEGY BACKTESTING COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("STRATEGY RANKINGS")
    lines.append("-" * 80)
    lines.append(comparison.ranked_strategies.to_string())
    lines.append("")
    
    lines.append("PERFORMANCE METRICS")
    lines.append("-" * 80)
    lines.append(comparison.metrics_table.to_string())
    lines.append("")
    
    lines.append("RETURN CORRELATIONS")
    lines.append("-" * 80)
    lines.append(comparison.correlation_matrix.to_string())
    lines.append("")
    
    lines.append("STATISTICAL TESTS")
    lines.append("-" * 80)
    
    for test_name, tests in comparison.statistical_tests.items():
        lines.append(f"\n{test_name}:")
        
        t_test = tests["t_test"]
        lines.append(f"  T-Test:")
        lines.append(f"    t-statistic: {t_test['t_statistic']:.4f}")
        lines.append(f"    p-value: {t_test['p_value']:.4f}")
        lines.append(f"    Significant: {t_test['significant']}")
        
        dm_test = tests["diebold_mariano"]
        lines.append(f"  Diebold-Mariano Test:")
        lines.append(f"    DM statistic: {dm_test['dm_statistic']:.4f}")
        lines.append(f"    p-value: {dm_test['p_value']:.4f}")
        lines.append(f"    Significant: {dm_test['significant']}")
        lines.append(f"    Better strategy: {dm_test['better_strategy']}")
    
    lines.append("")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        LOGGER.info(f"Report saved to {output_path}")
        
        if include_plots:
            try:
                _generate_plots(comparison, output_path.parent)
            except ImportError:
                LOGGER.warning("Matplotlib not available, skipping plots")
            except Exception as e:
                LOGGER.error(f"Failed to generate plots: {e}")
    
    return report_text


def _generate_plots(
    comparison: StrategyComparisonResult,
    output_dir: Path
):
    """Generate visualization plots for comparison.
    
    Requires matplotlib and plotly.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        LOGGER.warning("Matplotlib not available for plotting")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    for col in comparison.equity_curves.columns:
        ax1.plot(comparison.equity_curves.index, comparison.equity_curves[col], label=col)
    
    ax1.set_title("Equity Curves Comparison", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Equity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for col in comparison.drawdown_curves.columns:
        ax2.plot(comparison.drawdown_curves.index, comparison.drawdown_curves[col], label=col)
    
    ax2.set_title("Drawdown Comparison", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    plot_path = output_dir / "strategy_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"Plots saved to {plot_path}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    try:
        import seaborn as sns
        sns.heatmap(
            comparison.correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            ax=ax,
            square=True
        )
    except ImportError:
        im = ax.imshow(comparison.correlation_matrix, cmap='coolwarm', aspect='auto')
        ax.set_xticks(range(len(comparison.correlation_matrix.columns)))
        ax.set_yticks(range(len(comparison.correlation_matrix.index)))
        ax.set_xticklabels(comparison.correlation_matrix.columns, rotation=45)
        ax.set_yticklabels(comparison.correlation_matrix.index)
        plt.colorbar(im, ax=ax)
        
        for i in range(len(comparison.correlation_matrix.index)):
            for j in range(len(comparison.correlation_matrix.columns)):
                text = ax.text(j, i, f'{comparison.correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")
    
    ax.set_title("Strategy Returns Correlation Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    corr_path = output_dir / "correlation_matrix.png"
    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"Correlation matrix saved to {corr_path}")


def export_comparison_to_csv(
    comparison: StrategyComparisonResult,
    output_dir: Path
):
    """Export comparison results to CSV files.
    
    Args:
        comparison: StrategyComparisonResult object
        output_dir: Directory to save CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.metrics_table.to_csv(output_dir / "metrics_comparison.csv")
    comparison.equity_curves.to_csv(output_dir / "equity_curves.csv")
    comparison.drawdown_curves.to_csv(output_dir / "drawdown_curves.csv")
    comparison.correlation_matrix.to_csv(output_dir / "correlation_matrix.csv")
    comparison.ranked_strategies.to_csv(output_dir / "strategy_rankings.csv")
    
    tests_df = []
    for test_name, tests in comparison.statistical_tests.items():
        row = {"comparison": test_name}
        row.update({f"t_test_{k}": v for k, v in tests["t_test"].items()})
        row.update({f"dm_test_{k}": v for k, v in tests["diebold_mariano"].items()})
        tests_df.append(row)
    
    if tests_df:
        pd.DataFrame(tests_df).to_csv(output_dir / "statistical_tests.csv", index=False)
    
    LOGGER.info(f"Comparison data exported to {output_dir}")


def compare_strategies_from_file(
    results_path: Path,
    freq: str = "1h",
    ranking_method: str = "composite",
    output_dir: Optional[Path] = None
) -> StrategyComparisonResult:
    """Load backtest results from file and compare strategies.
    
    Expects a pickle or JSON file containing a dictionary of BacktestResult objects.
    
    Args:
        results_path: Path to file containing backtest results
        freq: Data frequency
        ranking_method: Ranking method
        output_dir: Optional directory for output files
        
    Returns:
        StrategyComparisonResult
    """
    results_path = Path(results_path)
    
    if results_path.suffix == '.pkl':
        import pickle
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    elif results_path.suffix == '.json':
        import json
        with open(results_path, 'r') as f:
            data = json.load(f)
        LOGGER.warning("JSON loading not fully implemented for BacktestResult objects")
        results = {}
    else:
        raise ValueError(f"Unsupported file format: {results_path.suffix}")
    
    comparison = compare_strategies(results, freq=freq, ranking_method=ranking_method)
    
    if output_dir:
        report_path = Path(output_dir) / "comparison_report.txt"
        generate_comparison_report(comparison, output_path=report_path, include_plots=True)
        export_comparison_to_csv(comparison, output_dir=Path(output_dir))
    
    return comparison
