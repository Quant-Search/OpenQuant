# Strategy Comparison - Quick Reference

## 30-Second Start

```python
from openquant.backtest.engine import backtest_signals
from openquant.reporting import compare_strategies, generate_comparison_report

# 1. Run backtests
results = {
    "Strategy_A": backtest_signals(df, signals_a, fee_bps=1.0),
    "Strategy_B": backtest_signals(df, signals_b, fee_bps=1.0),
}

# 2. Compare
comparison = compare_strategies(results, freq="1h")

# 3. View rankings
print(comparison.ranked_strategies)

# 4. Generate full report
generate_comparison_report(comparison, output_path="report.txt")
```

## Key Functions

| Function | Purpose |
|----------|---------|
| `compare_strategies()` | Main comparison function - returns StrategyComparisonResult |
| `generate_comparison_report()` | Create text report with optional plots |
| `export_comparison_to_csv()` | Export all data to CSV files |
| `calculate_metrics()` | Calculate metrics for single strategy |
| `rank_strategies()` | Rank strategies by various methods |

## StrategyComparisonResult Attributes

```python
comparison.metrics_table         # DataFrame: metrics × strategies
comparison.equity_curves          # DataFrame: time × strategies
comparison.drawdown_curves        # DataFrame: time × strategies
comparison.statistical_tests      # Dict: pairwise test results
comparison.correlation_matrix     # DataFrame: strategy × strategy
comparison.ranked_strategies      # DataFrame: rankings & scores
```

## Ranking Methods

```python
# Composite (default) - weighted average of multiple metrics
compare_strategies(results, ranking_method="composite")

# Sharpe only
compare_strategies(results, ranking_method="sharpe")

# Calmar only
compare_strategies(results, ranking_method="calmar")
```

## Statistical Tests

```python
# Access test results
test = comparison.statistical_tests["Strategy_A_vs_Strategy_B"]

# T-test
print(test["t_test"]["p_value"])        # < 0.05 = significant
print(test["t_test"]["significant"])     # True/False

# Diebold-Mariano test
print(test["diebold_mariano"]["p_value"])
print(test["diebold_mariano"]["better_strategy"])  # "strategy_a", "strategy_b", or "none"
```

## Key Metrics Available

- Total Return (%), CAGR (%), Final Equity
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Max Drawdown (%), CVaR 95% (%), Volatility (%)
- Win Rate (%), Profit Factor
- Avg Win (%), Avg Loss (%)
- Total Trades, Skewness, Kurtosis

## Export Everything

```python
from pathlib import Path

# Full export
output_dir = Path("reports/comparison/")
generate_comparison_report(comparison, output_path=output_dir / "report.txt", include_plots=True)
export_comparison_to_csv(comparison, output_dir=output_dir)

# Creates:
# - report.txt (full analysis)
# - strategy_comparison.png (equity + drawdown plots)
# - correlation_matrix.png (heatmap)
# - metrics_comparison.csv
# - equity_curves.csv
# - drawdown_curves.csv
# - correlation_matrix.csv
# - strategy_rankings.csv
# - statistical_tests.csv
```

## Common Patterns

### Select Best Strategy
```python
best = comparison.ranked_strategies.iloc[0].name
print(f"Best strategy: {best}")
```

### Find Uncorrelated Pairs
```python
corr = comparison.correlation_matrix
for i, a in enumerate(corr.columns):
    for b in corr.columns[i+1:]:
        if abs(corr.loc[a, b]) < 0.3:
            print(f"{a} + {b} = good for ensemble")
```

### Filter by Statistical Significance
```python
for test_name, tests in comparison.statistical_tests.items():
    dm = tests["diebold_mariano"]
    if dm["significant"]:
        print(f"{dm['better_strategy']} wins: {test_name}")
```

## Full Example

See `scripts/strategy_comparison_example.py` for complete working example.

## Documentation

Full documentation: `openquant/reporting/README_STRATEGY_COMPARISON.md`
