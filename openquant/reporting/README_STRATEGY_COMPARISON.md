# Strategy Backtesting Comparison Report

Comprehensive toolkit for comparing multiple trading strategy backtest results with statistical rigor.

## Features

- **Side-by-side metrics comparison** - Sharpe, Sortino, Max DD, Win Rate, Profit Factor, and more
- **Equity curve overlays** - Visual comparison of strategy performance over time
- **Drawdown analysis** - Compare maximum drawdowns and recovery patterns
- **Statistical tests** - T-test and Diebold-Mariano test for statistical significance
- **Return correlation** - Identify diversification opportunities
- **Automated ranking** - Composite scoring system for strategy selection
- **Export capabilities** - CSV, text reports, and visualization plots

## Quick Start

```python
from openquant.backtest.engine import backtest_signals
from openquant.reporting.strategy_comparison import compare_strategies, generate_comparison_report

# Run backtests for multiple strategies
results = {
    "Strategy_A": backtest_signals(df, signals_a, fee_bps=1.0),
    "Strategy_B": backtest_signals(df, signals_b, fee_bps=1.0),
    "Strategy_C": backtest_signals(df, signals_c, fee_bps=1.0),
}

# Compare strategies
comparison = compare_strategies(
    results=results,
    freq="1h",
    ranking_method="composite"
)

# Generate report
generate_comparison_report(
    comparison=comparison,
    output_path="data/comparison_report.txt",
    include_plots=True
)
```

## Metrics Calculated

### Performance Metrics
- **Total Return (%)** - Cumulative return over backtest period
- **CAGR (%)** - Compound Annual Growth Rate
- **Final Equity** - Ending portfolio value

### Risk-Adjusted Metrics
- **Sharpe Ratio** - Risk-adjusted return (annualized)
- **Sortino Ratio** - Downside risk-adjusted return
- **Calmar Ratio** - Return over maximum drawdown
- **CVaR 95%** - Conditional Value at Risk (average tail loss)

### Risk Metrics
- **Max Drawdown (%)** - Largest peak-to-trough decline
- **Volatility (%)** - Annualized standard deviation
- **Skewness** - Distribution asymmetry
- **Kurtosis** - Tail thickness

### Trading Metrics
- **Win Rate (%)** - Percentage of profitable trades
- **Profit Factor** - Ratio of gross profits to gross losses
- **Avg Win (%)** - Average winning trade return
- **Avg Loss (%)** - Average losing trade return
- **Total Trades** - Number of trades executed

## Statistical Tests

### Paired T-Test
Tests whether two strategies have significantly different mean returns.

```python
# Automatically computed for all strategy pairs
t_test = comparison.statistical_tests["Strategy_A_vs_Strategy_B"]["t_test"]
print(f"P-value: {t_test['p_value']:.4f}")
print(f"Significant: {t_test['significant']}")
```

**Interpretation:**
- p < 0.05: Strategies have statistically different returns
- p ≥ 0.05: No significant difference detected

### Diebold-Mariano Test
Tests whether two strategies have significantly different forecast accuracy (predictive power).

```python
dm_test = comparison.statistical_tests["Strategy_A_vs_Strategy_B"]["diebold_mariano"]
print(f"P-value: {dm_test['p_value']:.4f}")
print(f"Better strategy: {dm_test['better_strategy']}")
```

**Interpretation:**
- p < 0.05: Strategies have different predictive accuracy
- better_strategy indicates which one is superior
- Used for model selection and validation

## Ranking Methods

### Composite (Default)
Weighted average of multiple metrics:
- Sharpe Ratio: 25%
- Sortino Ratio: 15%
- Calmar Ratio: 20%
- Win Rate: 10%
- Profit Factor: 15%
- Max Drawdown: 15% (inverted)

### Sharpe
Ranks strategies by Sharpe ratio only.

### Calmar
Ranks strategies by Calmar ratio (return/drawdown).

### Custom
Define your own weights:

```python
# Implementation would require extending rank_strategies function
# to accept custom weights parameter
```

## Usage Examples

### Basic Comparison

```python
import pandas as pd
from openquant.backtest.engine import backtest_signals
from openquant.reporting.strategy_comparison import compare_strategies

# Prepare data and signals
df = pd.read_csv("ohlcv_data.csv", index_col=0, parse_dates=True)
signals_ma = generate_ma_signals(df)
signals_rsi = generate_rsi_signals(df)

# Run backtests
results = {
    "MA_Crossover": backtest_signals(df, signals_ma, fee_bps=1.0),
    "RSI_Strategy": backtest_signals(df, signals_rsi, fee_bps=1.0),
}

# Compare
comparison = compare_strategies(results, freq="1h")

# View rankings
print(comparison.ranked_strategies)
```

### Full Report with Exports

```python
from pathlib import Path
from openquant.reporting.strategy_comparison import (
    compare_strategies,
    generate_comparison_report,
    export_comparison_to_csv,
)

# Compare strategies
comparison = compare_strategies(results, freq="1h", ranking_method="composite")

# Generate text report
generate_comparison_report(
    comparison=comparison,
    output_path=Path("reports/comparison.txt"),
    include_plots=True
)

# Export to CSV
export_comparison_to_csv(
    comparison=comparison,
    output_dir=Path("reports/csv/")
)

# Outputs:
# - comparison.txt (full text report)
# - strategy_comparison.png (equity/drawdown plots)
# - correlation_matrix.png (heatmap)
# - metrics_comparison.csv
# - equity_curves.csv
# - drawdown_curves.csv
# - correlation_matrix.csv
# - strategy_rankings.csv
# - statistical_tests.csv
```

### Analyzing Specific Test Results

```python
comparison = compare_strategies(results, freq="1h")

# Check all statistical tests
for test_name, tests in comparison.statistical_tests.items():
    print(f"\n{test_name}:")
    
    # T-test results
    t_test = tests["t_test"]
    if t_test["significant"]:
        print(f"  ✓ Significantly different returns (p={t_test['p_value']:.4f})")
    
    # DM-test results
    dm_test = tests["diebold_mariano"]
    if dm_test["significant"]:
        print(f"  ✓ {dm_test['better_strategy']} has superior predictive accuracy")
```

### Correlation Analysis

```python
comparison = compare_strategies(results, freq="1h")

# View correlation matrix
print("Return Correlations:")
print(comparison.correlation_matrix)

# Find low-correlation pairs for diversification
for i, strat_a in enumerate(comparison.correlation_matrix.columns):
    for strat_b in comparison.correlation_matrix.columns[i+1:]:
        corr = comparison.correlation_matrix.loc[strat_a, strat_b]
        if abs(corr) < 0.3:
            print(f"{strat_a} vs {strat_b}: {corr:.2f} (good for ensemble)")
```

## Advanced Features

### Loading from File

```python
from openquant.reporting.strategy_comparison import compare_strategies_from_file

# Load pre-computed backtest results
comparison = compare_strategies_from_file(
    results_path=Path("data/backtest_results.pkl"),
    freq="1h",
    ranking_method="composite",
    output_dir=Path("reports/")
)
```

### Custom Metric Calculation

```python
from openquant.reporting.strategy_comparison import calculate_metrics

# Calculate metrics for single strategy
metrics = calculate_metrics(backtest_result, freq="1h")

print(f"Sharpe: {metrics['Sharpe Ratio']}")
print(f"Max DD: {metrics['Max Drawdown (%)']}")
```

### Drawdown Analysis

```python
from openquant.reporting.strategy_comparison import calculate_drawdown_series

# Get drawdown series
equity = backtest_result.equity_curve
drawdown = calculate_drawdown_series(equity)

# Find worst drawdown periods
worst_periods = drawdown.nsmallest(10)
print(worst_periods)
```

## Best Practices

1. **Always use multiple metrics** - Don't rely solely on Sharpe or returns
2. **Check statistical significance** - Use t-test and DM test before concluding
3. **Consider correlation** - Low-correlation strategies are better for ensembles
4. **Account for transaction costs** - Include realistic fee_bps in backtests
5. **Use consistent timeframes** - Compare strategies on same data period
6. **Verify data quality** - Bad data leads to misleading comparisons
7. **Out-of-sample testing** - Reserve holdout period for final validation

## Integration with Research Pipeline

```python
# Example: Multi-strategy research workflow
from openquant.backtest.engine import backtest_signals
from openquant.reporting.strategy_comparison import compare_strategies
from openquant.strategies.mixer import StrategyMixer

# 1. Backtest multiple strategies
strategies = ["MA", "RSI", "BB", "MACD", "Momentum"]
results = {}

for name in strategies:
    signals = generate_signals(df, strategy=name)
    results[name] = backtest_signals(df, signals, fee_bps=1.0)

# 2. Compare and rank
comparison = compare_strategies(results, freq="1h")
top_3 = comparison.ranked_strategies.head(3).index.tolist()

print(f"Top 3 strategies: {top_3}")

# 3. Check diversification
corr_matrix = comparison.correlation_matrix.loc[top_3, top_3]
print("Correlation among top 3:")
print(corr_matrix)

# 4. Build ensemble from top performers
if (abs(corr_matrix) < 0.7).all().all():
    print("Low correlation - good for ensemble")
    # Proceed with mixer strategy
else:
    print("High correlation - select single best strategy")
```

## Troubleshooting

### Issue: Empty metrics table
**Cause:** Invalid or empty BacktestResult objects
**Solution:** Verify backtests completed successfully and have data

### Issue: NaN in statistical tests
**Cause:** Insufficient overlapping data between strategies
**Solution:** Ensure strategies have common index/timestamps

### Issue: Plots not generating
**Cause:** matplotlib not installed or import error
**Solution:** `pip install matplotlib seaborn`

### Issue: Low p-values but similar Sharpe ratios
**Cause:** Large sample size can make small differences significant
**Solution:** Consider economic significance alongside statistical significance

## References

- Diebold, F. X., & Mariano, R. S. (1995). "Comparing Predictive Accuracy"
- Harvey, C. R., & Liu, Y. (2015). "Backtesting"
- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning"

## See Also

- `openquant/backtest/engine.py` - Backtesting engine
- `openquant/backtest/metrics.py` - Individual metric calculations
- `openquant/strategies/mixer.py` - Strategy ensemble
- `scripts/strategy_comparison_example.py` - Complete example
