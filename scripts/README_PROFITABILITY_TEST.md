# Profitability Testing Framework

## Overview

The `test_profitability.py` script provides a comprehensive end-to-end profitability testing framework for trading strategies. It performs walk-forward optimization, calculates multiple risk-adjusted metrics, runs Monte Carlo robustness tests, and generates go/no-go recommendations with confidence scores.

## Features

### 1. Walk-Forward Optimization
- Splits data into multiple training/testing windows (default: 12 months train, 3 months test)
- Prevents overfitting by validating on unseen out-of-sample data
- Tests strategy robustness across different market regimes

### 2. Risk-Adjusted Performance Metrics
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return relative to maximum drawdown
- **Omega Ratio**: Probability-weighted ratio of gains vs losses
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Max Drawdown**: Largest peak-to-trough decline

### 3. Monte Carlo Robustness Testing
- Runs 500+ bootstrap simulations (configurable)
- Provides confidence intervals for key metrics
- Tests strategy stability under different return sequences
- Calculates 5th and 95th percentile bounds

### 4. Target Validation
- **Return Target**: >50% total return (configurable)
- **Drawdown Constraint**: <25% maximum drawdown (configurable)
- Pass/fail validation for production deployment

### 5. Confidence Scoring & Recommendation
- 0-100 point confidence score based on:
  - Out-of-sample Sharpe ratio (30 points)
  - Monte Carlo stability (20 points)
  - Target achievement (30 points)
  - Win rate (10 points)
  - Profit factor (10 points)
- Actionable recommendations:
  - **GO - Strong**: Score ≥70, meets all targets
  - **GO - Moderate**: Score ≥60, meets all targets
  - **CONDITIONAL GO**: Score ≥50, paper trading recommended
  - **NO GO - Optimization**: Score ≥40
  - **NO GO - Fundamental**: Score <40

## Usage

### Basic Usage

```bash
python scripts/test_profitability.py --strategy stat_arb --symbol BTC/USDT
```

### Advanced Options

```bash
python scripts/test_profitability.py \
  --strategy kalman \
  --symbol ETH/USDT \
  --source ccxt:binance \
  --timeframe 4h \
  --return-target 0.60 \
  --max-drawdown 0.20 \
  --monte-carlo-runs 1000 \
  --min-years 3.5 \
  --output reports/my_test.json
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--strategy` | Strategy name from registry | `stat_arb` |
| `--symbol` | Trading symbol to test | `BTC/USDT` |
| `--source` | Data source (e.g., `ccxt:binance`) | `ccxt:binance` |
| `--timeframe` | Data timeframe (`1h`, `4h`, `1d`) | `1h` |
| `--return-target` | Minimum return target (0.50 = 50%) | `0.50` |
| `--max-drawdown` | Maximum drawdown allowed (0.25 = 25%) | `0.25` |
| `--monte-carlo-runs` | Number of bootstrap simulations | `500` |
| `--min-years` | Minimum years of historical data | `3.0` |
| `--output` | Output JSON file path | `reports/profitability_<strategy>_<timestamp>.json` |

## Available Strategies

The framework supports all strategies in the registry:

- `kalman` - Kalman Filter Mean Reversion
- `hurst` - Hurst Exponent Regime Detection
- `stat_arb` - Statistical Arbitrage (Cointegration)
- `liquidity` - Liquidity Provision Strategy
- `ml` - Machine Learning Strategy
- `mixer` - Ensemble of multiple strategies

## Output

### Console Output

The script prints a comprehensive report including:

```
================================================================================
PROFITABILITY TEST REPORT
================================================================================
Strategy:              stat_arb
Test Period:           2021-01-01 to 2024-06-30
Walk-Forward Windows:  12
Generated:             2024-01-15T10:30:00.000000

--------------------------------------------------------------------------------
IN-SAMPLE PERFORMANCE (Training)
--------------------------------------------------------------------------------
Total Return:          85.3%
Annualized Return:     23.4%
Sharpe Ratio:          1.85
Sortino Ratio:         2.42
Calmar Ratio:          1.23
Omega Ratio:           1.67
Max Drawdown:          -19.1%
Win Rate:              58.3%
Profit Factor:         1.89
Number of Trades:      342
Avg Trade Return:      0.249%

--------------------------------------------------------------------------------
OUT-OF-SAMPLE PERFORMANCE (Testing)
--------------------------------------------------------------------------------
Total Return:          56.7%
Annualized Return:     18.2%
Sharpe Ratio:          1.52
Sortino Ratio:         2.01
Calmar Ratio:          0.98
Omega Ratio:           1.45
Max Drawdown:          -18.6%
Win Rate:              55.1%
Profit Factor:         1.62
Number of Trades:      89
Avg Trade Return:      0.212%

--------------------------------------------------------------------------------
MONTE CARLO ROBUSTNESS TEST
--------------------------------------------------------------------------------
Sharpe Ratio:          1.52 ± 0.18 (5th: 1.23, 95th: 1.81)
Sortino Ratio:         2.01 ± 0.24
Max Drawdown:          18.6% ± 3.2% (5th: 13.1%, 95th: 24.7%)
Total Return:          56.7% ± 8.9% (5th: 42.3%, 95th: 71.2%)

--------------------------------------------------------------------------------
TARGET VALIDATION
--------------------------------------------------------------------------------
Return Target (>50%):      ✓ PASS (Actual: 56.7%)
Drawdown Limit (<25%):     ✓ PASS (Actual: 18.6%)

--------------------------------------------------------------------------------
FINAL ASSESSMENT
--------------------------------------------------------------------------------
Confidence Score:      73.5 / 100
Recommendation:        GO - Strong recommendation for production deployment
================================================================================
```

### JSON Output

A detailed JSON report is saved containing all metrics, Monte Carlo results, and window-by-window performance:

```json
{
  "strategy_name": "stat_arb",
  "test_period": "2021-01-01 to 2024-06-30",
  "walk_forward_windows": 12,
  "in_sample_metrics": { ... },
  "out_of_sample_metrics": { ... },
  "monte_carlo_results": { ... },
  "meets_return_target": true,
  "meets_drawdown_constraint": true,
  "confidence_score": 73.5,
  "recommendation": "GO - Strong recommendation for production deployment",
  "details": {
    "num_windows_tested": 12,
    "window_results": [ ... ]
  },
  "timestamp": "2024-01-15T10:30:00.000000"
}
```

## Integration with CI/CD

The script returns exit codes for automation:

- **Exit Code 0**: Confidence score ≥50 (deployment candidate)
- **Exit Code 1**: Confidence score <50 (requires work)

Example CI/CD integration:

```yaml
- name: Run Profitability Test
  run: |
    python scripts/test_profitability.py \
      --strategy ${{ matrix.strategy }} \
      --symbol ${{ matrix.symbol }} \
      --return-target 0.50 \
      --max-drawdown 0.25
  continue-on-error: false
```

## Interpreting Results

### Confidence Score Breakdown

| Component | Points | Description |
|-----------|--------|-------------|
| Sharpe Ratio | 0-30 | Higher Sharpe = better risk-adjusted returns |
| MC Stability | 0-20 | Lower std deviation in bootstrap = more robust |
| Target Achievement | 0-30 | Meeting both return and drawdown targets |
| Win Rate | 0-10 | Higher win rate = more consistent |
| Profit Factor | 0-10 | Ratio of gains to losses |

### Red Flags

Watch out for these warning signs:

1. **Large IS/OOS Gap**: If in-sample performance >> out-of-sample, strategy may be overfit
2. **High MC Variance**: Large standard deviations in Monte Carlo suggest instability
3. **Low Win Rate**: <45% win rate may indicate poor signal quality
4. **High Drawdown**: Max drawdown >25% is risky for most portfolios
5. **Few Trades**: <50 trades in OOS period may not be statistically significant

### Best Practices

1. **Test Multiple Timeframes**: Run tests on 1h, 4h, and 1d data
2. **Test Multiple Symbols**: Verify strategy works across different instruments
3. **Increase MC Runs**: Use 1000+ runs for more stable confidence intervals
4. **Review Window Results**: Check the `window_results` in JSON for consistency
5. **Paper Trade First**: Even with high confidence, start with paper trading

## Troubleshooting

### Data Loading Issues

If data loading fails:
- Verify internet connection
- Check exchange API is accessible
- Ensure symbol format is correct (e.g., `BTC/USDT` not `BTCUSDT`)
- Try alternative data source

### Strategy Errors

If strategy fails:
- Verify strategy exists in registry
- Check minimum data requirements for strategy
- Review strategy parameters in code

### Performance Issues

For faster execution:
- Reduce Monte Carlo runs to 100-200 for testing
- Use shorter data period (2 years minimum)
- Use larger timeframe (4h instead of 1h)

## Technical Details

### Walk-Forward Algorithm

1. Split data into overlapping windows
2. For each window:
   - Train on historical data
   - Test on forward period
   - Record performance
3. Combine all OOS results for final metrics

### Monte Carlo Bootstrap

1. Resample returns with replacement (block bootstrap)
2. Calculate equity curve and metrics
3. Repeat N times (default: 500)
4. Aggregate statistics (mean, std, percentiles)

### Confidence Score Formula

```
score = min(100, 
  sharpe_component +
  mc_stability_component +
  target_component +
  win_rate_component +
  profit_factor_component
)
```

## References

- Walk-Forward Analysis: Pardo (2008) "The Evaluation and Optimization of Trading Strategies"
- Monte Carlo Methods: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
- Risk Metrics: Bacon (2008) "Practical Portfolio Performance Measurement and Attribution"
