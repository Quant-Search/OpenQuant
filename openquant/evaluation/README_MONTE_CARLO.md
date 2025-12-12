# Monte Carlo Simulation for Strategy Robustness Testing

## Overview

The Monte Carlo simulation module provides comprehensive robustness testing for trading strategies through three complementary approaches:

1. **Path-Dependent Randomization**: Tests strategy performance on resampled price paths while preserving temporal structure
2. **Parameter Perturbation**: Evaluates strategy sensitivity to parameter changes
3. **Regime Shift Simulation**: Assesses strategy adaptability across different market conditions

## Features

### Path-Dependent Bootstrap
- Block bootstrap preserves autocorrelation structure
- Reconstructs OHLCV data from resampled returns
- Maintains realistic price dynamics
- Configurable block size for different time scales

### Parameter Perturbation
- Random noise addition to numeric parameters
- Configurable perturbation percentage
- Handles integer and float parameters appropriately
- Tests parameter robustness across multiple variations

### Regime Shift Simulation
- Volatility scaling (low/normal/high vol regimes)
- Trend injection (ranging/trending markets)
- Combined volatility-trend scenarios
- Multiple regime multipliers for comprehensive testing

## Usage

### Standalone Monte Carlo Analysis

```python
from openquant.evaluation import (
    run_comprehensive_mc,
    MonteCarloConfig,
    evaluate_robustness
)

# Configure Monte Carlo simulation
mc_config = MonteCarloConfig(
    n_simulations=500,           # Number of simulations
    block_size=20,               # Block size for bootstrap
    param_perturbation_pct=0.1,  # 10% parameter perturbation
    regime_shift_volatility_multipliers=[0.5, 1.0, 1.5, 2.0],
    regime_shift_trend_multipliers=[0.0, 0.5, 1.0, 1.5]
)

# Run comprehensive MC analysis
mc_results = run_comprehensive_mc(
    df=df,                       # OHLCV DataFrame
    strategy_factory=strategy_factory,  # Callable that returns strategy
    params={"param1": 10, "param2": 30},  # Strategy parameters
    config=mc_config,
    fee_bps=2.0,
    weight=1.0
)

# Evaluate robustness
robustness_eval = evaluate_robustness(mc_results)
print(f"Robustness Rating: {robustness_eval['rating']}")
print(f"Robustness Score: {robustness_eval['robustness_score']:.2f}")
```

### Integrated with Walk-Forward Optimization

```python
from openquant.evaluation import walk_forward_evaluate, WFOSpec

# Enable Monte Carlo in WFO
wfo_spec = WFOSpec(
    n_splits=4,
    train_frac=0.7,
    use_monte_carlo=True,        # Enable MC testing
    mc_n_simulations=100,         # Reduced for WFO efficiency
    mc_block_size=20,
    mc_param_perturbation_pct=0.1
)

# Run WFO with Monte Carlo
wfo_results = walk_forward_evaluate(
    df=df,
    strategy_factory=strategy_factory,
    param_grid=param_grid,
    wfo=wfo_spec
)

# Check robustness results
if "robustness_evaluation" in wfo_results:
    print(f"Rating: {wfo_results['robustness_evaluation']['rating']}")
```

### Individual MC Methods

```python
from openquant.evaluation import (
    run_path_dependent_mc,
    run_parameter_perturbation_mc,
    run_regime_shift_mc,
    MonteCarloConfig
)

config = MonteCarloConfig(n_simulations=200)

# Path-dependent bootstrap only
path_results = run_path_dependent_mc(df, strategy_factory, params, config)

# Parameter perturbation only
param_results = run_parameter_perturbation_mc(df, strategy_factory, params, config)

# Regime shift only
regime_results = run_regime_shift_mc(df, strategy_factory, params, config)
```

## Output Format

### MonteCarloResult

Each metric returns a `MonteCarloResult` with:
- `mean`: Average value across simulations
- `median`: Median value
- `std`: Standard deviation
- `percentile_5`: 5th percentile (worst case)
- `percentile_95`: 95th percentile (best case)
- `min`: Minimum value observed
- `max`: Maximum value observed
- `simulations`: List of all simulation values

### Robustness Evaluation

`evaluate_robustness()` returns:
- `robustness_score`: Normalized score 0-1
- `rating`: HIGHLY_ROBUST, ROBUST, MODERATE, or FRAGILE
- `details`: Dictionary with individual test results

## Configuration Options

### MonteCarloConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_simulations` | 500 | Number of Monte Carlo simulations |
| `block_size` | 20 | Block size for bootstrap (preserves autocorrelation) |
| `confidence_level` | 0.95 | Confidence level for intervals |
| `param_perturbation_pct` | 0.1 | Parameter perturbation percentage (10%) |
| `regime_shift_volatility_multipliers` | [0.5, 1.0, 1.5, 2.0] | Volatility regime multipliers |
| `regime_shift_trend_multipliers` | [0.0, 0.5, 1.0, 1.5] | Trend regime multipliers |

### WFOSpec MC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_monte_carlo` | False | Enable Monte Carlo testing in WFO |
| `mc_n_simulations` | 100 | Number of MC simulations (reduced for WFO) |
| `mc_block_size` | 20 | Block size for bootstrap |
| `mc_param_perturbation_pct` | 0.1 | Parameter perturbation percentage |

## Interpretation Guide

### Sharpe Ratio Metrics
- **5th Percentile > 0.5**: Highly robust strategy
- **5th Percentile > 0.0**: Strategy profitable in worst case
- **Mean > 0.5**: Generally good performance
- **High Std Dev**: Strategy performance is highly variable

### Robustness Ratings
- **HIGHLY_ROBUST** (≥0.8): Strategy performs well across all conditions
- **ROBUST** (≥0.6): Strategy is generally reliable
- **MODERATE** (≥0.4): Strategy has some weaknesses
- **FRAGILE** (<0.4): Strategy may not be production-ready

### Max Drawdown
- Lower is better (less negative)
- 5th percentile shows worst-case drawdown
- High variance indicates regime-dependent behavior

## Best Practices

1. **Block Size Selection**
   - Daily data: 10-30 days
   - Hourly data: 50-100 hours
   - Match to strategy's typical hold period

2. **Simulation Count**
   - Quick testing: 50-100 simulations
   - Standard analysis: 200-500 simulations
   - Publication-quality: 1000+ simulations

3. **Parameter Perturbation**
   - Start with 10% (0.1)
   - Increase to 20% (0.2) for stress testing
   - Decrease to 5% (0.05) for fine-tuned strategies

4. **Regime Testing**
   - Always test low/high volatility regimes
   - Include trend regimes for directional strategies
   - Test combined regimes for comprehensive analysis

5. **WFO Integration**
   - Use reduced simulation count (50-100) for efficiency
   - Run on final parameters after WFO optimization
   - Monitor robustness trends across WFO splits

## Performance Considerations

- **Path-Dependent MC**: Most computationally intensive
- **Parameter Perturbation**: Fast, good for initial testing
- **Regime Shift**: Fast, tests specific scenarios

For large-scale testing:
1. Start with parameter perturbation
2. Add regime shift testing
3. Use path-dependent bootstrap for final validation

## Examples

See `scripts/example_monte_carlo.py` for complete working examples of:
- Standalone Monte Carlo analysis
- WFO integration with Monte Carlo
- Custom configuration scenarios

## Integration with Other Modules

The Monte Carlo module works seamlessly with:
- **WFO Pipeline**: Automatic robustness testing after optimization
- **CPCV**: Compatible with combinatorially purged cross-validation
- **Deflated Sharpe**: Can be applied to MC results for multiple-testing adjustment
- **Regime Detection**: Use detected regimes to configure regime shift testing

## References

- Block Bootstrap: Künsch (1989) "The Jackknife and the Bootstrap"
- Parameter Robustness: López de Prado (2018) "Advances in Financial Machine Learning"
- Regime Testing: Aronson (2006) "Evidence-Based Technical Analysis"
