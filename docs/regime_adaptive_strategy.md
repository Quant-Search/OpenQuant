# Regime Adaptive Strategy

## Overview

The `RegimeAdaptiveStrategy` is a meta-strategy that dynamically routes to appropriate sub-strategies based on detected market regimes. It uses the `RegimeDetector` to classify market conditions and selects the most suitable trading approach.

## Architecture

### Regime Detection

The strategy uses `RegimeDetector` from `openquant.quant.regime_detector` to classify markets into:

- **Trending** (H > 0.55): Persistent directional movement
- **Mean-Reverting** (H < 0.45): Anti-persistent, range-bound behavior
- **High Volatility**: Elevated market uncertainty
- **Neutral** (0.45 ≤ H ≤ 0.55): Random walk behavior

### Sub-Strategy Routing

Based on the detected regime, the strategy routes to:

1. **Hurst Exponent Strategy** (`HurstExponentStrategy`)
   - Used for trending markets
   - Implements trend-following logic
   - Effective when H > 0.55

2. **Statistical Arbitrage Strategy** (`StatArbStrategy`)
   - Used for mean-reverting markets
   - Implements mean-reversion logic
   - Effective when H < 0.45

3. **Volatility Scaling**
   - During high volatility periods, exposure is reduced
   - Configurable reduction factor (default: 0.5)
   - Can be disabled entirely

## Usage

### Basic Example

```python
from openquant.strategies.regime_adaptive import RegimeAdaptiveStrategy
from openquant.backtest.engine import backtest_signals
import pandas as pd

# Create strategy
strategy = RegimeAdaptiveStrategy(
    lookback=100,
    hurst_threshold_trend=0.55,
    hurst_threshold_mr=0.45,
    vol_reduce_factor=0.5,
    enable_vol_scaling=True
)

# Generate signals
signals = strategy.generate_signals(df)

# Backtest
result = backtest_signals(df, signals, fee_bps=2.0, weight=1.0)

# Access regime history
regime_history = strategy.get_regime_history()
print(regime_history.tail())

# Get regime statistics
stats = strategy.get_regime_stats()
print(f"Mean Hurst: {stats['mean_hurst_exponent']:.3f}")
```

### Advanced Configuration

```python
# Custom sub-strategy parameters
hurst_params = {
    'lookback': 80,
    'trend_threshold': 0.6,
    'mr_threshold': 0.4
}

stat_arb_params = {
    'lookback': 80,
    'entry_z': 2.5,
    'exit_z': 0.5
}

strategy = RegimeAdaptiveStrategy(
    lookback=100,
    hurst_threshold_trend=0.6,
    hurst_threshold_mr=0.4,
    vol_reduce_factor=0.3,
    enable_vol_scaling=True,
    hurst_params=hurst_params,
    stat_arb_params=stat_arb_params
)
```

## Regime-Specific Backtesting

### Walk-Forward Optimization with Regime Tracking

```python
from openquant.evaluation.wfo import (
    walk_forward_evaluate_regime_specific,
    WFOSpec
)

def strategy_factory(lookback=100, vol_reduce_factor=0.5):
    return RegimeAdaptiveStrategy(
        lookback=lookback,
        vol_reduce_factor=vol_reduce_factor
    )

param_grid = {
    'lookback': [50, 100, 150],
    'vol_reduce_factor': [0.3, 0.5, 0.7]
}

wfo_spec = WFOSpec(n_splits=4, train_frac=0.7)

results = walk_forward_evaluate_regime_specific(
    df=df,
    strategy_factory=strategy_factory,
    param_grid=param_grid,
    fee_bps=2.0,
    weight=1.0,
    wfo=wfo_spec
)

# Access results
print(f"Mean Test Sharpe: {results['mean_test_sharpe']:.2f}")
print("\nRegime Performance:")
for regime, perf in results['regime_performance'].items():
    print(f"  {regime}: Sharpe={perf['sharpe']:.2f}")

print("\nRegime Distribution:")
for regime, dist in results['regime_distribution'].items():
    print(f"  {regime}: {dist['percentage']:.1f}%")
```

### Strategy Comparison by Regime

```python
from openquant.evaluation.wfo import compare_strategies_by_regime
from openquant.strategies.quant.hurst import HurstExponentStrategy
from openquant.strategies.quant.stat_arb import StatArbStrategy

strategies = {
    'RegimeAdaptive': RegimeAdaptiveStrategy(lookback=100),
    'HurstOnly': HurstExponentStrategy(lookback=100),
    'StatArbOnly': StatArbStrategy(lookback=100)
}

comparison_df = compare_strategies_by_regime(
    df=df,
    strategies=strategies,
    fee_bps=2.0,
    weight=1.0
)

print(comparison_df)

# Find best strategy per regime
for regime in ['overall', 'trending', 'mean_reverting', 'volatile']:
    regime_data = comparison_df[comparison_df['regime'] == regime]
    if not regime_data.empty:
        best = regime_data.loc[regime_data['sharpe'].idxmax()]
        print(f"{regime}: {best['strategy']} (Sharpe: {best['sharpe']:.2f})")
```

### Custom Regime Classifier

```python
def custom_classifier(df_window: pd.DataFrame) -> str:
    """Custom regime classifier based on volatility and momentum."""
    returns = df_window['Close'].pct_change().tail(50)
    vol = returns.std()
    momentum = returns.mean()
    
    if vol > 0.02:
        return 'crisis'
    elif momentum > 0.001:
        return 'bull_market'
    elif momentum < -0.001:
        return 'bear_market'
    else:
        return 'ranging'

results = walk_forward_evaluate_regime_specific(
    df=df,
    strategy_factory=strategy_factory,
    param_grid=param_grid,
    fee_bps=2.0,
    weight=1.0,
    wfo=wfo_spec,
    regime_classifier=custom_classifier
)
```

## Parameters

### RegimeAdaptiveStrategy

- `lookback` (int, default=100): Lookback period for regime detection
- `hurst_threshold_trend` (float, default=0.55): Hurst threshold for trending regime
- `hurst_threshold_mr` (float, default=0.45): Hurst threshold for mean-reverting regime
- `vol_reduce_factor` (float, default=0.5): Factor to reduce exposure during high volatility
- `enable_vol_scaling` (bool, default=True): Enable volatility-based position scaling
- `hurst_params` (dict, optional): Parameters for HurstExponentStrategy
- `stat_arb_params` (dict, optional): Parameters for StatArbStrategy
- `pair_df` (DataFrame, optional): DataFrame for StatArb pairs trading

### walk_forward_evaluate_regime_specific

- `df` (DataFrame): OHLCV data
- `strategy_factory` (callable): Function that creates strategy instances
- `param_grid` (dict): Parameter ranges to optimize
- `fee_bps` (float, default=2.0): Trading fee in basis points
- `weight` (float, default=1.0): Position sizing weight
- `wfo` (WFOSpec): WFO configuration
- `regime_classifier` (callable, optional): Custom regime classifier function

### compare_strategies_by_regime

- `df` (DataFrame): OHLCV data
- `strategies` (dict): Mapping of strategy_name -> strategy_instance
- `fee_bps` (float, default=2.0): Trading fee in basis points
- `weight` (float, default=1.0): Position sizing weight
- `regime_classifier` (callable, optional): Custom regime classifier function

## Output Formats

### Regime History DataFrame

```
                           trend_regime  volatility_regime  hurst_exponent  volatility
timestamp
2020-01-01 00:00:00        trending_up      low_volatility        0.621234    0.005123
2020-01-01 01:00:00        trending_up      low_volatility        0.618765    0.005234
...
```

### Regime Statistics Dictionary

```python
{
    'trend_regime_distribution': {
        'trending_up': 123,
        'trending_down': 45,
        'ranging': 234
    },
    'volatility_regime_distribution': {
        'high_volatility': 89,
        'low_volatility': 313
    },
    'mean_hurst_exponent': 0.523,
    'std_hurst_exponent': 0.087,
    'mean_volatility': 0.0067,
    'std_volatility': 0.0023
}
```

### WFO Results Dictionary

```python
{
    'test_sharpes': [1.23, 0.87, 1.45, 1.12],
    'mean_test_sharpe': 1.17,
    'best_params_per_split': [
        {'lookback': 100, 'vol_reduce_factor': 0.5},
        {'lookback': 150, 'vol_reduce_factor': 0.3},
        ...
    ],
    'regime_performance': {
        'trending': {
            'sharpe': 1.45,
            'mean_return': 0.0012,
            'std_return': 0.0089,
            'num_periods': 234
        },
        'mean_reverting': {...},
        'volatile': {...},
        'neutral': {...}
    },
    'regime_distribution': {
        'trending': {'count': 234, 'percentage': 35.2},
        'mean_reverting': {'count': 189, 'percentage': 28.4},
        ...
    }
}
```

## Performance Considerations

1. **Computational Cost**: Regime detection is performed at each time step, which can be computationally intensive for long datasets
2. **Lookback Period**: Shorter lookback periods provide more responsive regime detection but may be noisier
3. **Regime Switching**: Frequent regime switches can increase transaction costs
4. **Volatility Scaling**: Reducing exposure during high volatility can improve risk-adjusted returns but may miss profitable opportunities

## Best Practices

1. **Parameter Tuning**: Use WFO to find optimal parameters for your specific market and timeframe
2. **Regime Validation**: Verify that regime detection aligns with actual market conditions
3. **Transaction Costs**: Account for costs when frequently switching between sub-strategies
4. **Out-of-Sample Testing**: Always validate on unseen data before live trading
5. **Regime Analysis**: Use `get_regime_history()` to understand regime transitions

## Example Script

See `scripts/example_regime_adaptive.py` for comprehensive examples demonstrating:
- Basic usage
- Regime-specific WFO backtesting
- Strategy comparison by regime
- Custom regime classifiers

Run with:
```bash
python scripts/example_regime_adaptive.py
```

## Testing

Tests are available in:
- `tests/test_regime_adaptive.py`: Strategy tests
- `tests/test_regime_wfo.py`: WFO and comparison tests

Run tests with:
```bash
pytest tests/test_regime_adaptive.py
pytest tests/test_regime_wfo.py
```
