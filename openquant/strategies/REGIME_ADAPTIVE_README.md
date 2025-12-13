# Regime Adaptive Strategy

## Overview

The `RegimeAdaptiveStrategy` is an intelligent meta-strategy that automatically detects market regimes and routes trading decisions to the most appropriate sub-strategy. It leverages statistical regime detection to classify markets as trending, mean-reverting, or volatile, and adjusts position sizing accordingly.

## Key Features

- **Automatic Regime Detection**: Uses the Hurst exponent and volatility analysis to classify market conditions
- **Dynamic Strategy Routing**: 
  - **Trending markets** (H > 0.55): Routes to `HurstExponentStrategy`
  - **Mean-reverting markets** (H < 0.45): Routes to `StatArbStrategy`
  - **Neutral markets**: Stays flat
- **Regime-Aware Position Sizing**: Automatically scales exposure based on volatility regimes
- **Full WFO Integration**: Built-in support for regime-specific walk-forward optimization
- **Comprehensive Analytics**: Track regime history, transitions, and performance by regime

## Architecture

```
RegimeAdaptiveStrategy
├── RegimeDetector (from openquant.quant.regime_detector)
│   ├── Hurst Exponent Calculation (R/S Analysis)
│   ├── Volatility Analysis
│   └── Regime Classification
├── Sub-Strategies
│   ├── HurstExponentStrategy (trending regimes)
│   └── StatArbStrategy (mean-reverting regimes)
└── Position Scaling
    ├── High Volatility: Reduced exposure (vol_reduce_factor)
    └── Normal Volatility: Full exposure
```

## Usage

### Basic Example

```python
from openquant.strategies.regime_adaptive import RegimeAdaptiveStrategy

# Create strategy with default parameters
strategy = RegimeAdaptiveStrategy(lookback=100)

# Generate signals
signals = strategy.generate_signals(df)

# Analyze regime history
regime_history = strategy.get_regime_history()
regime_stats = strategy.get_regime_stats()
```

### Advanced Configuration

```python
# Custom parameters for sub-strategies
strategy = RegimeAdaptiveStrategy(
    lookback=100,
    hurst_threshold_trend=0.60,    # Higher threshold for trend detection
    hurst_threshold_mr=0.40,       # Lower threshold for mean reversion
    vol_reduce_factor=0.3,         # More aggressive vol scaling
    enable_vol_scaling=True,
    hurst_params={
        'lookback': 80,
        'trend_threshold': 0.6,
        'mr_threshold': 0.4
    },
    stat_arb_params={
        'lookback': 80,
        'entry_z': 2.5,
        'exit_z': 0.5
    }
)
```

### Pairs Trading Mode

```python
# Use StatArb in pairs trading mode for mean-reverting regimes
strategy = RegimeAdaptiveStrategy(
    lookback=100,
    stat_arb_params={'pair_symbol': 'SPY'},
    pair_df=spy_dataframe  # Provide paired asset data
)
```

## Walk-Forward Optimization

### Standard WFO

```python
from openquant.evaluation.wfo import walk_forward_evaluate, WFOSpec

def strategy_factory(lookback=100, vol_reduce_factor=0.5):
    return RegimeAdaptiveStrategy(
        lookback=lookback,
        vol_reduce_factor=vol_reduce_factor
    )

param_grid = {
    'lookback': [80, 100, 120],
    'vol_reduce_factor': [0.3, 0.5, 0.7]
}

wfo_spec = WFOSpec(n_splits=4, train_frac=0.7)

results = walk_forward_evaluate(
    df=df,
    strategy_factory=strategy_factory,
    param_grid=param_grid,
    wfo=wfo_spec
)
```

### Regime-Specific WFO

```python
from openquant.evaluation.wfo import walk_forward_evaluate_regime_specific

# Evaluates performance separately by regime
results = walk_forward_evaluate_regime_specific(
    df=df,
    strategy_factory=strategy_factory,
    param_grid=param_grid,
    wfo=wfo_spec
)

# Access regime-specific metrics
print(results['regime_performance'])
print(results['regime_distribution'])
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lookback` | int | 100 | Lookback period for regime detection |
| `hurst_threshold_trend` | float | 0.55 | Hurst threshold for trending regime |
| `hurst_threshold_mr` | float | 0.45 | Hurst threshold for mean-reverting regime |
| `vol_reduce_factor` | float | 0.5 | Position size multiplier during high volatility (0-1) |
| `enable_vol_scaling` | bool | True | Enable volatility-based position scaling |
| `hurst_params` | dict | None | Custom parameters for HurstExponentStrategy |
| `stat_arb_params` | dict | None | Custom parameters for StatArbStrategy |
| `pair_df` | DataFrame | None | Paired asset data for StatArb pairs trading |

## Regime Classification

### Trend Regimes
- **TRENDING_UP**: Hurst > threshold and positive returns
- **TRENDING_DOWN**: Hurst > threshold and negative returns
- **RANGING**: Hurst < threshold or neutral

### Volatility Regimes
- **HIGH_VOLATILITY**: Current vol > 1.5x historical vol
- **LOW_VOLATILITY**: Normal volatility conditions

## Position Sizing Logic

```python
if regime == HIGH_VOLATILITY:
    if enable_vol_scaling:
        position = base_signal * vol_reduce_factor
    else:
        position = 0  # Flat during high volatility
elif regime == TRENDING:
    position = hurst_strategy_signal  # Full exposure
elif regime == MEAN_REVERTING:
    position = stat_arb_signal  # Full exposure
else:
    position = 0  # Flat during neutral regime
```

## Analysis Methods

### Get Regime History

```python
history = strategy.get_regime_history()
# Returns DataFrame with columns:
# - trend_regime
# - volatility_regime
# - hurst_exponent
# - volatility
```

### Get Regime Statistics

```python
stats = strategy.get_regime_stats()
# Returns dict with:
# - trend_regime_distribution
# - volatility_regime_distribution
# - mean_hurst_exponent
# - std_hurst_exponent
# - mean_volatility
# - std_volatility
```

### Reset History

```python
strategy.reset_history()  # Clear regime history for new backtest
```

## Strategy Comparison

```python
from openquant.evaluation.wfo import compare_strategies_by_regime

strategies = {
    'RegimeAdaptive': RegimeAdaptiveStrategy(lookback=100),
    'HurstOnly': HurstExponentStrategy(lookback=100),
    'StatArbOnly': StatArbStrategy(lookback=100)
}

comparison = compare_strategies_by_regime(
    df=df,
    strategies=strategies,
    fee_bps=2.0
)

# Shows performance of each strategy in different regimes
print(comparison)
```

## Integration with Strategy Registry

```python
from openquant.strategies.registry import make_strategy

# Create via registry
strategy = make_strategy('regime_adaptive', lookback=100, vol_reduce_factor=0.5)
```

## Performance Considerations

1. **Lookback Period**: Shorter lookback (50-80) is more responsive but noisier; longer lookback (100-150) is more stable but slower to adapt
2. **Threshold Tuning**: Wider thresholds (0.35-0.65) favor the neutral regime; tighter thresholds (0.45-0.55) switch strategies more frequently
3. **Volatility Scaling**: Lower `vol_reduce_factor` (0.3) is more conservative during volatility; higher (0.7) maintains more exposure
4. **Sub-Strategy Parameters**: Tune `hurst_params` and `stat_arb_params` for specific market characteristics

## Examples

See `scripts/example_regime_adaptive.py` for comprehensive examples including:
- Basic usage
- Advanced configuration
- Regime analysis
- Strategy comparison
- WFO integration
- Regime-specific WFO

## Testing

Run the test suite:
```bash
pytest tests/test_regime_adaptive.py -v
pytest tests/test_regime_wfo.py -v
```

## References

- **Hurst Exponent**: Measures long-term memory and trend persistence
- **Regime Detection**: `openquant/quant/regime_detector.py`
- **WFO Framework**: `openquant/evaluation/wfo.py`
- **Sub-Strategies**:
  - `openquant/strategies/quant/hurst.py`
  - `openquant/strategies/quant/stat_arb.py`
