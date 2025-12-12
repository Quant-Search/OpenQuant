# Multi-Timeframe Strategy Documentation

## Overview

The Multi-Timeframe (MTF) strategy implementation provides sophisticated signal confirmation across multiple timeframes, ensuring that trades align with higher timeframe trends and market regimes before entry.

## Key Features

1. **Signal Alignment**: Requires confirmation from higher timeframes before taking positions
2. **Regime Filtering**: Validates market conditions (trending, ranging, volatile) across timeframes
3. **Ensemble Support**: Combines multiple strategies across different timeframes with weighted voting
4. **Flexible Configuration**: Supports various confirmation requirements and aggregation methods

## Components

### 1. MultiTimeframeStrategy

Wraps a base strategy and filters its signals based on higher timeframe confirmation.

**Key Parameters:**
- `base_strategy`: The underlying strategy to generate signals
- `timeframes`: List of timeframes in order (e.g., `['1h', '4h', '1d']`)
- `fetch_func`: Function to fetch OHLCV data for different timeframes
- `require_all_timeframes`: If True, all higher timeframes must confirm
- `min_confirmations`: Minimum number of confirmations required
- `use_strategy_signals`: Use strategy signals for confirmation vs. simple trend checks

**Usage Example:**
```python
from openquant.strategies.mtf_strategy import MultiTimeframeStrategy
from openquant.strategies.quant.stat_arb import StatArbStrategy

# Create base strategy
base_strategy = StatArbStrategy(entry_z=2.0, exit_z=0.5)

# Wrap with MTF confirmation
mtf_strategy = MultiTimeframeStrategy(
    base_strategy=base_strategy,
    timeframes=['1h', '4h', '1d'],
    fetch_func=your_fetch_function,
    require_all_timeframes=False,
    min_confirmations=1,
)

mtf_strategy.set_symbol('BTC/USDT')
signals = mtf_strategy.generate_signals(df_1h)
```

### 2. MultiTimeframeEnsemble

Runs different strategies on multiple timeframes and combines their signals.

**Key Parameters:**
- `strategies`: List of `(timeframe, strategy, weight)` tuples
- `fetch_func`: Function to fetch OHLCV data
- `aggregation`: How to combine signals:
  - `'weighted'`: Weighted average with threshold
  - `'majority'`: Majority voting
  - `'unanimous'`: All must agree
- `threshold`: For weighted aggregation, signal threshold

**Usage Example:**
```python
from openquant.strategies.mtf_strategy import MultiTimeframeEnsemble

ensemble = MultiTimeframeEnsemble(
    strategies=[
        ('1h', strategy_1h, 0.5),
        ('4h', strategy_4h, 0.3),
        ('1d', strategy_1d, 0.2),
    ],
    fetch_func=your_fetch_function,
    aggregation='weighted',
    threshold=0.3,
)

ensemble.set_symbol('BTC/USDT')
signals = ensemble.generate_signals(df_1h)
```

## Validation and Regime Filtering

### mtf_filter.py Functions

#### check_mtf_confirmation
Validates if higher timeframes confirm a signal direction using SMA-based trend checks.

```python
from openquant.validation.mtf_filter import check_mtf_confirmation

confirmed = check_mtf_confirmation(
    symbol='BTC/USDT',
    timeframe='1h',
    signal_direction=1,  # 1=long, -1=short, 0=flat
    fetch_func=your_fetch_function,
)
```

#### check_regime_filter
Filters signals based on market regime detection.

```python
from openquant.validation.mtf_filter import check_regime_filter

regime_mask = check_regime_filter(
    df=ohlcv_df,
    regime_type='trend',  # 'trend', 'range', 'volatile', or 'any'
    min_regime_strength=0.5,
)
```

#### get_regime_score
Calculates a score (0-1) indicating how favorable current conditions are for a signal.

```python
from openquant.validation.mtf_filter import get_regime_score

score = get_regime_score(
    df=ohlcv_df,
    signal_direction=1,  # 1=long, -1=short
)

# Use score to size positions or filter signals
if score > 0.7:
    # Strong regime, take position
    pass
```

## Confirmation Logic

### Trend-Based Confirmation

The strategy checks multiple indicators on higher timeframes:

1. **Price vs SMA50**: Price should be on the correct side of moving average
2. **SMA20 vs SMA50**: Short MA should confirm trend direction
3. **Recent Momentum**: Price momentum should align with signal direction

For a **long signal** to be confirmed on a higher timeframe:
- Price > SMA50 (price above trend)
- SMA20 > SMA50 (uptrend)
- Recent momentum > -1% (not falling too fast)

At least 2 of these 3 conditions must be met.

### Strategy-Based Confirmation

When `use_strategy_signals=True`, the base strategy runs on each higher timeframe:
- For long signals: Higher TF must show recent long signals
- For short signals: Higher TF must show recent short signals

## Regime Detection

### Trend Strength
Measures how strongly the market is trending using:
- Distance from SMA (normalized by volatility)
- Price momentum
- Returns value 0-1 (higher = stronger trend)

### Volatility Regime
Identifies high/low volatility environments using:
- Rolling standard deviation
- ATR (Average True Range)
- Returns percentile rank (0-1)

### Range Regime
Detects range-bound markets using:
- Position within recent price range
- Frequency of touching range boundaries
- Low directional movement

## Integration Examples

### Example 1: Conservative MTF Strategy
Requires all timeframes to confirm before entry:

```python
mtf_strategy = MultiTimeframeStrategy(
    base_strategy=your_strategy,
    timeframes=['1h', '4h', '1d'],
    fetch_func=fetch_data,
    require_all_timeframes=True,
    min_confirmations=3,
)
```

### Example 2: Aggressive MTF Strategy
Requires only one higher timeframe to confirm:

```python
mtf_strategy = MultiTimeframeStrategy(
    base_strategy=your_strategy,
    timeframes=['1h', '4h', '1d'],
    fetch_func=fetch_data,
    require_all_timeframes=False,
    min_confirmations=1,
)
```

### Example 3: Weighted Ensemble
Different strategies on different timeframes with weights:

```python
ensemble = MultiTimeframeEnsemble(
    strategies=[
        ('1h', scalping_strategy, 0.4),   # Fast signals
        ('4h', swing_strategy, 0.35),     # Medium-term
        ('1d', position_strategy, 0.25),  # Long-term bias
    ],
    aggregation='weighted',
    threshold=0.4,  # Need 40% weighted consensus
)
```

### Example 4: Regime-Aware Filtering
Use regime filtering with MTF strategy:

```python
from openquant.validation.mtf_filter import check_regime_filter, get_regime_score

# Generate signals
signals = mtf_strategy.generate_signals(df)

# Apply regime filter
regime_mask = check_regime_filter(df, regime_type='trend', min_regime_strength=0.6)
filtered_signals = signals * regime_mask

# Or use regime scoring for position sizing
for idx in signals.index:
    if signals.loc[idx] != 0:
        score = get_regime_score(df.loc[:idx], signals.loc[idx])
        position_size = base_size * score
```

## Timeframe Hierarchy

The system recognizes the following timeframe hierarchy:
```
1m -> 5m -> 15m -> 1h -> 4h -> 1d -> 1w
```

When checking MTF confirmation, the strategy looks at the next 1-2 higher timeframes.

## Performance Considerations

1. **Data Fetching**: Cache higher timeframe data to avoid excessive API calls
2. **Computation**: Trend checks are lightweight; strategy-based checks are more expensive
3. **Signal Frequency**: MTF filtering typically reduces signal frequency by 40-80%
4. **Latency**: Ensemble strategies need data from multiple timeframes

## Best Practices

1. **Start Conservative**: Begin with `require_all_timeframes=True` to reduce false signals
2. **Test Timeframe Combinations**: Different markets favor different timeframe combinations
3. **Monitor Regime Changes**: Use regime scores to adjust position sizing
4. **Backtest Thoroughly**: MTF strategies behave differently in different market conditions
5. **Handle Missing Data**: Ensure fetch_func handles missing/incomplete data gracefully

## Signal Convention

All strategies follow the standard signal convention:
- `1`: Long position
- `0`: Flat (no position)
- `-1`: Short position

## Error Handling

The implementation handles:
- Missing higher timeframe data (defaults to accepting signal)
- Empty DataFrames (returns flat signals)
- Failed strategy execution (continues with other timeframes)
- Unknown timeframes (bypasses MTF check)

## Testing

Run the comprehensive test suite:
```bash
pytest tests/test_mtf_strategy.py
pytest tests/test_mtf_validation.py
```

## See Also

- `openquant/strategies/base.py`: Base strategy interface
- `openquant/strategies/mixer.py`: Strategy ensemble without MTF
- `openquant/validation/strategy_validator.py`: Strategy validation utilities
- `scripts/mtf_strategy_example.py`: Complete usage examples
