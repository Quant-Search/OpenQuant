# Multi-Timeframe Strategy - Quick Start Guide

## Installation
No additional installation required. The MTF strategy is now part of OpenQuant.

## Basic Usage

### 1. Simple MTF Strategy (Recommended Starting Point)

```python
from openquant.strategies.mtf_strategy import MultiTimeframeStrategy
from openquant.strategies.quant.stat_arb import StatArbStrategy
from openquant.data.loader import DataLoader

# Create your base strategy
base_strategy = StatArbStrategy(entry_z=2.0, exit_z=0.5)

# Define fetch function
loader = DataLoader()
def fetch_data(symbol, timeframe):
    return loader.get_ohlcv(
        source='ccxt:binance',
        symbol=symbol,
        timeframe=timeframe,
    )

# Wrap with MTF confirmation
mtf_strategy = MultiTimeframeStrategy(
    base_strategy=base_strategy,
    timeframes=['1h', '4h', '1d'],  # Check these timeframes
    fetch_func=fetch_data,
    min_confirmations=1,  # Need 1 higher TF to confirm
)

# Set symbol and generate signals
mtf_strategy.set_symbol('BTC/USDT')
df = fetch_data('BTC/USDT', '1h')
signals = mtf_strategy.generate_signals(df)
```

### 2. Ensemble Strategy (Multiple Strategies)

```python
from openquant.strategies.mtf_strategy import MultiTimeframeEnsemble

# Create strategies for different timeframes
strategy_1h = StatArbStrategy(entry_z=2.0, exit_z=0.5)
strategy_4h = StatArbStrategy(entry_z=1.8, exit_z=0.4)
strategy_1d = StatArbStrategy(entry_z=1.5, exit_z=0.3)

# Create ensemble
ensemble = MultiTimeframeEnsemble(
    strategies=[
        ('1h', strategy_1h, 0.5),  # (timeframe, strategy, weight)
        ('4h', strategy_4h, 0.3),
        ('1d', strategy_1d, 0.2),
    ],
    fetch_func=fetch_data,
    aggregation='weighted',  # 'weighted', 'majority', or 'unanimous'
    threshold=0.3,
)

ensemble.set_symbol('BTC/USDT')
signals = ensemble.generate_signals(df)
```

### 3. With Regime Filtering

```python
from openquant.validation.mtf_filter import check_regime_filter, get_regime_score

# Generate signals
signals = mtf_strategy.generate_signals(df)

# Filter by regime
regime_mask = check_regime_filter(
    df, 
    regime_type='trend',  # 'trend', 'range', 'volatile', or 'any'
    min_regime_strength=0.6
)
filtered_signals = signals * regime_mask

# Or use regime scores for position sizing
for idx in signals[signals != 0].index:
    score = get_regime_score(df.loc[:idx], int(signals.loc[idx]))
    position_size = base_size * score  # Scale by regime favorability
```

### 4. Backtesting

```python
from openquant.backtest.engine import backtest_signals

# Generate signals
signals = mtf_strategy.generate_signals(df)

# Backtest
result = backtest_signals(
    df=df,
    signals=signals,
    fee_bps=2.0,  # 0.02% fees
    weight=1.0,   # 100% allocation
)

# Results
print(f"Total Return: {(result.equity_curve.iloc[-1] - 1) * 100:.2f}%")
print(f"Sharpe Ratio: {result.returns.mean() / result.returns.std() * np.sqrt(252):.2f}")
```

## Configuration Options

### MultiTimeframeStrategy Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_strategy` | Required | The strategy to wrap |
| `timeframes` | `['1h','4h','1d']` | Timeframes to check |
| `fetch_func` | Required | Function to fetch OHLCV data |
| `require_all_timeframes` | `False` | If True, all TFs must confirm |
| `min_confirmations` | `1` | Minimum TF confirmations needed |
| `use_strategy_signals` | `False` | Use strategy on higher TFs vs trend check |

### MultiTimeframeEnsemble Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategies` | Required | List of (timeframe, strategy, weight) |
| `fetch_func` | Required | Function to fetch OHLCV data |
| `aggregation` | `'weighted'` | How to combine signals |
| `threshold` | `0.3` | Threshold for weighted aggregation |

### Aggregation Methods

- **weighted**: Weighted average of signals (use threshold)
- **majority**: Simple majority vote
- **unanimous**: All strategies must agree

## Common Patterns

### Conservative (Fewer Signals, Higher Quality)
```python
mtf_strategy = MultiTimeframeStrategy(
    base_strategy=strategy,
    timeframes=['1h', '4h', '1d'],
    require_all_timeframes=True,  # All must confirm
    min_confirmations=3,
)
```

### Aggressive (More Signals)
```python
mtf_strategy = MultiTimeframeStrategy(
    base_strategy=strategy,
    timeframes=['1h', '4h'],
    require_all_timeframes=False,
    min_confirmations=1,  # Only one higher TF needed
)
```

### Balanced
```python
mtf_strategy = MultiTimeframeStrategy(
    base_strategy=strategy,
    timeframes=['1h', '4h', '1d'],
    require_all_timeframes=False,
    min_confirmations=2,  # Need 2 out of 3
)
```

## Timeframe Combinations

### Scalping
```python
timeframes = ['5m', '15m', '1h']  # Fast signals
```

### Day Trading
```python
timeframes = ['15m', '1h', '4h']  # Intraday
```

### Swing Trading
```python
timeframes = ['1h', '4h', '1d']  # Multi-day holds
```

### Position Trading
```python
timeframes = ['4h', '1d', '1w']  # Long-term
```

## Testing Your Strategy

```bash
# Run tests
pytest tests/test_mtf_strategy.py -v
pytest tests/test_mtf_validation.py -v

# Run examples
python scripts/mtf_strategy_example.py
python scripts/backtest_mtf_strategy.py
```

## Performance Tips

1. **Cache Data**: Store fetched data to avoid redundant API calls
2. **Start Simple**: Begin with trend-based confirmation (faster)
3. **Monitor Signal Frequency**: MTF typically reduces signals by 50-70%
4. **Use Regime Scores**: For dynamic position sizing
5. **Backtest Thoroughly**: Test across different market conditions

## Troubleshooting

### Too Few Signals?
- Reduce `min_confirmations`
- Set `require_all_timeframes=False`
- Use shorter timeframe combinations
- Lower regime strength threshold

### Too Many Signals?
- Increase `min_confirmations`
- Set `require_all_timeframes=True`
- Use longer timeframe combinations
- Add regime filtering

### Strategy Running Slow?
- Use trend-based confirmation (not strategy-based)
- Implement data caching
- Reduce number of timeframes
- Use wider timeframe spacing

## Next Steps

1. Read `MTF_STRATEGY_README.md` for detailed documentation
2. Check `MTF_IMPLEMENTATION_SUMMARY.md` for technical details
3. Explore example scripts in `scripts/`
4. Review tests in `tests/` for usage patterns

## Support

For questions or issues:
- Check the comprehensive README: `openquant/strategies/MTF_STRATEGY_README.md`
- Review test cases: `tests/test_mtf_strategy.py`
- Run examples: `scripts/mtf_strategy_example.py`
