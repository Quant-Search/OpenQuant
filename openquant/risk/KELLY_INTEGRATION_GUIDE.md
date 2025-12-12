# Kelly Criterion Integration Guide

Quick guide to integrating Kelly Criterion position sizing into existing OpenQuant code.

## Quick Start (5 minutes)

### 1. Import Required Modules

```python
from openquant.risk import KellyCriterion
from openquant.paper import (
    PortfolioState,
    MarketSnapshot,
    rebalance_to_targets_with_kelly,
)
```

### 2. Initialize Kelly Sizers

Create one Kelly sizer per trading key:

```python
# Initialize once at startup
kelly_sizers = {}

for key in trading_keys:  # (exchange, symbol, timeframe, strategy)
    kelly_sizers[key] = KellyCriterion(
        kelly_fraction=0.5,           # Half Kelly (safe default)
        min_trades=20,                # Conservative until 20 trades
        volatility_target=0.20,       # 20% target volatility
        max_drawdown_threshold=0.15,  # Reduce at 15% drawdown
        max_position_size=1.0,        # Max 100% capital
    )
```

### 3. Replace Standard Rebalancing

**Before (Standard):**
```python
summary = rebalance_to_targets(
    state=state,
    targets=[(key, weight)],
    snap=snap,
    fee_bps=10,
)
```

**After (With Kelly):**
```python
summary, kelly_stats = rebalance_to_targets_with_kelly(
    state=state,
    targets=[(key, signal)],  # Can be -1, 0, 1 or continuous
    snap=snap,
    kelly_sizers=kelly_sizers,
    fee_bps=10,
    volatility_window=20,
    annualization_factor=252*24,  # Adjust based on timeframe
)
```

### 4. Update MarketSnapshot

Add price history for volatility calculation:

```python
# Before
snap = MarketSnapshot(prices={key: price})

# After
snap = MarketSnapshot(
    prices={key: price},
    price_history={key: np.array([...])},  # Last N prices
)
```

### 5. Monitor Statistics (Optional)

```python
for key, stats in kelly_stats.items():
    logger.info(
        f"{key}: Kelly={stats['kelly_fraction']:.2%}, "
        f"WinRate={stats['win_rate']:.2%}, "
        f"DD={stats['current_drawdown']:.2%}"
    )
```

## Complete Example

```python
import numpy as np
from openquant.risk import KellyCriterion
from openquant.paper import (
    PortfolioState,
    MarketSnapshot,
    rebalance_to_targets_with_kelly,
)

# Setup
state = PortfolioState(cash=100_000.0)
key = ("BINANCE", "BTC/USDT", "1h", "ema_crossover")

# Initialize Kelly
kelly_sizers = {
    key: KellyCriterion(kelly_fraction=0.5)
}

# Build price history (last 50 bars)
price_history = np.array([...])  # Your historical prices

# Create snapshot
snap = MarketSnapshot(
    prices={key: 50000.0},
    price_history={key: price_history},
)

# Get trading signal (-1, 0, or 1)
signal = 1.0  # Long signal

# Rebalance with Kelly
targets = [(key, signal)]
summary, kelly_stats = rebalance_to_targets_with_kelly(
    state=state,
    targets=targets,
    snap=snap,
    kelly_sizers=kelly_sizers,
    fee_bps=10,
    slippage_bps=5,
    volatility_window=20,
    annualization_factor=252*24,  # Hourly -> annual
)

# Check results
print(f"Orders: {summary['orders']}")
print(f"Kelly fraction: {kelly_stats[key]['kelly_fraction']:.2%}")
print(f"Position: {state.position(key):.4f} units")
```

## Annualization Factors

Choose based on your data timeframe:

| Timeframe | Factor | Calculation |
|-----------|--------|-------------|
| 1 minute  | 525,600 | 365 * 24 * 60 |
| 5 minutes | 105,120 | 365 * 24 * 12 |
| 15 minutes | 35,040 | 365 * 24 * 4 |
| 1 hour | 8,760 | 365 * 24 |
| 4 hours | 2,190 | 365 * 6 |
| Daily | 365 | 365 |
| Weekly | 52 | 52 |

Or use trading days:
- Daily: 252
- Hourly (trading hours): 252 * 6.5 = 1,638
- 24/7 crypto hourly: 252 * 24 = 6,048

## Integration Checklist

- [ ] Import Kelly modules
- [ ] Initialize Kelly sizers per trading key
- [ ] Update MarketSnapshot to include price_history
- [ ] Replace rebalance calls with Kelly version
- [ ] Adjust annualization_factor for your timeframe
- [ ] Add logging for Kelly statistics
- [ ] Test with paper trading first
- [ ] Monitor win rate and drawdown
- [ ] Adjust kelly_fraction if needed (start conservative)

## Troubleshooting

### "Position too small"
- **Cause**: Insufficient trade history or high volatility
- **Solution**: Wait for more trades or increase kelly_fraction

### "Position too large"
- **Cause**: Kelly is too aggressive
- **Solution**: Reduce kelly_fraction (try 0.25 or 0.33)

### "Volatility always None"
- **Cause**: price_history not provided in MarketSnapshot
- **Solution**: Add price_history field with numpy array of prices

### "Kelly fraction is 0"
- **Cause**: Losing strategy (negative expectancy)
- **Solution**: Check strategy performance or wait for more trades

### "Frequent reductions"
- **Cause**: High drawdown or volatility
- **Solution**: This is working as intended - protective feature

## Advanced: Multiple Strategies

If trading multiple strategies per symbol:

```python
# Separate Kelly per strategy
kelly_sizers = {}

for exchange, symbol, timeframe, strategy in trading_keys:
    key = (exchange, symbol, timeframe, strategy)
    kelly_sizers[key] = KellyCriterion(
        kelly_fraction=strategy_params[strategy]['kelly_fraction'],
        volatility_target=strategy_params[strategy]['vol_target'],
    )

# Each strategy gets independent Kelly sizing
```

## Advanced: Persistent Kelly State

Save and restore Kelly state between sessions:

```python
import pickle

# Save
with open('kelly_state.pkl', 'wb') as f:
    pickle.dump(kelly_sizers, f)

# Restore
with open('kelly_state.pkl', 'rb') as f:
    kelly_sizers = pickle.load(f)
```

Or use JSON for human-readable format:

```python
import json

# Save
state = {
    str(key): kelly.get_summary()
    for key, kelly in kelly_sizers.items()
}
with open('kelly_state.json', 'w') as f:
    json.dump(state, f, indent=2)

# Note: Full restoration requires rebuilding trade history
# Consider using database for production
```

## Advanced: Custom Configuration Per Symbol

```python
# Different params per asset class
configs = {
    'crypto': {'kelly_fraction': 0.3, 'vol_target': 0.30},
    'forex': {'kelly_fraction': 0.5, 'vol_target': 0.15},
    'stocks': {'kelly_fraction': 0.5, 'vol_target': 0.20},
}

kelly_sizers = {}
for exchange, symbol, timeframe, strategy in keys:
    asset_class = determine_asset_class(symbol)
    config = configs[asset_class]
    
    key = (exchange, symbol, timeframe, strategy)
    kelly_sizers[key] = KellyCriterion(**config)
```

## Migration Path

### Phase 1: Parallel Running (Week 1-2)
Run Kelly alongside existing system, compare results, don't trade

### Phase 2: Paper Trading (Week 3-4)
Use Kelly in paper trading environment, monitor performance

### Phase 3: Small Live Test (Week 5-6)
Deploy with small capital (1-5%), monitor closely

### Phase 4: Full Deployment (Week 7+)
Gradually increase allocation if performance is good

## Performance Monitoring

Track these metrics:

```python
def monitor_kelly_performance(kelly_stats):
    for key, stats in kelly_stats.items():
        # Red flags
        if stats['win_rate'] < 0.40:
            logger.warning(f"{key}: Low win rate {stats['win_rate']:.1%}")
        
        if stats['expectancy'] < 0:
            logger.warning(f"{key}: Negative expectancy {stats['expectancy']:.2f}")
        
        if stats['current_drawdown'] > 0.20:
            logger.warning(f"{key}: High drawdown {stats['current_drawdown']:.1%}")
        
        # Green flags
        if stats['total_trades'] > 50 and stats['expectancy'] > 0:
            logger.info(f"{key}: Mature profitable strategy")
```

## Support

- Full documentation: `openquant/risk/README_KELLY.md`
- Example script: `scripts/kelly_criterion_example.py`
- Test suite: `tests/test_kelly_criterion.py`
- Implementation details: `KELLY_IMPLEMENTATION.md`
