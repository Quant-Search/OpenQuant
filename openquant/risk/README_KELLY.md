# Kelly Criterion Adaptive Position Sizing

This module implements the Kelly Criterion for optimal position sizing with adaptive adjustments based on volatility, drawdown, and trade history.

## Overview

The Kelly Criterion is a mathematical formula for optimal position sizing that maximizes long-term growth while accounting for risk. The formula is:

```
f* = p - q/b
```

Where:
- `p` = win probability (win rate)
- `q` = loss probability (1 - p)
- `b` = payoff ratio (avg_win / avg_loss)
- `f*` = optimal fraction of capital to risk

## Features

### Core Kelly Criterion
- **Win Rate Estimation**: Tracks winning and losing trades to estimate probability of success
- **Payoff Ratio Calculation**: Computes average win / average loss from historical trades
- **Fractional Kelly**: Uses a fraction (default 0.5) of full Kelly for safety
- **Minimum Trade Requirement**: Conservative sizing until sufficient trade history

### Volatility Adjustment
- **Inverse Volatility Weighting**: Reduces position size when volatility increases
- **Rolling Volatility Calculation**: Uses recent price history to estimate current volatility
- **Volatility Target**: Scales positions to achieve target annualized volatility
- **Annualization Factor**: Supports different timeframes (daily, hourly, etc.)

### Drawdown-Based Scaling
- **Drawdown Tracking**: Monitors peak-to-trough equity decline
- **Quadratic Reduction**: Aggressively reduces size as drawdown approaches threshold
- **Threshold-Based**: Only activates when drawdown exceeds configured threshold
- **Minimum Size Floor**: Maintains at least 10% of base size even in severe drawdown

## Usage

### Basic Setup

```python
from openquant.risk.kelly_criterion import KellyCriterion

# Initialize Kelly sizer
kelly = KellyCriterion(
    kelly_fraction=0.5,           # Half Kelly for safety
    min_trades=20,                # Conservative until 20 trades
    volatility_target=0.20,       # Target 20% annualized volatility
    max_drawdown_threshold=0.15,  # Start reducing at 15% drawdown
    max_position_size=1.0,        # Max 100% of capital
)
```

### Recording Trades

```python
# After closing a trade, record the outcome
kelly.record_trade(
    pnl=150.50,              # Profit/loss in currency units
    entry_price=50000.0,     # Entry price
    exit_price=50500.0,      # Exit price
    size=0.1,                # Position size in units
    duration_bars=24,        # Optional: how many bars trade lasted
)

# Update equity for drawdown tracking
kelly.update_equity(current_equity=105_000.0)
```

### Computing Position Size

```python
# Compute position size with all adjustments
position_size = kelly.compute_position_size(
    volatility=0.25,  # Current annualized volatility (optional)
)

# position_size is a fraction (e.g., 0.35 = 35% of capital)
```

### Integration with Paper Trading

```python
from openquant.paper.simulator import rebalance_to_targets_with_kelly
from openquant.paper.state import PortfolioState
from openquant.paper.simulator import MarketSnapshot

# Initialize components
state = PortfolioState(cash=100_000.0)
kelly_sizers = {
    key: KellyCriterion() for key in trading_keys
}

# Create market snapshot with price history
snap = MarketSnapshot(
    prices={key: current_price},
    price_history={key: price_array},  # For volatility calculation
)

# Rebalance with Kelly sizing
targets = [(key, signal_weight) for key, signal_weight in signals.items()]

summary, kelly_stats = rebalance_to_targets_with_kelly(
    state=state,
    targets=targets,
    snap=snap,
    kelly_sizers=kelly_sizers,
    fee_bps=10,                    # 0.1% fees
    volatility_window=20,          # 20 bars for volatility
    annualization_factor=252*24,   # Hourly data -> annual
)

# Check statistics
for key, stats in kelly_stats.items():
    print(f"{key}:")
    print(f"  Kelly fraction: {stats['kelly_fraction']:.2%}")
    print(f"  Win rate: {stats['win_rate']:.2%}")
    print(f"  Payoff ratio: {stats['payoff_ratio']:.2f}")
    print(f"  Current drawdown: {stats['current_drawdown']:.2%}")
```

## Configuration Parameters

### KellyCriterion Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kelly_fraction` | float | 0.5 | Fraction of Kelly to use (0.5 = Half Kelly) |
| `min_trades` | int | 20 | Minimum trades before using full Kelly |
| `volatility_target` | float | 0.20 | Target annualized volatility (20%) |
| `max_drawdown_threshold` | float | 0.15 | Drawdown threshold to start reducing (15%) |
| `drawdown_scale_factor` | float | 2.0 | How aggressively to reduce on drawdown |
| `max_position_size` | float | 1.0 | Maximum position size (100% of capital) |
| `min_position_size` | float | 0.0 | Minimum position size threshold |

### Recommended Settings

**Conservative (Default)**
```python
kelly = KellyCriterion(
    kelly_fraction=0.5,           # Half Kelly
    max_position_size=1.0,        # Max 100%
    max_drawdown_threshold=0.15,  # Reduce at 15% DD
)
```

**Moderate**
```python
kelly = KellyCriterion(
    kelly_fraction=0.75,          # 3/4 Kelly
    max_position_size=1.5,        # Max 150%
    max_drawdown_threshold=0.20,  # Reduce at 20% DD
)
```

**Aggressive (Not Recommended)**
```python
kelly = KellyCriterion(
    kelly_fraction=1.0,           # Full Kelly
    max_position_size=2.0,        # Max 200%
    max_drawdown_threshold=0.25,  # Reduce at 25% DD
)
```

## How It Works

### Step-by-Step Position Sizing

1. **Base Kelly Size**: Computed from trade statistics
   - If `trades < min_trades`: Use conservative ramp-up (10% → 100%)
   - Otherwise: Use Kelly formula with fractional adjustment

2. **Volatility Adjustment**: Scale by inverse volatility
   - `adjustment = target_vol / current_vol`
   - Capped at 2x to prevent excessive leverage

3. **Drawdown Scaling**: Reduce if drawdown exceeds threshold
   - Excess drawdown = `current_dd - threshold`
   - Scale factor = `1 - (excess_dd * drawdown_scale_factor)²`
   - Minimum scale of 0.1 (never go below 10%)

4. **Final Caps**: Apply min/max position size limits

### Example Calculation

Assume:
- 30 trades: 18 wins (60%), 12 losses (40%)
- Avg win: $200, Avg loss: $100
- Current volatility: 30%, Target: 20%
- Current drawdown: 18%, Threshold: 15%
- Kelly fraction: 0.5

**Step 1: Base Kelly**
```
Payoff ratio (b) = 200 / 100 = 2.0
Kelly = p - q/b = 0.6 - 0.4/2.0 = 0.6 - 0.2 = 0.4
Fractional Kelly = 0.4 × 0.5 = 0.20 (20%)
```

**Step 2: Volatility Adjustment**
```
Vol adjustment = 0.20 / 0.30 = 0.667
Adjusted size = 0.20 × 0.667 = 0.133 (13.3%)
```

**Step 3: Drawdown Scaling**
```
Excess DD = 0.18 - 0.15 = 0.03
Reduction = (0.03 × 2.0)² = 0.0036
Scale factor = 1 - 0.0036 = 0.9964
Final size = 0.133 × 0.9964 = 0.133 (13.3%)
```

**Result**: Position size = 13.3% of capital

## Statistics and Monitoring

### Get Current Statistics

```python
stats = kelly.get_summary()

print(f"Kelly fraction: {stats['kelly_fraction']:.2%}")
print(f"Win rate: {stats['win_rate']:.2%}")
print(f"Payoff ratio: {stats['payoff_ratio']:.2f}")
print(f"Expectancy: ${stats['expectancy']:.2f}")
print(f"Total trades: {stats['total_trades']}")
print(f"Current drawdown: {stats['current_drawdown']:.2%}")
```

### Expectancy

The expectancy tells you the expected profit per trade:

```
Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
```

Positive expectancy means the strategy is profitable on average.

## Best Practices

1. **Use Fractional Kelly**: Half Kelly (0.5) is recommended for most strategies
2. **Sufficient History**: Wait for 20+ trades before trusting Kelly estimates
3. **Monitor Drawdown**: Reduce exposure during drawdown periods
4. **Volatility Targeting**: Adjust for changing market conditions
5. **Set Conservative Caps**: Limit maximum position size (e.g., 100%)
6. **Track Statistics**: Regularly review win rate, payoff ratio, and expectancy
7. **Combine with Other Risk Controls**: Use alongside stop-losses, circuit breakers, etc.

## Limitations

- Assumes independence between trades (may not hold in reality)
- Past performance doesn't guarantee future results
- Can be aggressive if trade statistics are overfit
- Requires sufficient trade history to be accurate
- Doesn't account for correlation between positions

## References

- Kelly, J. L. (1956). "A New Interpretation of Information Rate"
- Thorp, E. O. (2008). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
- Poundstone, W. (2005). "Fortune's Formula: The Untold Story of the Scientific Betting System That Beat the Casinos and Wall Street"
