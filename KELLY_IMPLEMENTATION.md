# Kelly Criterion Implementation Summary

## Overview

This document summarizes the implementation of adaptive position sizing using the Kelly Criterion with volatility adjustment and drawdown-based scaling, integrated into the OpenQuant paper trading simulator.

## Files Created/Modified

### New Files

1. **openquant/risk/kelly_criterion.py**
   - Core Kelly Criterion implementation
   - `KellyCriterion` class with full adaptive sizing logic
   - `TradeRecord` and `KellyStats` dataclasses
   - Utility functions: `compute_rolling_volatility`, `estimate_win_rate_from_signals`

2. **openquant/risk/README_KELLY.md**
   - Comprehensive documentation
   - Usage examples and best practices
   - Configuration parameters
   - Mathematical explanations

3. **scripts/kelly_criterion_example.py**
   - Demonstration script showing Kelly Criterion usage
   - Examples of different scenarios
   - Trade history simulation
   - Drawdown impact demonstration

4. **tests/test_kelly_criterion.py**
   - Comprehensive test suite with 20+ tests
   - Tests for all Kelly components
   - Integration tests with paper simulator
   - Edge case coverage

5. **openquant/risk/__init__.py**
   - Module exports for easy importing

6. **openquant/paper/__init__.py**
   - Module exports for paper trading simulator

### Modified Files

1. **openquant/paper/simulator.py**
   - Added `compute_target_units_with_kelly()` function
   - Added `compute_rebalance_orders_with_kelly()` function
   - Added `rebalance_to_targets_with_kelly()` function
   - Added `record_closed_trades()` function for trade tracking
   - Enhanced `MarketSnapshot` with `price_history` field for volatility calculation
   - Imported Kelly Criterion and volatility utilities

## Key Features

### 1. Kelly Criterion Core

**Formula:** `f* = p - q/b`

Where:
- `p` = win probability (win rate)
- `q` = loss probability (1 - p)
- `b` = payoff ratio (avg_win / avg_loss)

**Implementation:**
- Tracks all trade outcomes (PnL, entry/exit prices, size)
- Computes win rate and payoff ratio from historical trades
- Applies fractional Kelly (default 0.5x) for safety
- Uses conservative sizing until minimum trades threshold reached

### 2. Volatility Adjustment

**Method:** Inverse volatility weighting

**Implementation:**
- Computes rolling annualized volatility from price history
- Scales position size inversely to volatility
- Formula: `adjusted_size = base_size * (target_vol / current_vol)`
- Capped at 2x to prevent excessive leverage

### 3. Drawdown-Based Scaling

**Method:** Quadratic reduction beyond threshold

**Implementation:**
- Tracks peak equity and current drawdown
- No reduction until drawdown exceeds threshold (default 15%)
- Quadratic scaling: `scale = 1 - (excess_dd * scale_factor)²`
- Maintains minimum 10% of base size even in severe drawdown

### 4. Integration with Paper Trading

**Seamless Integration:**
- Drop-in replacement for standard rebalancing
- Maintains all existing simulator features (fees, slippage, fills)
- Automatic trade recording for Kelly statistics
- Returns Kelly statistics alongside execution summary

## Usage Example

```python
from openquant.paper import (
    PortfolioState,
    MarketSnapshot,
    rebalance_to_targets_with_kelly,
)
from openquant.risk import KellyCriterion

# Initialize
state = PortfolioState(cash=100_000.0)
kelly_sizers = {
    key: KellyCriterion(
        kelly_fraction=0.5,
        volatility_target=0.20,
        max_drawdown_threshold=0.15,
    )
    for key in trading_keys
}

# Create snapshot with price history
snap = MarketSnapshot(
    prices={key: current_price},
    price_history={key: historical_prices},
)

# Rebalance with Kelly sizing
targets = [(key, signal) for key, signal in signals.items()]
summary, kelly_stats = rebalance_to_targets_with_kelly(
    state=state,
    targets=targets,
    snap=snap,
    kelly_sizers=kelly_sizers,
    fee_bps=10,
    volatility_window=20,
    annualization_factor=252*24,  # Hourly bars
)

# Monitor statistics
for key, stats in kelly_stats.items():
    print(f"Kelly fraction: {stats['kelly_fraction']:.2%}")
    print(f"Win rate: {stats['win_rate']:.2%}")
    print(f"Drawdown: {stats['current_drawdown']:.2%}")
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kelly_fraction` | 0.5 | Fraction of Kelly to use (0.5 = Half Kelly) |
| `min_trades` | 20 | Minimum trades before using full Kelly |
| `volatility_target` | 0.20 | Target annualized volatility (20%) |
| `max_drawdown_threshold` | 0.15 | Drawdown threshold to start reducing (15%) |
| `drawdown_scale_factor` | 2.0 | How aggressively to reduce on drawdown |
| `max_position_size` | 1.0 | Maximum position size (100% of capital) |
| `min_position_size` | 0.0 | Minimum position size threshold |

## Position Sizing Pipeline

The Kelly Criterion applies position sizing in the following order:

1. **Base Kelly Calculation**
   - If insufficient trades: Conservative ramp-up (10% → 100%)
   - Otherwise: Kelly formula with fractional adjustment

2. **Volatility Adjustment**
   - Scale by `target_vol / current_vol`
   - Capped at 2x

3. **Drawdown Scaling**
   - Only if drawdown > threshold
   - Quadratic reduction based on excess drawdown
   - Minimum 10% floor

4. **Final Caps**
   - Apply `min_position_size` and `max_position_size` limits

## Testing

Comprehensive test suite includes:

- Initialization and configuration tests
- Trade recording and statistics computation
- Kelly formula calculations
- Volatility adjustment tests
- Drawdown scaling tests
- Full pipeline integration tests
- Edge cases (zero trades, negative Kelly, etc.)
- Integration with paper trading simulator

Run tests:
```bash
pytest tests/test_kelly_criterion.py -v
```

## Example Script

Run the example demonstration:
```bash
python scripts/kelly_criterion_example.py
```

This shows:
- First trade with no history
- Building trade statistics
- Trade with updated Kelly
- Drawdown impact on position size

## Safety Features

1. **Fractional Kelly**: Default 0.5x prevents over-leveraging
2. **Conservative Ramp**: Gradual increase until sufficient trade history
3. **Position Caps**: Maximum and minimum size limits
4. **Drawdown Protection**: Automatic size reduction during losses
5. **Volatility Scaling**: Reduces size in volatile conditions
6. **Non-Negative Kelly**: Clamps negative Kelly to zero
7. **Minimum Size Floor**: Never reduces below 10% during drawdown

## Best Practices

1. **Use Half Kelly (0.5)**: Full Kelly can be too aggressive
2. **Wait for History**: Need 20+ trades for reliable estimates
3. **Monitor Drawdown**: Watch for excessive position reduction
4. **Set Conservative Caps**: Limit max position size to 100% or less
5. **Track Statistics**: Regularly review win rate, payoff ratio, expectancy
6. **Combine with Other Controls**: Use alongside stop-losses, circuit breakers
7. **Backtest First**: Validate Kelly parameters before live trading

## Limitations

- Assumes independence between trades (may not hold in reality)
- Past performance doesn't guarantee future results
- Can be aggressive if statistics are overfit
- Requires sufficient trade history to be accurate
- Doesn't account for correlation between positions
- Drawdown scaling is reactive, not predictive

## Future Enhancements

Potential improvements for future versions:

1. **Time-Weighted Statistics**: Give more weight to recent trades
2. **Regime Detection**: Adjust Kelly based on market regime
3. **Correlation Adjustment**: Account for position correlations
4. **Monte Carlo Validation**: Simulate different scenarios
5. **Dynamic Thresholds**: Adjust drawdown threshold based on volatility
6. **Per-Strategy Kelly**: Different Kelly parameters per strategy
7. **Confidence Intervals**: Provide uncertainty bounds on estimates

## References

- Kelly, J. L. (1956). "A New Interpretation of Information Rate"
- Thorp, E. O. (2008). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
- Poundstone, W. (2005). "Fortune's Formula"

## Support

For questions or issues:
- See `openquant/risk/README_KELLY.md` for detailed documentation
- Run `python scripts/kelly_criterion_example.py` for demonstrations
- Review `tests/test_kelly_criterion.py` for usage examples
