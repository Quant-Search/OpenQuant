# Transaction Cost Model Enhancements

This document describes the enhanced transaction cost models implemented in the OpenQuant backtest engine.

## Overview

The backtest engine now includes sophisticated transaction cost modeling to provide more realistic performance estimates:

1. **Time-of-Day Spread Modeling** - Spreads vary by hour based on liquidity patterns
2. **Volume-Dependent Slippage** - Slippage increases when trading during low volume
3. **Market Impact for Large Orders** - Square-root impact model based on academic research
4. **Funding Rates for Perpetual Swaps** - Both static and dynamic funding rate calculations

## Features

### 1. Time-of-Day Spread Modeling

Spreads are typically tighter during high-liquidity hours (e.g., London/NY overlap for FX) and wider during low-liquidity hours (e.g., Asian session).

**Usage:**
```python
from openquant.backtest import backtest_signals, TOD_MULTIPLIERS_CRYPTO_MAJOR

result = backtest_signals(
    df=df,
    signals=signals,
    spread_bps=2.0,  # Base spread
    use_tod_spread=True,
    tod_multipliers=TOD_MULTIPLIERS_CRYPTO_MAJOR,  # or custom dict
)
```

**Built-in Multiplier Presets:**
- `TOD_MULTIPLIERS_FX_MAJOR` - FX major pairs (EUR/USD, GBP/USD, etc.)
- `TOD_MULTIPLIERS_CRYPTO_MAJOR` - BTC/USD, ETH/USD patterns
- `TOD_MULTIPLIERS_CRYPTO_ALTCOIN` - Altcoins with wider spreads
- `TOD_MULTIPLIERS_FLAT` - No time variation (constant spread)

**Custom Multipliers:**
```python
custom_multipliers = {
    0: 1.5,   # Midnight - 50% wider spread
    9: 0.8,   # 9 AM - 20% tighter spread
    # ... for each hour 0-23
}
```

### 2. Volume-Dependent Slippage

Slippage increases when trade size is large relative to market volume. This model calculates slippage based on the ratio of current volume to average volume.

**Usage:**
```python
result = backtest_signals(
    df=df,  # Must include 'Volume' column
    signals=signals,
    slippage_bps=0.5,  # Base slippage
    use_volume_slippage=True,
    volume_impact_coeff=0.15,  # Controls sensitivity to volume
)
```

**Parameters:**
- `slippage_bps`: Minimum slippage in basis points
- `volume_impact_coeff`: Higher values increase slippage during low volume (typical: 0.1-0.3)

**Formula:**
```
volume_ratio = avg_volume / current_volume
slippage = base_slippage * (1 + coeff * (volume_ratio - 1))
```

### 3. Market Impact Model

Implements the square-root market impact model based on research by Almgren, Chriss, and others. Market impact scales with the square root (or custom exponent) of trade size relative to volume.

**Usage:**
```python
result = backtest_signals(
    df=df,  # Must include 'Volume' column
    signals=signals,
    use_market_impact=True,
    participation_rate=0.05,  # Expected 5% of volume participation
    impact_exponent=0.6,      # Power law exponent (0.5-0.7 typical)
)
```

**Parameters:**
- `participation_rate`: Expected fraction of market volume (0.01-0.1 typical)
- `impact_exponent`: Power law exponent, typically 0.5-0.7
  - 0.5 = classic square-root model
  - 0.6 = slightly more aggressive
  - 0.7 = higher impact for large trades

**Formula:**
```
trade_size_ratio = position_change / avg_volume
impact_bps = 100 * (trade_size_ratio / participation_rate) ^ exponent * volatility_scalar
```

### 4. Funding Rates for Perpetual Swaps

Two models for funding rates:

#### A. Static Funding Rate
Fixed funding rate charged/paid at regular intervals (typically every 8 hours for crypto).

**Usage:**
```python
result = backtest_signals(
    df=df,
    signals=signals,
    funding_rate_bps=1.0,  # 0.01% per funding interval
    funding_interval_hours=8,
)
```

#### B. Dynamic Funding Rate
Funding rate adjusts based on perpetual-spot premium, making it more realistic.

**Usage:**
```python
result = backtest_signals(
    df=df,
    signals=signals,
    use_dynamic_funding=True,
    funding_rate_bps=1.0,          # Base funding rate
    funding_interval_hours=8,
    premium_sensitivity=0.15,       # Sensitivity to premium
    index_prices=spot_prices,       # Optional spot prices
)
```

**Parameters:**
- `funding_rate_bps`: Base funding rate in basis points
- `funding_interval_hours`: Hours between payments (8 for crypto, 24 for some FX)
- `premium_sensitivity`: How much funding responds to premium (0.1-0.2 typical)
- `index_prices`: Optional Series of spot prices (uses price momentum if not provided)

**Notes:**
- Positive positions (longs) pay positive funding rates
- Negative positions (shorts) receive positive funding rates
- Funding is typically applied at 00:00, 08:00, 16:00 UTC for crypto

## Cost Presets

Pre-configured cost models for common markets:

```python
from openquant.backtest import CostPreset

# FX major pairs on ECN broker
preset = CostPreset.fx_major_ecn()

# Crypto spot on tier-1 exchange
preset = CostPreset.crypto_spot_tier1()

# Crypto perpetual swaps
preset = CostPreset.crypto_perp_tier1(avg_funding_rate_bps=1.0)

# Altcoin spot (higher costs)
preset = CostPreset.altcoin_spot()

# Use preset in backtest
result = backtest_signals(df, signals, **preset)
```

**Available Presets:**
- `fx_major_ecn()` - FX majors on ECN (tight spreads, low fees)
- `fx_major_retail()` - FX majors on retail broker (wider spreads)
- `crypto_spot_tier1()` - Top-tier crypto exchange (Binance, Coinbase)
- `crypto_spot_tier2()` - Smaller crypto exchange
- `crypto_perp_tier1(funding_rate)` - Perpetual swaps on tier-1 exchange
- `altcoin_spot()` - Altcoin trading (higher costs)
- `paper_trading_conservative()` - Conservative assumptions
- `paper_trading_optimistic()` - Optimistic assumptions

## Utility Functions

### Compare Presets
```python
from openquant.backtest import compare_presets

comparison = compare_presets(
    "fx_major_ecn",
    "crypto_spot_tier1",
    "crypto_perp_tier1"
)
print(comparison)
```

### Estimate Total Costs
```python
from openquant.backtest import estimate_total_cost, CostPreset

preset = CostPreset.crypto_spot_tier1()
costs = estimate_total_cost(
    preset=preset,
    avg_position_changes_per_day=4.0,
    avg_holding_period_days=0.5,
)

print(f"Per-trade cost: {costs['per_trade_cost_bps']:.2f} bps")
print(f"Breakeven return: {costs['breakeven_return_bps']:.2f} bps")
```

## Complete Example

```python
from openquant.backtest import backtest_signals, CostPreset
import pandas as pd

# Load your data
df = pd.read_csv("ohlcv_data.csv", index_col=0, parse_dates=True)
signals = generate_signals(df)  # Your signal generation logic

# Option 1: Use a preset
preset = CostPreset.crypto_perp_tier1(avg_funding_rate_bps=1.0)
result = backtest_signals(df, signals, weight=1.0, leverage=2.0, **preset)

# Option 2: Customize everything
result = backtest_signals(
    df=df,
    signals=signals,
    fee_bps=2.5,
    spread_bps=2.0,
    slippage_bps=1.0,
    weight=1.0,
    leverage=2.0,
    # Time-of-day spread
    use_tod_spread=True,
    tod_multipliers=None,  # Uses default
    # Volume slippage
    use_volume_slippage=True,
    volume_impact_coeff=0.15,
    # Market impact
    use_market_impact=True,
    participation_rate=0.05,
    impact_exponent=0.6,
    # Dynamic funding
    use_dynamic_funding=True,
    funding_rate_bps=1.0,
    funding_interval_hours=8,
    premium_sensitivity=0.15,
)

# Analyze results
print(f"Final Return: {(result.equity_curve.iloc[-1] - 1) * 100:.2f}%")
print(f"Number of Trades: {result.trades[result.trades > 0].sum()}")
```

## Standalone Functions

You can also use the cost calculation functions independently:

```python
from openquant.backtest import (
    calculate_tod_spread,
    calculate_volume_slippage,
    calculate_market_impact,
    calculate_funding_rate,
    calculate_dynamic_funding_rate,
)

# Calculate time-of-day spread
tod_spreads = calculate_tod_spread(
    timestamps=df.index,
    base_spread_bps=2.0,
    tod_multipliers=None,  # Uses default
)

# Calculate volume-dependent slippage
slippage = calculate_volume_slippage(
    volumes=df["Volume"],
    position_changes=pos_changes,
    base_slippage_bps=0.5,
    volume_impact_coeff=0.15,
)

# Calculate market impact
impact = calculate_market_impact(
    prices=df["Close"],
    position_changes=pos_changes,
    volumes=df["Volume"],
    weight=1.0,
    leverage=1.0,
    participation_rate=0.05,
    impact_exponent=0.6,
)

# Calculate funding rate
funding = calculate_funding_rate(
    timestamps=df.index,
    positions=positions,
    funding_rate_bps=1.0,
    funding_interval_hours=8,
)

# Calculate dynamic funding rate
dynamic_funding = calculate_dynamic_funding_rate(
    timestamps=df.index,
    positions=positions,
    prices=df["Close"],
    index_prices=spot_prices,
    base_funding_bps=1.0,
    funding_interval_hours=8,
    premium_sensitivity=0.15,
)
```

## Implementation Notes

1. **Backward Compatibility**: All enhancements are opt-in. Existing code continues to work without changes.

2. **Performance**: The models use vectorized pandas operations for efficiency.

3. **Data Requirements**:
   - Time-of-day spread: Requires DatetimeIndex
   - Volume slippage: Requires 'Volume' column
   - Market impact: Requires 'Volume' column
   - Funding rates: Requires DatetimeIndex

4. **Typical Parameter Ranges**:
   - `volume_impact_coeff`: 0.1-0.3
   - `participation_rate`: 0.01-0.1
   - `impact_exponent`: 0.5-0.7
   - `funding_rate_bps`: 0.5-2.0
   - `premium_sensitivity`: 0.1-0.2

## References

- Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions"
- Cont, R., Kukanov, A., & Stoikov, S. (2014). "The price impact of order book events"
- Crypto perpetual funding rates: Binance, Bybit documentation

## Testing

Run the example script to see all features in action:

```bash
python scripts/transaction_cost_models_example.py
```

This will demonstrate:
- Basic usage with enhanced cost models
- Market impact comparison
- Funding rate calculations
- Using cost presets
- Cost estimation utilities
- All features combined
