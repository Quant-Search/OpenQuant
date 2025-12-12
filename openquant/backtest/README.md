# Backtest Module

Enhanced transaction cost modeling for realistic backtesting.

## Quick Start

```python
from openquant.backtest import backtest_signals

# Basic backtest with enhanced costs
result = backtest_signals(
    df=df,
    signals=signals,
    fee_bps=1.0,
    spread_bps=5.0,
    use_enhanced_costs=True
)
```

## Features

### 1. Time-of-Day Spread Adjustment
Spreads widen during illiquid hours (overnight, pre-market, after-hours).

```python
from openquant.backtest import SpreadSchedule

schedule = SpreadSchedule(base_spread_bps=5.0)
```

### 2. Volume-Dependent Market Impact
Square-root market impact model: `slippage = λ × √(order/volume) × volatility`

```python
from openquant.backtest import MarketImpactModel

impact = MarketImpactModel(impact_coeff=0.1)
```

### 3. Tick Size Constraints
Realistic price execution at discrete tick levels.

```python
from openquant.backtest import TickRounder

rounder = TickRounder(tick_size=0.01)  # $0.01 for stocks
```

## Example

```python
from openquant.backtest import (
    backtest_signals,
    SpreadSchedule,
    MarketImpactModel,
    TickRounder
)

result = backtest_signals(
    df=df,
    signals=signals,
    fee_bps=1.0,
    spread_schedule=SpreadSchedule(base_spread_bps=3.0),
    impact_model=MarketImpactModel(impact_coeff=0.15),
    tick_rounder=TickRounder(tick_size=0.01),
    use_enhanced_costs=True
)

print(f"Final Return: {(result.equity_curve.iloc[-1] - 1.0) * 100:.2f}%")
```

## Files

- `engine.py` - Core backtest engine with cost integration
- `cost_model.py` - Transaction cost components (SpreadSchedule, MarketImpactModel, TickRounder)
- `metrics.py` - Performance metrics (Sharpe, Sortino, etc.)
- `gpu_backtest.py` - GPU-accelerated backtesting

## See Also

- `scripts/example_enhanced_costs.py` - Complete usage examples
- AGENTS.md - Repository setup and commands
