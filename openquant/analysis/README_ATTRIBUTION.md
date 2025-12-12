# Performance Attribution Module

Comprehensive performance attribution system that decomposes trading returns into actionable components.

## Overview

The Performance Attribution module provides detailed analysis of where returns come from by breaking them down into four key effects:

1. **Timing Effect**: Quality of entry and exit timing relative to price extremes
2. **Selection Effect**: Contribution from instrument and strategy choices
3. **Sizing Effect**: Impact of position sizing decisions
4. **Cost Drag**: Transaction costs including fees, slippage, and funding

## Features

- **Trade-Level Analysis**: Detailed attribution for each completed trade
- **Strategy Comparison**: Compare performance across different strategies
- **Instrument Comparison**: Analyze which instruments perform best
- **Execution Quality Metrics**: Track signal-to-execution lag and execution rates
- **Comprehensive Reporting**: Generate JSON reports with full attribution breakdown

## Quick Start

### Basic Usage

```python
from openquant.analysis.attribution import PerformanceAttributor

# Initialize attributor
attributor = PerformanceAttributor()

# Analyze last 30 days
result = attributor.analyze(days=30)

# Print summary
print(result.summary())
```

### Quick Analysis

```python
from openquant.analysis import quick_attribution

# One-liner for quick attribution
result = quick_attribution(days=7)
```

## Detailed Usage

### Strategy Comparison

```python
from openquant.analysis.attribution import PerformanceAttributor

attributor = PerformanceAttributor()

# Compare all strategies over last 30 days
comparison = attributor.compare_strategies(days=30)

for strategy, metrics in comparison.items():
    print(f"{strategy}: {metrics['total_return']:.2%} return")
    print(f"  Timing Effect: {metrics['timing_effect']:.2%}")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
```

### Instrument Comparison

```python
# Compare performance across instruments
comparison = attributor.compare_instruments(days=30)

for symbol, metrics in comparison.items():
    print(f"{symbol}: {metrics['total_pnl']:.2f} P&L")
    print(f"  Win Rate: {metrics['win_rate']:.1%}")
```

### Trade-Level Attribution

```python
# Get detailed attribution for individual trades
trades = attributor.get_trade_level_attribution(days=7)

for trade in trades:
    print(f"{trade.symbol}: {trade.pnl_pct:.2%}")
    print(f"  Timing Quality: {trade.timing_quality:.1%}")
    print(f"  Holding Period: {trade.holding_period_hours:.1f} hours")
```

### Generate Comprehensive Report

```python
# Generate full attribution report
report = attributor.generate_report(
    days=30,
    output_path="data/attribution_report.json"
)

# Report includes:
# - Overall attribution breakdown
# - Strategy comparison
# - Instrument comparison
# - Sample trade-level data
# - Summary statistics
```

## Command-Line Interface

The module includes a CLI script for easy analysis:

```bash
# Basic attribution analysis
python scripts/run_attribution_analysis.py --days 30

# Compare strategies
python scripts/run_attribution_analysis.py --compare-strategies --days 30

# Compare instruments
python scripts/run_attribution_analysis.py --compare-instruments --days 30

# Show trade-level attribution
python scripts/run_attribution_analysis.py --trade-level --days 7

# Generate comprehensive report
python scripts/run_attribution_analysis.py --report --output data/my_report.json
```

## Attribution Components Explained

### Timing Effect

Measures how well entries and exits were timed relative to price extremes during the period.

- **Positive**: Buying near lows, selling near highs
- **Negative**: Buying near highs, selling near lows
- **Calculation**: Compares entry/exit prices to min/max prices in signal history

### Selection Effect

Measures contribution from choosing specific instruments and strategies.

- **Positive**: Strategies/instruments outperformed equal-weight average
- **Negative**: Strategies/instruments underperformed
- **Calculation**: Deviation of strategy/instrument returns from portfolio average

### Sizing Effect

Measures impact of position sizing decisions on returns.

- **Positive**: Larger positions in winners, smaller in losers
- **Negative**: Larger positions in losers, smaller in winners
- **Calculation**: Difference between actual weighted returns and equal-sized returns

### Cost Drag

Captures all transaction costs reducing returns.

- **Always negative** (costs reduce returns)
- **Components**: Trading fees, slippage, funding costs
- **Calculation**: Aggregates all costs from TCA monitor and audit trail

## Integration with Audit Trail

The attribution module tightly integrates with the audit trail system:

```python
from openquant.storage.audit_trail import AuditTrail

audit = AuditTrail()

# Log trading events for attribution
audit.log_signal(symbol="BTC/USD", strategy="kalman", side="BUY", price=50000)
audit.log_execution(symbol="BTC/USD", strategy="kalman", side="BUY", 
                   quantity=0.1, price=50010)
                   
# Later, run attribution
attributor = PerformanceAttributor(audit_trail=audit)
result = attributor.analyze(days=1)
```

## Advanced Features

### Custom Time Ranges

```python
from datetime import datetime, timedelta

end_time = datetime.now()
start_time = end_time - timedelta(days=7)

result = attributor.analyze(start_time=start_time, end_time=end_time)
```

### Execution Quality Metrics

```python
from openquant.storage.audit_trail import AuditTrail

audit = AuditTrail()

# Get execution quality metrics
metrics = audit.get_execution_quality_metrics(start_time, end_time)
print(f"Execution Rate: {metrics['execution_rate']:.1%}")

# Get signal-to-execution lag
lag = audit.get_signal_to_execution_lag(start_time, end_time)
print(f"Average Lag: {lag['avg_lag_seconds']:.2f}s")
```

### Custom TCA Integration

```python
from openquant.analysis.tca import TCAMonitor

tca = TCAMonitor()

# Log orders for cost analysis
tca.log_order("order_1", "BTC/USD", "buy", 0.1, 50000)
tca.update_fill("order_1", 50010, 0.1, fee=5.0)

# Attribution will automatically use TCA data
attributor = PerformanceAttributor(tca_monitor=tca)
result = attributor.analyze(days=1)
```

## Data Structures

### AttributionResult

```python
@dataclass
class AttributionResult:
    period_start: datetime
    period_end: datetime
    total_return: float
    timing_effect: float
    selection_effect: float
    sizing_effect: float
    cost_drag: float
    residual: float
    details: Dict[str, Any]
```

### TradeAttribution

```python
@dataclass
class TradeAttribution:
    symbol: str
    strategy: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    pnl: float
    pnl_pct: float
    timing_quality: float
    execution_quality: float
    cost_impact: float
    holding_period_hours: float
```

## Best Practices

1. **Regular Analysis**: Run attribution analysis daily or weekly to track performance trends
2. **Strategy Comparison**: Use strategy comparison to identify top performers
3. **Cost Monitoring**: Monitor cost drag to optimize execution
4. **Timing Improvement**: Use timing effect to improve entry/exit logic
5. **Position Sizing**: Use sizing effect to optimize position sizing algorithms

## Performance Considerations

- **Database Queries**: Attribution queries the audit trail database; ensure indexes are in place
- **Memory Usage**: Trade-level attribution loads all trades into memory; use time filters for large datasets
- **Computation Time**: Full attribution analysis may take several seconds for large datasets

## Troubleshooting

### No Trades Found

```python
result = attributor.analyze(days=30)
if result.details.get('message') == 'No trades in period':
    print("No completed trades in the specified period")
    # Check audit trail has data
    summary = audit.get_summary(days=30)
    print(f"Events in period: {summary}")
```

### Missing Attribution Components

If certain effects are always zero:
- **Timing Effect**: Requires SIGNAL events in audit trail
- **Cost Drag**: Requires TCA monitor data or execution details
- **Selection Effect**: Requires multiple strategies or instruments

## Examples

See `scripts/run_attribution_analysis.py` for comprehensive examples of all features.

## Related Modules

- `openquant.storage.audit_trail`: Event logging for attribution
- `openquant.analysis.tca`: Transaction cost analysis
- `openquant.reporting.performance_tracker`: Real-time performance tracking
- `openquant.reporting.trade_analyzer`: Trade pattern analysis
