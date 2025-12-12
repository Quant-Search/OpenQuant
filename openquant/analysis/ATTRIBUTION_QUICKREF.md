# Performance Attribution Quick Reference

## Installation & Setup

No additional setup required. The module integrates with existing audit trail and TCA systems.

```python
from openquant.analysis.attribution import PerformanceAttributor
```

## Core Methods

### Quick Analysis
```python
from openquant.analysis import quick_attribution
result = quick_attribution(days=30)  # Analyze last 30 days
print(result.summary())
```

### Full Analysis
```python
attributor = PerformanceAttributor()
result = attributor.analyze(days=30)

# Access components
print(f"Total Return: {result.total_return:.2%}")
print(f"Timing Effect: {result.timing_effect:.2%}")
print(f"Selection Effect: {result.selection_effect:.2%}")
print(f"Sizing Effect: {result.sizing_effect:.2%}")
print(f"Cost Drag: {result.cost_drag:.2%}")
```

### Strategy Comparison
```python
comparison = attributor.compare_strategies(days=30)
for strategy, metrics in comparison.items():
    print(f"{strategy}: {metrics['total_return']:.2%}")
```

### Instrument Comparison
```python
comparison = attributor.compare_instruments(days=30)
for symbol, metrics in comparison.items():
    print(f"{symbol}: {metrics['win_rate']:.1%}")
```

### Trade-Level Attribution
```python
trades = attributor.get_trade_level_attribution(days=7)
for trade in trades:
    print(f"{trade.symbol}: {trade.pnl_pct:.2%}, timing: {trade.timing_quality:.1%}")
```

### Generate Report
```python
report = attributor.generate_report(
    days=30,
    output_path="data/attribution_report.json"
)
```

## CLI Usage

```bash
# Basic analysis
python scripts/run_attribution_analysis.py --days 30

# Compare strategies
python scripts/run_attribution_analysis.py --compare-strategies

# Compare instruments
python scripts/run_attribution_analysis.py --compare-instruments

# Trade-level detail
python scripts/run_attribution_analysis.py --trade-level --days 7

# Full report
python scripts/run_attribution_analysis.py --report --output my_report.json
```

## Return Components

| Component | Meaning | Good/Bad |
|-----------|---------|----------|
| **Timing Effect** | Entry/exit timing quality | Positive = good timing |
| **Selection Effect** | Strategy/instrument choice | Positive = good choices |
| **Sizing Effect** | Position sizing impact | Positive = good sizing |
| **Cost Drag** | Transaction costs | Always negative |
| **Residual** | Unexplained return | Should be small |

## Integration Points

### Audit Trail
```python
from openquant.storage.audit_trail import AuditTrail

audit = AuditTrail()
audit.log_signal(symbol="BTC/USD", strategy="kalman", side="BUY", price=50000)
audit.log_execution(symbol="BTC/USD", strategy="kalman", side="BUY", quantity=0.1, price=50100)
```

### TCA Monitor
```python
from openquant.analysis.tca import TCAMonitor

tca = TCAMonitor()
tca.log_order("order_1", "BTC/USD", "buy", 0.1, 50000)
tca.update_fill("order_1", 50100, 0.1, fee=5.0)
```

### Custom Integration
```python
attributor = PerformanceAttributor(
    audit_trail=custom_audit,
    tca_monitor=custom_tca
)
```

## Advanced Queries

### Execution Quality
```python
audit = AuditTrail()
metrics = audit.get_execution_quality_metrics(start_time, end_time)
print(f"Execution Rate: {metrics['execution_rate']:.1%}")
```

### Signal Lag
```python
lag = audit.get_signal_to_execution_lag(start_time, end_time)
print(f"Avg Lag: {lag['avg_lag_seconds']:.2f}s")
```

### Trade Pairs
```python
pairs = audit.get_trade_pairs(symbol="BTC/USD", strategy="kalman")
```

## Common Patterns

### Daily Analysis
```python
def daily_attribution():
    result = quick_attribution(days=1)
    if result.cost_drag < -0.01:  # Cost > 1%
        print("WARNING: High costs detected!")
```

### Strategy Selection
```python
def best_strategy(days=30):
    comparison = PerformanceAttributor().compare_strategies(days=days)
    best = max(comparison.items(), key=lambda x: x[1]['total_return'])
    return best[0]  # Strategy name
```

### Cost Monitoring
```python
def monitor_costs(threshold=0.005):
    result = PerformanceAttributor().analyze(days=7)
    if abs(result.cost_drag) > threshold:
        print(f"Costs exceed threshold: {result.cost_drag:.2%}")
```

## Data Structures

### AttributionResult
- `period_start`, `period_end`: datetime
- `total_return`: float (total return %)
- `timing_effect`: float (timing contribution %)
- `selection_effect`: float (selection contribution %)
- `sizing_effect`: float (sizing contribution %)
- `cost_drag`: float (cost impact %)
- `residual`: float (unexplained %)
- `details`: Dict (additional metadata)

### TradeAttribution
- `symbol`, `strategy`: str
- `entry_time`, `exit_time`: datetime
- `entry_price`, `exit_price`: float
- `quantity`: float
- `side`: str ("LONG" or "SHORT")
- `pnl`: float (absolute P&L)
- `pnl_pct`: float (percentage return)
- `timing_quality`: float (0-1 score)
- `execution_quality`: float (0-1 score)
- `cost_impact`: float (cost as % of notional)
- `holding_period_hours`: float

## Common Issues

### No Trades Found
- Ensure audit trail has ORDER_EXECUTION events
- Check time range with `audit.get_summary(days=30)`

### Zero Effects
- **Timing**: Needs SIGNAL events
- **Selection**: Needs multiple strategies/instruments
- **Cost**: Needs TCA data or execution details

### High Residual
- Normal up to 1-2%
- Check for missing events in audit trail
- Verify TCA data completeness

## Performance Tips

1. Use time filters for large datasets
2. Run daily analyses off-peak
3. Archive old audit trail data
4. Limit trade-level queries to recent periods
5. Use in-memory databases for testing

## Example Workflow

```python
# 1. Initialize
attributor = PerformanceAttributor()

# 2. Daily analysis
daily = attributor.analyze(days=1)
print(daily.summary())

# 3. Weekly deep-dive
weekly = attributor.analyze(days=7)
strategies = attributor.compare_strategies(days=7)
instruments = attributor.compare_instruments(days=7)

# 4. Generate report
report = attributor.generate_report(days=30, output_path="data/weekly_report.json")

# 5. Check metrics
audit = AuditTrail()
exec_quality = audit.get_execution_quality_metrics()
print(f"Execution Rate: {exec_quality['execution_rate']:.1%}")
```

## Resources

- Full docs: `openquant/analysis/README_ATTRIBUTION.md`
- Examples: `scripts/attribution_example.py`
- CLI tool: `scripts/run_attribution_analysis.py`
- Tests: `tests/test_attribution.py`
