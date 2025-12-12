# Execution Quality Monitoring

The `ExecutionQualityMonitor` provides comprehensive tracking and analysis of order execution quality, helping identify degradation in fill rates, slippage, and overall execution performance.

## Features

- **Fill Rate Tracking**: Monitor the percentage of orders successfully filled
- **Rejection Rate Analysis**: Track order rejections and cancellations
- **Slippage Distribution**: Analyze slippage patterns with percentiles and histograms
- **TCA Benchmark Comparison**: Compare current performance against historical benchmarks
- **Automated Alerts**: Get notified when execution quality degrades below thresholds
- **Historical Snapshots**: Store and retrieve execution quality metrics over time

## Quick Start

```python
from openquant.analysis.execution_quality import ExecutionQualityMonitor

# Initialize monitor with custom thresholds
monitor = ExecutionQualityMonitor(
    db_path="data/tca.duckdb",
    fill_rate_threshold=0.95,           # 95% minimum fill rate
    rejection_rate_threshold=0.05,      # 5% maximum rejection rate
    slippage_threshold_bps=10.0,        # 10 bps maximum average slippage
    slippage_std_threshold_bps=20.0,    # 20 bps maximum slippage volatility
    lookback_hours=24                   # Look back 24 hours
)

# Run monitoring cycle
metrics, alerts = monitor.monitor(save_snapshot=True)

# Check results
print(f"Fill Rate: {metrics.fill_rate:.2%}")
print(f"Avg Slippage: {metrics.avg_slippage_bps:.2f} bps")

for alert in alerts:
    if alert.severity == "critical":
        print(f"ALERT: {alert.message}")
```

## Running Examples

**Monitor current execution quality:**
```bash
python scripts/monitor_execution_quality.py
```

**Simulate trading session with monitoring:**
```bash
python scripts/execution_quality_integration_example.py
```

## Key Metrics

### Fill Rates
- **Fill Rate**: Percentage of orders successfully filled (target: >95%)
- **Rejection Rate**: Percentage of orders rejected/cancelled (target: <5%)
- **Partial Fill Rate**: Percentage of fills that were partial

### Slippage Analysis
- **Average Slippage**: Mean slippage across all fills (in basis points)
- **Median Slippage**: 50th percentile (more robust to outliers)
- **Slippage Std Dev**: Volatility of slippage
- **Percentiles**: p5, p25, p50, p75, p95, p99

### Performance
- **Average Fill Time**: Time from order submission to execution
- **Total Fees**: Sum of trading fees
- **Average Fee (bps)**: Fees as percentage of trade value

## Alert Types

### fill_rate_degradation
Triggered when fill rate falls below threshold. Possible causes:
- Broker connectivity issues
- Market conditions (low liquidity)
- Order parameters (invalid prices)

### rejection_rate_high
Triggered when rejection rate exceeds threshold. Possible causes:
- Insufficient margin/buying power
- Position limits exceeded
- Market halted
- Broker-side restrictions

### slippage_high
Triggered when average slippage exceeds threshold. Possible causes:
- High market volatility
- Large order sizes relative to liquidity
- Market orders in illiquid markets
- Broker routing issues

### slippage_volatility_high
Triggered when slippage standard deviation is too high. Indicates:
- Inconsistent execution quality
- Intermittent liquidity issues
- Need for order size adjustments

## Integration with Brokers

The module works seamlessly with TCAMonitor, which is already integrated in AlpacaBroker:

```python
from openquant.broker.alpaca_broker import AlpacaBroker
from openquant.analysis.execution_quality import ExecutionQualityMonitor

broker = AlpacaBroker()
monitor = ExecutionQualityMonitor()

# Place orders (TCA tracks automatically)
broker.place_order("AAPL", 100, "buy")
broker.sync_tca()

# Monitor quality
metrics, alerts = monitor.monitor()
```

## Benchmark Comparison

Compare current execution quality against historical performance:

```python
metrics = monitor.calculate_metrics(lookback_hours=24)
comparison = monitor.compare_to_benchmark(metrics, benchmark_lookback_hours=168)

if comparison["has_benchmark"]:
    print(f"Fill Rate Change: {comparison['percent_changes']['fill_rate']:+.1f}%")
    print(f"Slippage Change: {comparison['percent_changes']['avg_slippage']:+.1f}%")
```

## Recommended Thresholds by Trading Style

### High-Frequency (Seconds to Minutes)
- Fill Rate: >98%
- Rejection Rate: <2%
- Avg Slippage: <5 bps
- Lookback: 1-4 hours

### Medium-Frequency (Minutes to Hours)
- Fill Rate: >95%
- Rejection Rate: <5%
- Avg Slippage: <10 bps
- Lookback: 4-24 hours

### Low-Frequency (Hours to Days)
- Fill Rate: >90%
- Rejection Rate: <10%
- Avg Slippage: <20 bps
- Lookback: 24-168 hours

## Database Schema

Two new tables extend the TCA database:

**execution_alerts**: Stores all generated alerts
**execution_quality_snapshots**: Periodic metrics snapshots for trend analysis

## Best Practices

1. Set thresholds appropriate for your trading style and assets
2. Review daily/weekly reports to identify trends
3. Use symbol-level analysis for different execution characteristics
4. Combine with circuit breakers for risk management
5. Monitor during different market conditions separately
6. Save regular snapshots for long-term analysis

## See Also

- `openquant/analysis/tca.py` - Transaction Cost Analysis
- `openquant/broker/alpaca_broker.py` - Broker integration example
- `scripts/monitor_execution_quality.py` - Monitoring script
- `scripts/execution_quality_integration_example.py` - Integration example
