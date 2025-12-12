# Analysis Module

The analysis module provides tools for monitoring and analyzing trading performance, execution quality, and market sentiment.

## Components

### TCA Monitor (`tca.py`)

Transaction Cost Analysis - tracks order execution details and slippage.

**Features:**
- Log order submissions with arrival price
- Update with fill details (price, quantity, fees)
- Calculate slippage in basis points
- Persist data to DuckDB
- Query aggregate statistics

**Usage:**
```python
from openquant.analysis.tca import TCAMonitor

tca = TCAMonitor(db_path="data/tca.duckdb")

# Log order
tca.log_order("ORD123", "BTC/USD", "buy", 1.0, 50000.0)

# Update when filled
tca.update_fill("ORD123", fill_price=50005.0, fill_qty=1.0, fee=2.5)

# Get statistics
stats = tca.get_stats()
print(f"Avg Slippage: {stats['avg_slippage_bps']:.2f} bps")
```

### Execution Quality Monitor (`execution_quality.py`)

Comprehensive monitoring of order execution quality with automated alerts.

**Features:**
- Fill rate tracking
- Rejection rate monitoring
- Slippage distribution analysis
- Benchmark comparison
- Automated degradation alerts
- Historical metrics snapshots

**Usage:**
```python
from openquant.analysis.execution_quality import ExecutionQualityMonitor

monitor = ExecutionQualityMonitor(
    fill_rate_threshold=0.95,
    slippage_threshold_bps=10.0
)

# Run full monitoring cycle
metrics, alerts = monitor.monitor(save_snapshot=True)

# Check for issues
for alert in alerts:
    if alert.severity == "critical":
        print(f"ALERT: {alert.message}")
```

**Key Metrics:**
- Fill Rate: % of orders filled
- Rejection Rate: % of orders rejected
- Partial Fill Rate: % of partial fills
- Avg/Median Slippage: Slippage in basis points
- Slippage Volatility: Standard deviation
- Fill Time: Average execution time
- Fees: Total and average per trade

**Alert Types:**
- `fill_rate_degradation`: Fill rate below threshold
- `rejection_rate_high`: Too many rejections
- `slippage_high`: Average slippage too high
- `slippage_volatility_high`: Inconsistent execution

See `EXECUTION_QUALITY.md` for detailed documentation.

### Sentiment Analyzer (`sentiment.py`)

Market sentiment analysis for signal adjustment.

**Features:**
- Crypto Fear & Greed Index
- Signal confidence modifiers
- Contrarian approach

**Usage:**
```python
from openquant.analysis.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# Get current sentiment
sentiment = analyzer.get_crypto_fear_greed()
print(f"Fear & Greed: {sentiment['value']} ({sentiment['class']})")

# Adjust signal
modifier = analyzer.get_signal_modifier(sentiment['value'])
adjusted_signal = base_signal * modifier
```

## Integration

### With Brokers

The Alpaca broker automatically integrates TCA:

```python
from openquant.broker.alpaca_broker import AlpacaBroker

broker = AlpacaBroker()
# TCA is initialized automatically
# Orders are logged automatically when placed

# Sync TCA with filled orders
broker.sync_tca()
```

### With Trading Loop

```python
from openquant.analysis.tca import TCAMonitor
from openquant.analysis.execution_quality import ExecutionQualityMonitor

tca = TCAMonitor()
monitor = ExecutionQualityMonitor()

# In trading loop:
while trading:
    # Place orders
    order = broker.place_order(...)
    
    # TCA tracks automatically via broker
    
    # Periodic quality check (e.g., hourly)
    if should_check_quality():
        metrics, alerts = monitor.monitor(save_snapshot=True)
        handle_alerts(alerts)
```

## Database Schema

All analysis modules use DuckDB for persistence:

**Location:** `data/tca.duckdb`

**Tables:**
- `orders`: TCA order tracking
- `execution_alerts`: Quality degradation alerts
- `execution_quality_snapshots`: Historical metrics

## Scripts

- `scripts/monitor_execution_quality.py`: Standalone monitoring tool
- `scripts/execution_quality_integration_example.py`: Integration examples

## Best Practices

1. **Regular Monitoring**: Run execution quality checks hourly or after N orders
2. **Set Appropriate Thresholds**: Calibrate based on trading style and asset class
3. **React to Critical Alerts**: Implement automated responses (pause trading, adjust orders)
4. **Trend Analysis**: Review weekly snapshots to identify gradual degradation
5. **Symbol-Specific Analysis**: Different assets have different execution characteristics
6. **Combine with Risk**: Integrate alerts with circuit breakers and kill switches

## Performance

- Optimized database queries with indexes
- Efficient aggregations for large order volumes
- Minimal overhead on trading loop
- Suitable for high-frequency monitoring

## Testing

```bash
pytest tests/test_execution_quality.py -v
```
