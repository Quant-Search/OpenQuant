# Performance Tracker Quick Start

## 1. Basic Usage (5 minutes)

```python
from openquant.monitoring import PerformanceTracker

# Initialize
tracker = PerformanceTracker(
    backtest_sharpe=2.5,
    backtest_max_drawdown=0.15,
    alert_threshold=0.20
)

# Update with each trade/period
tracker.update(equity=105000.0, returns=0.02)

# Get metrics
metrics = tracker.get_current_metrics()
print(f"Sharpe: {metrics['sharpe_ratio']:.3f}")
print(f"Drawdown: {metrics['drawdown']:.2%}")
```

## 2. With Correlation Tracking

```python
import numpy as np

# Provide backtest correlation matrix
backtest_corr = np.array([
    [1.0, 0.6, 0.3],
    [0.6, 1.0, 0.5],
    [0.3, 0.5, 1.0]
])

tracker = PerformanceTracker(
    backtest_correlation_matrix=backtest_corr,
    backtest_sharpe=2.0
)

# Update with symbol-level returns
tracker.update(
    equity=100000.0,
    symbol_returns={
        "BTC/USD": 0.02,
        "ETH/USD": 0.015,
        "SOL/USD": 0.018
    }
)

# Check correlation drift
metrics = tracker.get_current_metrics()
if metrics['correlation_drift'] and metrics['correlation_drift'] > 0.5:
    print("⚠️  Correlation structure has drifted!")
```

## 3. Circuit Breaker Pattern

```python
tracker = PerformanceTracker(
    backtest_sharpe=2.5,
    backtest_max_drawdown=0.10,
    alert_threshold=0.20
)

# In your trading loop
while trading:
    # Execute trades...
    
    tracker.update(equity=current_equity)
    metrics = tracker.get_current_metrics()
    
    # Halt if performance degrades
    if metrics['drawdown'] > 0.25:
        print("Circuit breaker triggered!")
        break
    
    if metrics['sharpe_degradation_pct'] > 40:
        print("Performance degraded significantly!")
        break
```

## 4. Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backtest_sharpe` | None | Reference Sharpe from backtest |
| `backtest_max_drawdown` | None | Reference max drawdown (as positive value) |
| `alert_threshold` | 0.20 | Alert when metrics degrade by this % |
| `sharpe_halflife_days` | 30 | Half-life for exponential weighting |
| `freq` | "1d" | Data frequency: "1d", "1h", "4h", etc. |

## 5. Alert Configuration

Set environment variables:

```bash
# Webhook
export OPENQUANT_ALERT_WEBHOOK="https://hooks.slack.com/..."

# Email
export OPENQUANT_SMTP_HOST="smtp.gmail.com"
export OPENQUANT_SMTP_PORT="587"
export OPENQUANT_SMTP_USER="you@gmail.com"
export OPENQUANT_SMTP_PASS="your-password"
export OPENQUANT_SMTP_TO="alerts@yourdomain.com"
```

## 6. Metrics Returned

```python
metrics = tracker.get_current_metrics()
```

Returns dict with:
- `sharpe_ratio`: Current EW Sharpe
- `drawdown`: Current drawdown from peak
- `peak_equity`: Peak equity achieved
- `current_equity`: Current equity
- `correlation_drift`: Frobenius norm vs backtest (if enabled)
- `sharpe_degradation_pct`: % degradation from backtest
- `drawdown_increase_pct`: % increase from backtest
- `num_observations`: Number of data points

## 7. Full Examples

See:
- `examples/monitoring_example.py` - Complete simulation
- `examples/monitoring_integration.py` - Integration patterns
- `openquant/monitoring/README.md` - Full documentation

## 8. Testing

```bash
pytest tests/test_monitoring.py -v -k performance_tracker
```
