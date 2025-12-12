# Performance Monitoring Module

Real-time performance monitoring system with exponentially-weighted metrics, drawdown tracking, correlation drift detection, and automated alerting.

## Features

### 1. Exponentially-Weighted Rolling Sharpe Ratio
- Uses configurable half-life (default: 30 days)
- More recent observations get higher weight
- Adapts quickly to changing market conditions
- Properly annualized based on data frequency

### 2. Drawdown Tracking from Peak
- Tracks peak equity continuously
- Computes current drawdown relative to peak
- Maintains full equity curve history

### 3. Correlation Drift Detection
- Compares live correlation matrix vs backtest reference
- Uses Frobenius norm to measure matrix distance
- Alerts when correlation structure changes significantly
- Helps detect regime changes and strategy behavior shifts

### 4. Automated Alert Triggering
- Alerts when metrics degrade beyond threshold (default: 20%)
- Cooldown period prevents alert spam (default: 1 hour)
- Integrates with `openquant/utils/alerts.py`
- Supports multiple alert channels (webhook, email, SMS)

## Usage

### Basic Setup

```python
from openquant.monitoring import PerformanceTracker
import numpy as np

# Create reference correlation matrix from backtest
backtest_corr = np.array([
    [1.0, 0.6, 0.3],
    [0.6, 1.0, 0.5],
    [0.3, 0.5, 1.0]
])

# Initialize tracker
tracker = PerformanceTracker(
    backtest_correlation_matrix=backtest_corr,
    backtest_sharpe=2.5,
    backtest_max_drawdown=0.15,
    alert_threshold=0.20,  # Alert on 20% degradation
    sharpe_halflife_days=30,
    freq="1d"
)
```

### Updating with New Data

```python
# Update with portfolio-level data
tracker.update(
    equity=current_equity,
    returns=portfolio_return,  # Optional, computed from equity if not provided
    symbol_returns={"BTC/USD": 0.02, "ETH/USD": 0.015},  # For correlation tracking
    timestamp=datetime.now()  # Optional, defaults to now
)
```

### Retrieving Metrics

```python
# Get current metrics
metrics = tracker.get_current_metrics()
print(f"Sharpe: {metrics['sharpe_ratio']:.3f}")
print(f"Drawdown: {metrics['drawdown']:.2%}")
print(f"Correlation Drift: {metrics['correlation_drift']:.3f}")

# Get human-readable summary
print(tracker.get_summary())

# Get historical metrics
history = tracker.get_metrics_history(lookback_days=30)
```

## Alert Configuration

Alerts are triggered when:
- **Sharpe Ratio** degrades by more than threshold (default: 20%)
- **Drawdown** increases beyond threshold relative to backtest
- **Correlation Drift** exceeds 0.5 (Frobenius norm)

Configure alerts via environment variables:
```bash
# Webhook
export OPENQUANT_ALERT_WEBHOOK="https://your-webhook-url.com"

# Email
export OPENQUANT_SMTP_HOST="smtp.gmail.com"
export OPENQUANT_SMTP_PORT="587"
export OPENQUANT_SMTP_USER="your-email@gmail.com"
export OPENQUANT_SMTP_PASS="your-password"
export OPENQUANT_SMTP_TO="alerts@your-domain.com"

# SMS (via webhook)
export OPENQUANT_SMS_WEBHOOK="https://your-sms-gateway.com"
```

## Implementation Details

### Exponential Weighting Formula

The exponentially-weighted Sharpe ratio uses:
- Decay parameter: `λ = exp(-ln(2) / half_life)`
- Weight for observation i: `w_i = (1 - λ) * λ^i`
- Weights are normalized to sum to 1

### Correlation Drift Calculation

1. Align returns across symbols using common timestamps
2. Compute live correlation matrix
3. Calculate Frobenius norm: `||C_live - C_backtest||_F`
4. Alert if norm exceeds threshold

### Thread Safety

All public methods are thread-safe using `threading.Lock()`.

## Example

See `examples/monitoring_example.py` for a complete working example demonstrating:
- Simulated trading session over 100 days
- Performance degradation and recovery
- Correlation structure changes
- Alert triggering

Run the example:
```bash
python examples/monitoring_example.py
```

## Integration with Existing Systems

### With Paper Trading

```python
from openquant.monitoring import PerformanceTracker
from openquant.paper import PaperBroker

tracker = PerformanceTracker(
    backtest_sharpe=2.0,
    backtest_max_drawdown=0.10
)

# In your trading loop
equity = broker.get_equity()
returns = (equity - prev_equity) / prev_equity
tracker.update(equity=equity, returns=returns)
```

### With Live Broker

```python
# After each trade or periodic update
current_equity = broker.get_account_value()
tracker.update(equity=current_equity)

# Get real-time metrics
metrics = tracker.get_current_metrics()
if metrics['drawdown'] > 0.25:
    # Trigger emergency stop
    pass
```

## Performance Considerations

- History is stored in memory (dataclass list)
- For long-running systems, consider periodic pruning of old metrics
- Symbol-level returns are stored separately for correlation tracking
- Correlation computation is O(n*m) where n=symbols, m=timestamps

## Dependencies

- `numpy` - For matrix operations and statistics
- `pandas` - For correlation computation
- `threading` - For thread safety
- `openquant.utils.logging` - For logging
- `openquant.utils.alerts` - For alert dispatch
