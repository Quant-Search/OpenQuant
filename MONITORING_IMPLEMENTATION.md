# Performance Monitoring Implementation Summary

## Overview

Implemented a comprehensive real-time performance monitoring system in `openquant/monitoring/performance_tracker.py` with the following features:

### Core Features Implemented

1. **Exponentially-Weighted Rolling Sharpe Ratio**
   - Configurable half-life (default: 30 days)
   - Proper exponential decay weighting: λ = exp(-ln(2) / half_life)
   - Recent observations weighted more heavily
   - Fully annualized based on data frequency

2. **Drawdown Tracking from Peak**
   - Continuous peak equity tracking
   - Real-time drawdown calculation: (peak - current) / peak
   - Full equity curve history maintained

3. **Correlation Drift Detection**
   - Compares live vs backtest correlation matrices
   - Uses Frobenius norm: ||C_live - C_backtest||_F
   - Handles multi-symbol portfolios
   - Automatic timestamp alignment across symbols

4. **Automated Alert Triggering**
   - Configurable degradation threshold (default: 20%)
   - Alerts on Sharpe ratio degradation
   - Alerts on excessive drawdown
   - Alerts on correlation drift (>0.5 Frobenius norm)
   - 1-hour cooldown to prevent alert spam
   - Full integration with `openquant/utils/alerts.py`

## Files Created

### Core Implementation
- `openquant/monitoring/__init__.py` - Module exports
- `openquant/monitoring/performance_tracker.py` - Main implementation (370 lines)
  - `PerformanceMetrics` dataclass
  - `PerformanceTracker` class with all features

### Documentation
- `openquant/monitoring/README.md` - Comprehensive documentation
- `openquant/monitoring/QUICKSTART.md` - Quick start guide

### Examples
- `examples/monitoring_example.py` - Complete simulation example
- `examples/monitoring_integration.py` - Integration patterns with brokers

### Tests
- `tests/test_monitoring.py` - Extended with 9 new test functions:
  - `test_performance_tracker_basic`
  - `test_performance_tracker_drawdown`
  - `test_performance_tracker_ew_sharpe`
  - `test_performance_tracker_correlation_drift`
  - `test_performance_tracker_alert_triggering`
  - `test_performance_tracker_metrics_history`
  - `test_performance_tracker_thread_safety`
  - `test_performance_tracker_summary`

## Key Implementation Details

### Exponential Weighting Formula

```python
decay = exp(-ln(2) / half_life)
weight_i = (1 - decay) * decay^i
# Weights normalized to sum to 1
```

### Correlation Drift Calculation

1. Align returns across symbols using common timestamps
2. Build returns matrix for live trading
3. Compute correlation matrix
4. Calculate Frobenius norm of difference from backtest
5. Alert if norm exceeds threshold

### Thread Safety

- All public methods protected with `threading.Lock()`
- Safe for concurrent access from multiple threads
- Suitable for multi-threaded trading systems

## API Summary

### Initialization

```python
tracker = PerformanceTracker(
    backtest_correlation_matrix=np.ndarray,  # Optional
    backtest_sharpe=float,                   # Optional
    backtest_max_drawdown=float,             # Optional
    alert_threshold=0.20,                    # Default: 20%
    sharpe_halflife_days=30,                 # Default: 30 days
    freq="1d"                                # "1d", "1h", "4h", etc.
)
```

### Update Method

```python
tracker.update(
    equity=float,                    # Required
    returns=float,                   # Optional, computed if not provided
    symbol_returns=Dict[str, float], # Optional, for correlation tracking
    timestamp=datetime               # Optional, defaults to now
)
```

### Metrics Retrieval

```python
metrics = tracker.get_current_metrics()
# Returns: dict with sharpe_ratio, drawdown, correlation_drift, etc.

history = tracker.get_metrics_history(lookback_days=30)
# Returns: list of historical metric snapshots

summary = tracker.get_summary()
# Returns: human-readable string summary
```

## Alert Integration

Integrates with `openquant/utils/alerts.py` supporting:
- Webhook notifications (env: `OPENQUANT_ALERT_WEBHOOK`)
- Email notifications (env: `OPENQUANT_SMTP_*`)
- SMS via webhook (env: `OPENQUANT_SMS_WEBHOOK`)

Alerts triggered when:
- Sharpe ratio degrades >threshold from backtest
- Drawdown exceeds threshold above backtest
- Correlation drift Frobenius norm >0.5

## Usage Patterns

### Pattern 1: Basic Monitoring
```python
tracker = PerformanceTracker(backtest_sharpe=2.5)
tracker.update(equity=current_equity)
print(tracker.get_summary())
```

### Pattern 2: With Correlation Tracking
```python
tracker = PerformanceTracker(
    backtest_correlation_matrix=backtest_corr,
    backtest_sharpe=2.5
)
tracker.update(
    equity=equity,
    symbol_returns={"BTC/USD": 0.02, "ETH/USD": 0.015}
)
```

### Pattern 3: Circuit Breaker
```python
metrics = tracker.get_current_metrics()
if metrics['drawdown'] > 0.25 or metrics['sharpe_degradation_pct'] > 40:
    halt_trading()
```

## Performance Considerations

- All metrics computed incrementally
- History stored in memory (consider pruning for long-running systems)
- Correlation computation: O(n*m) where n=symbols, m=common timestamps
- Thread-safe but lock contention possible under high update frequency

## Dependencies

- `numpy` - Matrix operations and statistics
- `pandas` - Correlation computation
- `threading` - Thread safety
- Standard library: `dataclasses`, `datetime`, `typing`

## Testing

Run tests:
```bash
pytest tests/test_monitoring.py -v -k performance_tracker
```

Run examples:
```bash
python examples/monitoring_example.py
python examples/monitoring_integration.py
```

## Integration Points

Can be integrated with:
- Paper trading system (`openquant.paper`)
- Live brokers (`openquant.broker`)
- Backtesting engine (`openquant.backtest`)
- Dashboard (`openquant.gui.dashboard`)
- Research workflows (`openquant.research`)

## Next Steps (Not Implemented)

Potential future enhancements:
- Persistence to disk/database
- Web dashboard integration
- More sophisticated regime detection
- Automatic parameter adjustment based on drift
- Multi-strategy decomposition of metrics
