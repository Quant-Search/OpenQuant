# Graceful Shutdown

OpenQuant implements comprehensive graceful shutdown handling to ensure safe termination of trading operations.

## Overview

The shutdown handler catches `SIGTERM` and `SIGINT` signals (e.g., Ctrl+C, `kill` command) and performs an orderly cleanup sequence to protect your trading positions and state.

## Shutdown Sequence

When a shutdown signal is received, the following steps are executed in order:

### 1. Stop Scheduler
- Stops the robot scheduler to prevent new trades
- Waits for any in-progress cycle to complete
- Stops position monitoring if active

### 2. Close All Positions
- Iterates through all registered brokers (MT5, Alpaca, etc.)
- Closes all open positions
- Logs the number of positions closed

### 3. Save Portfolio State
- Saves current portfolio state to `data/paper_state.json`
- Includes cash balance, holdings, entry prices, stop-loss, and take-profit levels
- Ensures state persistence across restarts

### 4. Persist Circuit Breaker Status
- Saves circuit breaker state to `data/circuit_breaker_state.json`
- Preserves peak equity, daily start equity, and trip status
- Maintains drawdown tracking across sessions

### 5. Cleanup Broker Connections
- Properly closes connections to MT5, Alpaca, and other brokers
- Calls `shutdown()` or `close()` methods on each broker
- Releases platform resources

### 6. Flush Logs
- Forces all pending log writes to disk
- Ensures complete audit trail
- Flushes both JSON and console log handlers

## Usage

### In Python Scripts

```python
import signal
from openquant.utils.shutdown_handler import SHUTDOWN_HANDLER

# Register signal handlers
signal.signal(signal.SIGINT, SHUTDOWN_HANDLER)
signal.signal(signal.SIGTERM, SHUTDOWN_HANDLER)

# Register components for cleanup
SHUTDOWN_HANDLER.register_scheduler(scheduler)
SHUTDOWN_HANDLER.register_broker(mt5_broker)
SHUTDOWN_HANDLER.register_broker(alpaca_broker)

# Your trading loop here...
```

### Automatic Registration

The `run.py` and `scripts/run_robot_cli.py` scripts automatically:
- Register signal handlers on startup
- Register the scheduler
- Register brokers as they are created during trading cycles

### Manual Shutdown

You can also trigger a shutdown programmatically:

```python
from openquant.utils.shutdown_handler import SHUTDOWN_HANDLER

# Trigger graceful shutdown
SHUTDOWN_HANDLER.shutdown()
```

## Signal Handling

### SIGINT (Ctrl+C)
- Triggered by pressing Ctrl+C in terminal
- Initiates graceful shutdown
- Safe for manual intervention

### SIGTERM (kill)
- Triggered by `kill <pid>` command
- Triggered by system shutdown
- Initiates graceful shutdown
- Allows automated termination

### Emergency Exit
If shutdown hangs or you need immediate termination:
- Send `SIGKILL`: `kill -9 <pid>` (Windows: Task Manager)
- Note: This bypasses graceful shutdown and may leave positions open

## Testing

Run the shutdown handler test:

```bash
python tests/test_shutdown_handler.py
```

This test verifies:
- Broker position closing
- Broker connection cleanup
- Scheduler stopping
- State persistence
- Circuit breaker persistence

## Logs

Shutdown events are logged to:
- `logs/openquant.log` - JSON structured logs
- Console output - Human-readable status

Example shutdown log output:

```
============================================================
  GRACEFUL SHUTDOWN INITIATED
  Signal: SIGINT
============================================================

  [1/5] Stopping scheduler...
  [2/5] Closed 3 position(s)
  [3/5] Portfolio state saved
  [4/5] Circuit breaker status saved
  [5/5] Broker cleanup complete

  Flushing logs...
  Logs flushed

============================================================
  SHUTDOWN COMPLETE
============================================================
```

## Best Practices

1. **Always use graceful shutdown** - Avoid `kill -9` unless absolutely necessary
2. **Monitor shutdown logs** - Check for errors or warnings during shutdown
3. **Verify state persistence** - After shutdown, verify state files exist:
   - `data/paper_state.json`
   - `data/circuit_breaker_state.json`
4. **Test shutdown regularly** - Run test suite to ensure shutdown reliability
5. **Register components early** - Register brokers and scheduler as soon as they're created

## Integration

The shutdown handler is integrated into:
- `run.py` - Main entry point
- `scripts/run_robot_cli.py` - CLI runner
- `openquant/gui/scheduler.py` - Robot scheduler

All brokers created during execution are automatically registered with the shutdown handler.

## Troubleshooting

### Shutdown Hangs
- Check if any threads are blocking
- Check logs for errors during shutdown
- Use timeout on subprocess termination

### State Not Saved
- Verify `data/` directory exists and is writable
- Check logs for permission errors
- Ensure sufficient disk space

### Positions Not Closed
- Verify broker connectivity during shutdown
- Check if market is open (for non-24/7 markets)
- Review broker-specific error messages in logs

### Dashboard Process Remains
- The CLI runner terminates dashboard process first
- If zombie process remains, manually kill: `kill <dashboard_pid>`

## Architecture

The shutdown handler uses a centralized registry pattern:
- **Single global instance**: `SHUTDOWN_HANDLER`
- **Component registration**: Brokers and scheduler register themselves
- **Signal handling**: Catches OS signals and triggers shutdown
- **Idempotent**: Multiple shutdown calls are safely ignored
- **Exception handling**: Errors in one step don't block subsequent steps

## Future Enhancements

Potential improvements:
- Email/SMS notifications on shutdown
- Pre-shutdown position hedging
- Configurable timeout per shutdown step
- Remote shutdown API endpoint
- Shutdown metrics and analytics
