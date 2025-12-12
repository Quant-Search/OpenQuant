# Risk Management System

Comprehensive risk management framework for OpenQuant, featuring trade-level validation and portfolio protection mechanisms.

## Overview

The risk management system integrates multiple layers of protection:

1. **Trade Validator** - Central pre-trade validation system
2. **Asset Limits** - Position size and concentration controls
3. **Kill Switch** - Emergency stop mechanism
4. **Circuit Breaker** - Automatic halt on losses/drawdown
5. **Correlation Constraints** - Prevent over-concentration in correlated assets

## Trade Validator

The `TradeValidator` class provides comprehensive pre-trade risk checks that integrate all risk management components.

### Features

- **Kill Switch Check**: Emergency stop mechanism (file-based trigger)
- **Circuit Breaker Check**: Automatic halt on daily loss or drawdown limits
- **Asset Limits**: Position size, concentration, and leverage constraints
- **Correlation Check**: Warns about highly correlated positions (threshold: 0.8)
- **Batch Validation**: Validate multiple orders with simulated position updates

### Usage

```python
from openquant.risk import TRADE_VALIDATOR

# Basic validation
result = TRADE_VALIDATOR.validate_trade(
    symbol="EURUSD",
    quantity=10000,
    price=1.10,
    side="buy",
    portfolio_value=100000,
    current_positions={"EURUSD": 5000, "GBPUSD": 3000},
    current_equity=100000
)

if result.allowed:
    # Execute trade
    print(f"Trade approved: {result.reason}")
else:
    # Reject trade
    print(f"Trade rejected: {result.reason}")
    print(f"Failed checks: {result.failed_checks}")
```

### Integration Points

The trade validator is automatically called before order execution in:

- **AlpacaBroker.place_order()** - Alpaca live/paper trading
- **MT5Broker.place_order()** - MetaTrader 5 live trading
- **mt5_bridge.apply_allocation()** - MT5 allocation mirroring
- **simulator.execute_orders()** - Paper trading simulator

This ensures all trades pass risk checks regardless of execution path.

## Asset Limits

Position sizing and concentration controls.

### Configuration

```python
from openquant.risk import ASSET_LIMITS

# Configure default limits
ASSET_LIMITS.config.default.max_pct_portfolio = 0.20  # 20% max per asset
ASSET_LIMITS.config.default.max_notional = 50000.0     # $50k max position
ASSET_LIMITS.config.max_total_positions = 10           # Max 10 positions
ASSET_LIMITS.config.max_leverage = 1.0                 # No leverage

# Symbol-specific overrides
ASSET_LIMITS.config.symbols["BTCUSD"] = AssetLimit(
    max_pct_portfolio=0.10,  # 10% max for BTC
    max_notional=25000.0
)

# Save configuration
ASSET_LIMITS.save_config()
```

### Validation

```python
allowed, reason = ASSET_LIMITS.check_trade(
    symbol="EURUSD",
    quantity=10000,
    price=1.10,
    portfolio_value=100000,
    current_positions={"EURUSD": 5000}
)
```

## Kill Switch

Emergency stop mechanism to halt all trading.

### Activation

```python
from openquant.risk import KILL_SWITCH

# Activate (creates data/STOP file)
KILL_SWITCH.activate()

# Check status
if KILL_SWITCH.is_active():
    print("Trading halted")

# Deactivate (removes data/STOP file)
KILL_SWITCH.deactivate()
```

### Manual Activation

Create a file named `data/STOP` to immediately halt all trading:

```bash
touch data/STOP
```

## Circuit Breaker

Automatic trading halt based on losses and drawdown.

### Configuration

```python
from openquant.risk import CIRCUIT_BREAKER

# Configure thresholds
CIRCUIT_BREAKER.daily_loss_limit = 0.02   # 2% daily loss
CIRCUIT_BREAKER.drawdown_limit = 0.10     # 10% max drawdown
CIRCUIT_BREAKER.volatility_limit = 0.05   # 5% volatility spike

# Update with current equity
CIRCUIT_BREAKER.update(current_equity=95000.0)

# Check status
if CIRCUIT_BREAKER.is_tripped():
    status = CIRCUIT_BREAKER.get_status()
    print(f"Breaker tripped: {status}")
    
# Reset breaker
CIRCUIT_BREAKER.reset()
```

### State Persistence

Circuit breaker state is persisted to `data/circuit_breaker_state.json` and survives restarts.

## Correlation Constraints

Prevent over-concentration in correlated forex pairs.

### Correlation Matrix

Built-in correlation data for major forex pairs (source: typical 1-year correlations):

- EUR/USD ↔ GBP/USD: 0.85 (highly correlated)
- EUR/USD ↔ USD/CHF: -0.95 (highly inverse)
- AUD/USD ↔ NZD/USD: 0.90 (highly correlated)

### Usage

```python
from openquant.risk.forex_correlation import get_correlation, check_portfolio_correlation

# Get correlation between two pairs
corr = get_correlation("EURUSD", "GBPUSD")  # Returns 0.85

# Check if candidate is highly correlated with holdings
current_holdings = ["EURUSD", "AUDUSD"]
is_correlated = check_portfolio_correlation(
    candidate_symbol="GBPUSD",
    current_holdings=current_holdings,
    threshold=0.8
)
```

### Validation Behavior

The trade validator **warns** about high correlation (≥0.8) but does not block trades by default. To enable blocking, uncomment the rejection logic in `trade_validator.py`:

```python
# In TradeValidator.validate_trade(), line ~196
if check_portfolio_correlation(symbol, current_holdings, self.correlation_threshold):
    # Currently logs warning only
    # To block: uncomment the following
    # failed_checks.append("correlation")
    # return TradeValidationResult(allowed=False, ...)
```

## Batch Validation

Validate multiple orders with simulated position tracking:

```python
orders = [
    ("EURUSD", 1.0, 10000.0, "buy"),
    ("GBPUSD", 1.0, 8000.0, "buy"),
    ("USDJPY", 1.0, 7000.0, "buy"),
]

allowed, rejected = TRADE_VALIDATOR.validate_order_batch(
    orders=orders,
    portfolio_value=100000.0,
    current_positions={},
)

# Execute allowed orders
for symbol, qty, price, side in allowed:
    broker.place_order(symbol, qty, side, price=price)

# Log rejected orders
for symbol, reason in rejected:
    logger.warning(f"Order rejected: {symbol} - {reason}")
```

## Validation Status

Get comprehensive status of all risk validators:

```python
status = TRADE_VALIDATOR.get_validation_status()

print(f"Kill Switch: {status['kill_switch_active']}")
print(f"Circuit Breaker: {status['circuit_breaker']['is_tripped']}")
print(f"Max Positions: {status['asset_limits']['max_total_positions']}")
print(f"Max Leverage: {status['asset_limits']['max_leverage']}x")
```

## Example Script

Run the comprehensive example:

```bash
python scripts/trade_validator_example.py
```

This demonstrates:
- Basic validation
- Kill switch activation
- Circuit breaker triggering
- Asset limit enforcement
- Correlation warnings
- Batch validation
- Status monitoring

## Architecture

```
TradeValidator (trade_validator.py)
├── KillSwitch (kill_switch.py)
├── CircuitBreaker (circuit_breaker.py)
├── AssetLimitsManager (asset_limits.py)
└── forex_correlation (forex_correlation.py)
```

All components can be used independently or through the unified `TradeValidator` interface.

## Configuration Files

- `data/asset_limits.json` - Asset limit configuration
- `data/circuit_breaker_state.json` - Circuit breaker state
- `data/STOP` - Kill switch trigger file

## Best Practices

1. **Always use trade validator** - Integrate into all order execution paths
2. **Monitor circuit breaker** - Check daily for trips and investigate causes
3. **Configure asset limits** - Set appropriate limits for your strategy and risk tolerance
4. **Test kill switch** - Periodically verify emergency stop works
5. **Review correlation warnings** - Adjust portfolio when correlation alerts fire
6. **Persist configuration** - Save asset limits after tuning

## Testing

```bash
pytest tests/test_trade_validator.py -v
```

## See Also

- `scripts/trade_validator_example.py` - Complete usage examples
- `AGENTS.md` - Architecture overview
- `SECURITY.md` - Risk management policies
