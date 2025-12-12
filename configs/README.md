# Configuration Files

This directory contains YAML configuration files for OpenQuant.

## Available Configurations

- **default.yaml** - Default configuration with balanced settings
- **production.yaml** - Conservative settings for live trading
- **aggressive.yaml** - Higher risk settings for aggressive trading

## Using Configuration Files

### Method 1: Environment Variable

Set the `OPENQUANT_CONFIG` environment variable to specify which config to use:

```bash
export OPENQUANT_CONFIG=configs/production.yaml
python scripts/run_robot_cli.py
```

### Method 2: Programmatic Loading

```python
from openquant.config import ConfigManager

# Load specific config
config = ConfigManager.from_yaml("configs/production.yaml")

# Access values
dd_limit = config.get("risk_limits.dd_limit")
print(f"Max drawdown limit: {dd_limit}")

# Get entire section
risk_config = config.get_section("risk_limits")
print(f"Daily loss cap: {risk_config.daily_loss_cap}")

# Update values
config.set("risk_limits.dd_limit", 0.18)

# Save modified config
config.to_yaml("configs/custom.yaml")
```

### Method 3: Global Configuration

```python
from openquant.config import get_config

# Get global config (loads from OPENQUANT_CONFIG env var or uses default)
config = get_config()

# Use throughout application
dd_limit = config.get("risk_limits.dd_limit")
```

## Configuration Sections

### forex
Forex-specific settings including symbol spreads, swaps, and trading sessions.

### risk_limits
Portfolio-level risk management limits:
- `dd_limit`: Maximum drawdown (0.20 = 20%)
- `daily_loss_cap`: Daily loss limit (0.05 = 5%)
- `cvar_limit`: Conditional Value at Risk limit
- `max_exposure_per_symbol`: Maximum exposure per symbol

### circuit_breaker
Automatic trading halt thresholds:
- `daily_loss_limit`: Daily loss threshold to halt trading
- `drawdown_limit`: Drawdown threshold to halt trading
- `volatility_limit`: Volatility spike threshold

### backtest
Backtest engine parameters:
- `fee_bps`: Trading fees in basis points
- `slippage_bps`: Slippage in basis points
- `spread_bps`: Bid-ask spread
- `leverage`: Leverage multiplier
- `weight`: Capital allocation fraction

### strategy_mixer
Strategy ensemble parameters:
- `threshold`: Signal threshold for long/short decisions
- `equal_weights`: Whether to use equal weights for strategies

### adaptive_sizing
Position sizing configuration:
- `method`: Sizing method (kelly or volatility)
- `target_risk`: Target risk per trade
- `max_drawdown`: Maximum drawdown threshold
- `aggressive_mode`: Enable aggressive sizing
- `target_volatility`: Target annualized volatility
- `max_leverage`: Maximum leverage

### stationarity
Statistical testing thresholds:
- `adf_threshold`: ADF test p-value threshold
- `kpss_threshold`: KPSS test p-value threshold
- `hurst_mean_reverting`: Hurst exponent threshold for mean-reverting
- `hurst_trending`: Hurst exponent threshold for trending

### concentration_limits
Portfolio concentration constraints:
- `max_per_symbol`: Maximum configurations per symbol
- `max_per_strategy_per_symbol`: Maximum configurations per strategy-symbol pair

### paper_trading
Paper trading simulator settings:
- `fee_bps`: Trading fees
- `slippage_bps`: Slippage
- `next_bar_fill`: Fill orders at next bar open
- `max_fill_fraction`: Maximum fill fraction
- `daily_loss_limit`: Daily loss limit

## Creating Custom Configurations

1. Copy an existing config file:
   ```bash
   cp configs/default.yaml configs/custom.yaml
   ```

2. Edit the values as needed

3. Load it in your application:
   ```python
   config = ConfigManager.from_yaml("configs/custom.yaml")
   ```

## Validation

All configurations are validated using Pydantic schemas. Invalid values will raise a `ValidationError` with details about what's wrong.
