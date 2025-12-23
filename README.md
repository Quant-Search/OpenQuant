# OpenQuant MVP

## What is this?

A simple, working trading robot that:
1. Connects to MetaTrader 5 (or runs in paper mode)
2. Fetches price data for configured symbols
3. Generates trading signals using Kalman Filter Mean Reversion
4. Executes trades with basic risk management (position sizing, stop loss)

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure (for live trading)

Create a `.env` file with your MT5 credentials:

```env
MT5_LOGIN=12345678
MT5_PASSWORD=your_password_here
MT5_SERVER=YourBroker-Server
MT5_TERMINAL_PATH=C:/Program Files/MetaTrader 5/terminal64.exe
```

### 3. Run

```bash
# Paper trading (simulated, no real money)
python main.py --mode paper

# Backtest the strategy on historical data
python main.py --mode backtest

# Live trading (requires MT5 credentials)
python main.py --mode live

# Or use the batch file on Windows
run_robot.bat          # Paper mode (default)
run_robot.bat live     # Live mode
run_robot.bat backtest # Backtest mode
```

## Configuration

Edit the `Config` class in `mvp_robot.py`:

```python
class Config:
    # Trading symbols (MT5 format)
    SYMBOLS: List[str] = ["EURUSD", "GBPUSD", "USDJPY"]
    
    # Timeframe for analysis
    TIMEFRAME: str = "1h"
    
    # Strategy parameters (Kalman Filter)
    PROCESS_NOISE: float = 1e-5       # How much true price varies
    MEASUREMENT_NOISE: float = 1e-3   # How noisy are observations
    SIGNAL_THRESHOLD: float = 1.5     # Z-score threshold for signals
    
    # Risk management
    RISK_PER_TRADE: float = 0.02      # Risk 2% of equity per trade
    MAX_POSITIONS: int = 3            # Maximum concurrent positions
    STOP_LOSS_ATR_MULT: float = 2.0   # Stop loss = 2x ATR
    TAKE_PROFIT_ATR_MULT: float = 3.0 # Take profit = 3x ATR
```

## How the Strategy Works

The robot uses a **Kalman Filter Mean Reversion** strategy:

1. **Kalman Filter** estimates the "true" price from noisy market data
2. **Deviation** is calculated: observed price - estimated price
3. **Z-score** normalizes the deviation using rolling standard deviation
4. **Signals**:
   - LONG when z-score < -threshold (price below fair value)
   - SHORT when z-score > threshold (price above fair value)

### Mathematical Model

```
State equation:     x(t+1) = x(t) + w(t),  where w(t) ~ N(0, Q)
Observation:        z(t) = x(t) + v(t),    where v(t) ~ N(0, R)

Kalman Update:
  Predict: x_pred = x_prev, P_pred = P_prev + Q
  Update:  K = P_pred / (P_pred + R)
           x = x_pred + K * (z - x_pred)
           P = (1 - K) * P_pred
```

## Risk Management

- **Position Sizing**: Risk X% of equity per trade (default: 2%)
- **Stop Loss**: ATR-based (default: 2x ATR from entry)
- **Take Profit**: ATR-based (default: 3x ATR from entry)
- **Max Positions**: Limit concurrent positions (default: 3)

## Project Structure

```
OpenQuant/
  main.py            # Entry point (start here!)
  robot/             # Modular robot package
    config.py        # Configuration
    strategy.py      # Strategy interface + Kalman implementation
    data_fetcher.py  # Data fetching (MT5/yfinance)
    risk_manager.py  # Risk calculations (ATR, position sizing)
    trader.py        # Trade execution (paper/live)
    robot.py         # Main orchestrator
  run_robot.bat      # Windows launcher
  requirements.txt   # Python dependencies
  .env               # Your credentials (create this)
  ARCHITECTURE.md    # Design documentation
```

**Modular Design**: Follows SOLID principles - each module has a single responsibility.

## Requirements

- Python 3.10+
- Windows (for MT5) or any OS for paper/backtest mode
- MetaTrader 5 terminal (for live trading)

## License

MIT
