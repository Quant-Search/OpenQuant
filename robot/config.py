"""
Configuration Management

Single Responsibility: Only handles configuration loading and validation.
"""
import os
from typing import Optional, List


def _safe_int(value: Optional[str]) -> Optional[int]:
    """Convert string to int safely, return None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


class Config:
    """Robot configuration - all settings in one place."""
    
    # Trading symbols (MT5 format)
    SYMBOLS: List[str] = ["EURUSD", "GBPUSD", "USDJPY"]
    
    # Timeframe for analysis (1h, 4h, 1d)
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
    
    # Loop settings
    LOOP_INTERVAL_SECONDS: int = 3600  # Run every hour
    
    # MT5 credentials (from environment)
    MT5_LOGIN: Optional[int] = _safe_int(os.getenv("MT5_LOGIN"))
    MT5_PASSWORD: Optional[str] = os.getenv("MT5_PASSWORD")
    MT5_SERVER: Optional[str] = os.getenv("MT5_SERVER")
    MT5_TERMINAL_PATH: Optional[str] = os.getenv("MT5_TERMINAL_PATH")


