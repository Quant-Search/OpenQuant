# Forex Symbol Configuration
# DEPRECATED: Use ConfigManager instead
# This module is kept for backward compatibility

from typing import Optional
import warnings
from .manager import get_config

warnings.warn(
    "openquant.config.forex is deprecated. Use ConfigManager instead.",
    DeprecationWarning,
    stacklevel=2
)


def get_spread_bps(symbol: str) -> float:
    """Get symbol-specific spread in basis points.
    
    DEPRECATED: Use config.get(f"forex.symbols.{symbol}.spread_bps") instead.
    """
    config = get_config()
    forex_config = config.get_section("forex")
    if forex_config and symbol in forex_config.symbols:
        return forex_config.symbols[symbol].spread_bps
    return 5.0


def get_swap_cost(symbol: str, direction: str) -> float:
    """Get daily swap cost in pips.
    
    DEPRECATED: Use config.get(f"forex.symbols.{symbol}.swap_{direction}") instead.
    """
    config = get_config()
    forex_config = config.get_section("forex")
    if forex_config and symbol in forex_config.symbols:
        symbol_config = forex_config.symbols[symbol]
        if direction == "long":
            return symbol_config.swap_long
        elif direction == "short":
            return symbol_config.swap_short
    return 0.0


def is_optimal_session(symbol: str, hour_utc: int) -> bool:
    """Check if current hour is in optimal trading session for symbol.
    
    DEPRECATED: Use ConfigManager for session checks.
    """
    config = get_config()
    forex_config = config.get_section("forex")
    if not forex_config or symbol not in forex_config.symbols:
        return False
    
    symbol_config = forex_config.symbols[symbol]
    optimal_sessions = symbol_config.optimal_sessions
    
    for session_name in optimal_sessions:
        if session_name in forex_config.sessions:
            session = forex_config.sessions[session_name]
            if session.start <= hour_utc < session.end:
                return True
    return False


# Backward compatibility - legacy constants
FOREX_CONFIG = {}
SESSIONS = {}

def _load_legacy_constants():
    """Load legacy constants from ConfigManager for backward compatibility."""
    global FOREX_CONFIG, SESSIONS
    config = get_config()
    forex_config = config.get_section("forex")
    
    if forex_config:
        FOREX_CONFIG = {
            symbol: {
                "spread_bps": cfg.spread_bps,
                "swap_long": cfg.swap_long,
                "swap_short": cfg.swap_short,
                "optimal_sessions": cfg.optimal_sessions,
                "pip_value": cfg.pip_value,
            }
            for symbol, cfg in forex_config.symbols.items()
        }
        
        SESSIONS = {
            name: {
                "start": session.start,
                "end": session.end,
            }
            for name, session in forex_config.sessions.items()
        }

_load_legacy_constants()
