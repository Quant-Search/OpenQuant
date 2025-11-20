# Forex Symbol Configuration

FOREX_CONFIG = {
    "EURUSD": {
        "spread_bps": 0.5,  # 0.5 pips = ~0.05 bps for EUR/USD
        "swap_long": -0.5,  # Daily swap cost (pips)
        "swap_short": 0.2,
        "optimal_sessions": ["london", "newyork"],
        "pip_value": 0.0001,
    },
    "GBPUSD": {
        "spread_bps": 0.8,
        "swap_long": -0.7,
        "swap_short": 0.3,
        "optimal_sessions": ["london", "newyork"],
        "pip_value": 0.0001,
    },
    "USDJPY": {
        "spread_bps": 0.6,
        "swap_long": 0.3,
        "swap_short": -0.8,
        "optimal_sessions": ["asian", "london"],
        "pip_value": 0.01,
    },
    "AUDUSD": {
        "spread_bps": 1.0,
        "swap_long": -0.3,
        "swap_short": 0.1,
        "optimal_sessions": ["asian", "newyork"],
        "pip_value": 0.0001,
    },
    "USDCAD": {
        "spread_bps": 1.2,
        "swap_long": -0.4,
        "swap_short": 0.2,
        "optimal_sessions": ["newyork"],
        "pip_value": 0.0001,
    },
    "USDCHF": {
        "spread_bps": 1.5,
        "swap_long": 0.5,
        "swap_short": -1.0,
        "optimal_sessions": ["london"],
        "pip_value": 0.0001,
    },
    "NZDUSD": {
        "spread_bps": 1.5,
        "swap_long": -0.2,
        "swap_short": 0.0,
        "optimal_sessions": ["asian"],
        "pip_value": 0.0001,
    },
    "EURGBP": {
        "spread_bps": 1.0,
        "swap_long": -0.6,
        "swap_short": 0.2,
        "optimal_sessions": ["london"],
        "pip_value": 0.0001,
    },
}

# Trading sessions (UTC)
SESSIONS = {
    "asian": {"start": 0, "end": 9},      # 00:00 - 09:00 UTC
    "london": {"start": 8, "end": 17},    # 08:00 - 17:00 UTC
    "newyork": {"start": 13, "end": 22},  # 13:00 - 22:00 UTC
}

def get_spread_bps(symbol: str) -> float:
    """Get symbol-specific spread in basis points."""
    return FOREX_CONFIG.get(symbol, {}).get("spread_bps", 5.0)

def get_swap_cost(symbol: str, direction: str) -> float:
    """Get daily swap cost in pips."""
    config = FOREX_CONFIG.get(symbol, {})
    if direction == "long":
        return config.get("swap_long", 0.0)
    elif direction == "short":
        return config.get("swap_short", 0.0)
    return 0.0

def is_optimal_session(symbol: str, hour_utc: int) -> bool:
    """Check if current hour is in optimal trading session for symbol."""
    config = FOREX_CONFIG.get(symbol, {})
    optimal = config.get("optimal_sessions", [])
    
    for session_name in optimal:
        session = SESSIONS.get(session_name, {})
        start = session.get("start", 0)
        end = session.get("end", 24)
        if start <= hour_utc < end:
            return True
    return False
