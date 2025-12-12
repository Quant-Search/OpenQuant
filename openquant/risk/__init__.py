"""Risk management modules for OpenQuant."""
from .kelly_criterion import (
    KellyCriterion,
    compute_rolling_volatility,
    estimate_win_rate_from_signals,
    TradeRecord,
    KellyStats,
)
from .trade_validator import TradeValidator, TradeValidationResult, TRADE_VALIDATOR
from .asset_limits import AssetLimitsManager, ASSET_LIMITS
from .kill_switch import KillSwitch, KILL_SWITCH
from .circuit_breaker import CircuitBreaker, CIRCUIT_BREAKER
from .forex_correlation import get_correlation, check_portfolio_correlation

__all__ = [
    "KellyCriterion",
    "compute_rolling_volatility",
    "estimate_win_rate_from_signals",
    "TradeRecord",
    "KellyStats",
    "TradeValidator",
    "TradeValidationResult",
    "TRADE_VALIDATOR",
    "AssetLimitsManager",
    "ASSET_LIMITS",
    "KillSwitch",
    "KILL_SWITCH",
    "CircuitBreaker",
    "CIRCUIT_BREAKER",
    "get_correlation",
    "check_portfolio_correlation",
]
