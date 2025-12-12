"""Risk management modules for OpenQuant."""
from .kelly_criterion import (
    KellyCriterion,
    compute_rolling_volatility,
    estimate_win_rate_from_signals,
    TradeRecord,
    KellyStats,
)

__all__ = [
    "KellyCriterion",
    "compute_rolling_volatility",
    "estimate_win_rate_from_signals",
    "TradeRecord",
    "KellyStats",
]
