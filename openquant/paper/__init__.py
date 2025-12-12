"""Paper trading simulator module."""
from .state import PortfolioState, Key
from .simulator import (
    MarketSnapshot,
    compute_target_units,
    compute_target_units_with_kelly,
    compute_rebalance_orders,
    compute_rebalance_orders_with_kelly,
    execute_orders,
    check_exits,
    check_daily_loss,
    rebalance_to_targets,
    rebalance_to_targets_with_kelly,
    record_closed_trades,
)

__all__ = [
    "PortfolioState",
    "Key",
    "MarketSnapshot",
    "compute_target_units",
    "compute_target_units_with_kelly",
    "compute_rebalance_orders",
    "compute_rebalance_orders_with_kelly",
    "execute_orders",
    "check_exits",
    "check_daily_loss",
    "rebalance_to_targets",
    "rebalance_to_targets_with_kelly",
    "record_closed_trades",
]
