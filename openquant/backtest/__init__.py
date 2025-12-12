"""Backtesting utilities."""

from .engine import (
    BacktestResult,
    backtest_signals,
    auto_backtest,
    summarize_performance,
    sharpe,
    calculate_tod_spread,
    calculate_volume_slippage,
    calculate_market_impact,
    calculate_funding_rate,
    calculate_dynamic_funding_rate,
)

from .cost_models import (
    CostPreset,
    estimate_total_cost,
    compare_presets,
    TOD_MULTIPLIERS_FX_MAJOR,
    TOD_MULTIPLIERS_CRYPTO_MAJOR,
    TOD_MULTIPLIERS_CRYPTO_ALTCOIN,
    TOD_MULTIPLIERS_FLAT,
)

__all__ = [
    "BacktestResult",
    "backtest_signals",
    "auto_backtest",
    "summarize_performance",
    "sharpe",
    "calculate_tod_spread",
    "calculate_volume_slippage",
    "calculate_market_impact",
    "calculate_funding_rate",
    "calculate_dynamic_funding_rate",
    "CostPreset",
    "estimate_total_cost",
    "compare_presets",
    "TOD_MULTIPLIERS_FX_MAJOR",
    "TOD_MULTIPLIERS_CRYPTO_MAJOR",
    "TOD_MULTIPLIERS_CRYPTO_ALTCOIN",
    "TOD_MULTIPLIERS_FLAT",
]
