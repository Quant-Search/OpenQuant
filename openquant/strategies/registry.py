"""Strategy registry to support many strategies with simple names.

This registry contains ONLY quantitative/mathematical strategies:
- Kalman Filter based mean reversion
- Hurst Exponent for regime detection
- Statistical Arbitrage (cointegration-based)
- Liquidity/Microstructure strategies
- Machine Learning strategies

NO retail indicators (SMA, EMA, RSI, MACD, Bollinger, etc.)
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from .base import BaseStrategy
from .mixer import StrategyMixer
from .ml_strategy import MLStrategy
from .regime_adaptive import RegimeAdaptiveStrategy
from .quant.hurst import HurstExponentStrategy
from .quant.kalman import KalmanMeanReversionStrategy
from .quant.market_micro import LiquidityProvisionStrategy
from .quant.stat_arb import StatArbStrategy

REGISTRY: dict[str, Callable[..., BaseStrategy]] = {
    "kalman": KalmanMeanReversionStrategy,
    "hurst": HurstExponentStrategy,
    "stat_arb": StatArbStrategy,
    "liquidity": LiquidityProvisionStrategy,
    "ml": MLStrategy,
    "regime_adaptive": RegimeAdaptiveStrategy,
}


def make_strategy(name: str, **params: Any) -> BaseStrategy:
    """Factory function to create strategy instances by name.

    Args:
        name: Strategy name (must be in REGISTRY or 'mixer')
        **params: Strategy-specific parameters

    Returns:
        Strategy instance with generate_signals() method

    Raises:
        KeyError: If strategy name is not found in registry
    """
    key = name.lower()

    if key == "mixer":
        subs: list[str] = params.get("sub_strategies", [])
        weights: list[float] | None = params.get("weights", None)

        instances: list[BaseStrategy] = []
        for sub_name in subs:
            if sub_name not in REGISTRY:
                continue

            strat_cls = REGISTRY[sub_name]
            sig = inspect.signature(strat_cls)

            valid_params: dict[str, Any] = {
                k: v for k, v in params.items()
                if k in sig.parameters
            }

            instances.append(make_strategy(sub_name, **valid_params))

        return StrategyMixer(instances, weights)

    if key not in REGISTRY:
        raise KeyError(f"Unknown strategy: {name}. Available: {list(REGISTRY.keys())}")

    return REGISTRY[key](**params)


def get_strategy(name: str, **params):
    """Alias for make_strategy for backward compatibility."""
    return make_strategy(name, **params)
