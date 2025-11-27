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
from typing import Dict, Callable, Any
import inspect

from .quant.kalman import KalmanMeanReversionStrategy
from .quant.hurst import HurstExponentStrategy
from .quant.stat_arb import StatArbStrategy
from .quant.market_micro import LiquidityProvisionStrategy
from .ml_strategy import MLStrategy
from .mixer import StrategyMixer

# Registry of all available quantitative strategies
# Each strategy is a class that implements generate_signals(df) -> pd.Series
REGISTRY: Dict[str, Callable[..., Any]] = {
    "kalman": KalmanMeanReversionStrategy,      # Kalman filter mean reversion
    "hurst": HurstExponentStrategy,             # Hurst exponent regime detection
    "stat_arb": StatArbStrategy,                # Cointegration-based stat arb
    "liquidity": LiquidityProvisionStrategy,    # Market microstructure
    "ml": MLStrategy,                           # Scikit-learn ML strategy
}


def make_strategy(name: str, **params):
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

    # Handle strategy mixer (ensemble of multiple strategies)
    if key == "mixer":
        # Mixer expects 'sub_strategies' list in params
        # e.g. sub_strategies=["kalman", "hurst"], weights=[0.5, 0.5]
        subs = params.get("sub_strategies", [])
        weights = params.get("weights", None)

        instances = []
        for sub_name in subs:
            # Skip unknown strategies
            if sub_name not in REGISTRY:
                continue

            # Get strategy class and its signature
            strat_cls = REGISTRY[sub_name]
            sig = inspect.signature(strat_cls)

            # Filter params: only pass params that the strategy accepts
            valid_params = {
                k: v for k, v in params.items()
                if k in sig.parameters
            }

            # Create strategy instance
            instances.append(make_strategy(sub_name, **valid_params))

        return StrategyMixer(instances, weights)

    # Standard strategy lookup
    if key not in REGISTRY:
        raise KeyError(f"Unknown strategy: {name}. Available: {list(REGISTRY.keys())}")

    return REGISTRY[key](**params)

