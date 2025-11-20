"""Strategy registry to support many strategies with simple names.
Add entries here to expose new strategies to runners.
"""
from __future__ import annotations
from typing import Dict, Callable, Any

from .quant.kalman import KalmanMeanReversionStrategy
from .quant.hurst import HurstExponentStrategy
from .quant.stat_arb import StatArbStrategy
from .quant.market_micro import LiquidityProvisionStrategy

REGISTRY: Dict[str, Callable[..., Any]] = {
    "kalman": KalmanMeanReversionStrategy,
    "hurst": HurstExponentStrategy,
    "stat_arb": StatArbStrategy,
    "liquidity": LiquidityProvisionStrategy,
}


from .wrappers.pandas_ta import make_pta_strategy

def make_strategy(name: str, **params):
    key = name.lower()
    if key.startswith("pta_"):
        return make_pta_strategy(key, **params)
        
    if key not in REGISTRY:
        raise KeyError(f"Unknown strategy: {name}")
    return REGISTRY[key](**params)

