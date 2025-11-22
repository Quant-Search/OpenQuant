"""Strategy registry to support many strategies with simple names.
Add entries here to expose new strategies to runners.
"""
from __future__ import annotations
from typing import Dict, Callable, Any

from .quant.kalman import KalmanMeanReversionStrategy
from .quant.hurst import HurstExponentStrategy
from .quant.stat_arb import StatArbStrategy
from .quant.market_micro import LiquidityProvisionStrategy
from .ml_strategy import MLStrategy

REGISTRY: Dict[str, Callable[..., Any]] = {
    "kalman": KalmanMeanReversionStrategy,
    "hurst": HurstExponentStrategy,
    "stat_arb": StatArbStrategy,
    "liquidity": LiquidityProvisionStrategy,
    "ml": MLStrategy,
}


from .wrappers.pandas_ta import make_pta_strategy

from .mixer import StrategyMixer

def make_strategy(name: str, **params):
    key = name.lower()
    if key.startswith("pta_"):
        return make_pta_strategy(key, **params)
    
    if key == "mixer":
        # Mixer expects 'sub_strategies' list in params
        # e.g. sub_strategies=["kalman", "hurst"], weights=[0.5, 0.5]
        subs = params.get("sub_strategies", [])
        weights = params.get("weights", None)
        
        # Instantiate sub-strategies
        # We need to know which params belong to which strategy.
        # For simplicity, we pass the FULL params dict to each sub-strategy,
        # assuming they will ignore unknown params or we prefix them?
        # Better approach: sub_strategies is a list of dicts? 
        # Or we just use default params for subs if not specified?
        # Let's assume simple case: use default params for subs, or shared params.
        
        # Smart filtering: only pass params that the sub-strategy accepts
        import inspect
        
        instances = []
        for sub_name in subs:
            if sub_name not in REGISTRY:
                continue
                
            strat_cls = REGISTRY[sub_name]
            sig = inspect.signature(strat_cls)
            
            # Filter params for this specific strategy
            valid_params = {}
            for k, v in params.items():
                if k in sig.parameters:
                    valid_params[k] = v
            
            # Recursive call (but we call the class directly or make_strategy?)
            # Calling make_strategy is safer for wrappers like pta_
            # But make_strategy doesn't take a class, it takes a name.
            # So we can't easily inspect make_strategy target without resolving it first.
            # But we verified sub_name is in REGISTRY above.
            # Wait, make_strategy handles pta_ too.
            
            if sub_name.startswith("pta_"):
                 # PTA strategies usually accept **kwargs or specific params.
                 # For now, pass all sub_params to PTA
                 instances.append(make_strategy(sub_name, **sub_params))
            else:
                 instances.append(make_strategy(sub_name, **valid_params))
            
        return StrategyMixer(instances, weights)

    if key not in REGISTRY:
        raise KeyError(f"Unknown strategy: {name}")
    return REGISTRY[key](**params)

