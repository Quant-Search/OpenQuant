"""Strategy registry to support many strategies with simple names.
Add entries here to expose new strategies to runners.
"""
from __future__ import annotations
from typing import Dict, Callable, Any

from .rule_based.momentum import SMACrossoverStrategy
from .rule_based.ema import EMACrossoverStrategy
from .rule_based.rsi import RSICrossStrategy
from .rule_based.macd import MACDCrossoverStrategy
from .rule_based.bollinger import BollingerMeanReversionStrategy

REGISTRY: Dict[str, Callable[..., Any]] = {
    "sma": SMACrossoverStrategy,
    "ema": EMACrossoverStrategy,
    "rsi": RSICrossStrategy,
    "macd": MACDCrossoverStrategy,
    "bollinger": BollingerMeanReversionStrategy,
}


def make_strategy(name: str, **params):
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(f"Unknown strategy: {name}")
    return REGISTRY[key](**params)

