from __future__ import annotations
"""Paper-trading portfolio state container.

Tracks cash and per-key holdings. Key is (exchange, symbol, timeframe, strategy).
This module is pure data; it does not perform execution itself.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple

Key = Tuple[str, str, str, str]


@dataclass
class PortfolioState:
    cash: float = 100_000.0  # starting paper capital
    holdings: Dict[Key, float] = field(default_factory=dict)  # units (notional weight applied elsewhere)

    def position(self, key: Key) -> float:
        """Return units held for a key (0.0 if none)."""
        return float(self.holdings.get(key, 0.0))

    def set_position(self, key: Key, units: float) -> None:
        """Set units for key (replace)."""
        self.holdings[key] = float(units)

