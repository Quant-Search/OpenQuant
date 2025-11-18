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

    @property
    def positions_value(self) -> float:
        """Total value of all positions (requires external pricing, but here we approximate or track separately).
        For now, we don't have live prices in state, so we might need to pass them in or store last known value.
        
        However, the scheduler tries to access this. Let's add a placeholder or a way to update it.
        Actually, the scheduler calculates equity using `state.cash + state.positions_value`.
        If we don't store prices, we can't know the value.
        
        Fix: We should probably store `equity` or `last_known_value` in state, or compute it using a price source.
        For simplicity in this phase, let's add a field `last_positions_value` that gets updated.
        """
        return self._last_positions_value

    _last_positions_value: float = 0.0
    
    def update_valuation(self, value: float):
        self._last_positions_value = value

    def position(self, key: Key) -> float:
        """Return units held for a key (0.0 if none)."""
        return float(self.holdings.get(key, 0.0))

    def set_position(self, key: Key, units: float) -> None:
        """Set units for key (replace)."""
        self.holdings[key] = float(units)

