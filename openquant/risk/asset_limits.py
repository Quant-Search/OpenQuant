"""
Per-Asset Risk Limits Module.

Provides configurable position size limits per symbol/asset class:
- Maximum position size (notional value)
- Maximum position size (percentage of portfolio)
- Maximum number of positions per asset class
- Concentration limits

Mathematical basis:
- Position size = quantity * price
- Concentration = position_size / total_portfolio_value
- Asset class exposure = sum of positions in asset class / total_portfolio_value

The limits are loaded from a JSON configuration file for flexibility.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AssetLimit:
    """Limit configuration for a single asset or asset class."""
    max_notional: float = float('inf')  # Maximum notional value in base currency
    max_pct_portfolio: float = 1.0  # Maximum as percentage of portfolio (0.2 = 20%)
    max_positions: int = 100  # Maximum number of positions in this asset/class
    min_notional: float = 0.0  # Minimum position size (avoid dust positions)


@dataclass
class AssetLimitsConfig:
    """Configuration for all asset limits."""
    # Default limits for any asset not specifically configured
    default: AssetLimit = field(default_factory=lambda: AssetLimit(
        max_pct_portfolio=0.20,  # 20% max per asset by default
        max_positions=10
    ))

    # Per-symbol overrides (e.g., {"BTCUSD": AssetLimit(max_pct_portfolio=0.10)})
    symbols: Dict[str, AssetLimit] = field(default_factory=dict)

    # Per-asset-class limits (e.g., {"forex": AssetLimit(max_pct_portfolio=0.50)})
    asset_classes: Dict[str, AssetLimit] = field(default_factory=dict)

    # Global limits
    max_total_positions: int = 50  # Maximum total positions across all assets
    max_leverage: float = 1.0  # Maximum portfolio leverage (1.0 = no leverage)


class AssetLimitsManager:
    """
    Manages per-asset position limits and validates proposed trades.

    Usage:
        manager = AssetLimitsManager()

        # Check if a trade is allowed
        allowed, reason = manager.check_trade(
            symbol="EURUSD",
            quantity=10000,
            price=1.10,
            portfolio_value=100000,
            current_positions={"EURUSD": 5000}
        )

        if not allowed:
            print(f"Trade rejected: {reason}")
    """

    def __init__(self, config_file: str = "data/asset_limits.json"):
        """Initialize with configuration file path."""
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> AssetLimitsConfig:
        """Load configuration from file or use defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return self._parse_config(data)
            except Exception as e:
                LOGGER.warning(f"Failed to load asset limits config: {e}")
        return AssetLimitsConfig()

    def _parse_config(self, data: Dict[str, Any]) -> AssetLimitsConfig:
        """Parse JSON config into AssetLimitsConfig."""
        config = AssetLimitsConfig()

        # Parse default limits
        if "default" in data:
            config.default = AssetLimit(**data["default"])

        # Parse symbol-specific limits
        if "symbols" in data:
            for sym, limits in data["symbols"].items():
                config.symbols[sym] = AssetLimit(**limits)

        # Parse asset class limits
        if "asset_classes" in data:
            for cls, limits in data["asset_classes"].items():
                config.asset_classes[cls] = AssetLimit(**limits)

        # Parse global limits
        if "max_total_positions" in data:
            config.max_total_positions = int(data["max_total_positions"])
        if "max_leverage" in data:
            config.max_leverage = float(data["max_leverage"])

        return config

    def save_config(self) -> None:
        """Save current configuration to file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "default": {
                "max_notional": self.config.default.max_notional,
                "max_pct_portfolio": self.config.default.max_pct_portfolio,
                "max_positions": self.config.default.max_positions,
                "min_notional": self.config.default.min_notional,
            },
            "symbols": {
                sym: {
                    "max_notional": lim.max_notional,
                    "max_pct_portfolio": lim.max_pct_portfolio,
                    "max_positions": lim.max_positions,
                    "min_notional": lim.min_notional,
                }
                for sym, lim in self.config.symbols.items()
            },
            "asset_classes": {
                cls: {
                    "max_notional": lim.max_notional,
                    "max_pct_portfolio": lim.max_pct_portfolio,
                    "max_positions": lim.max_positions,
                    "min_notional": lim.min_notional,
                }
                for cls, lim in self.config.asset_classes.items()
            },
            "max_total_positions": self.config.max_total_positions,
            "max_leverage": self.config.max_leverage,
        }

        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_limit(self, symbol: str, asset_class: Optional[str] = None) -> AssetLimit:
        """Get the applicable limit for a symbol.

        Priority: symbol-specific > asset-class > default
        """
        # Check symbol-specific first
        if symbol in self.config.symbols:
            return self.config.symbols[symbol]

        # Check asset class
        if asset_class and asset_class in self.config.asset_classes:
            return self.config.asset_classes[asset_class]

        # Return default
        return self.config.default

    def check_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
        asset_class: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Check if a proposed trade is within limits.

        Args:
            symbol: The symbol to trade
            quantity: Proposed quantity (positive for buy, negative for sell)
            price: Current price
            portfolio_value: Total portfolio value
            current_positions: Dict of {symbol: current_notional_value}
            asset_class: Optional asset class for the symbol

        Returns:
            (allowed, reason) tuple
        """
        # Get applicable limit
        limit = self.get_limit(symbol, asset_class)

        # Calculate proposed position
        trade_notional = abs(quantity * price)
        current_notional = current_positions.get(symbol, 0.0)

        # For buys, add to position; for sells, subtract
        if quantity > 0:
            new_notional = current_notional + trade_notional
        else:
            new_notional = max(0, current_notional - trade_notional)

        # Check minimum position size (avoid dust)
        if new_notional > 0 and new_notional < limit.min_notional:
            return False, f"Position {new_notional:.2f} below minimum {limit.min_notional:.2f}"

        # Check maximum notional
        if new_notional > limit.max_notional:
            return False, f"Position {new_notional:.2f} exceeds max notional {limit.max_notional:.2f}"

        # Check percentage of portfolio
        if portfolio_value > 0:
            new_pct = new_notional / portfolio_value
            if new_pct > limit.max_pct_portfolio:
                return False, f"Position {new_pct:.1%} exceeds max {limit.max_pct_portfolio:.1%} of portfolio"

        # Check total positions count
        total_positions = len([v for v in current_positions.values() if v > 0])
        if symbol not in current_positions and total_positions >= self.config.max_total_positions:
            return False, f"Total positions {total_positions} at maximum {self.config.max_total_positions}"

        # Check leverage
        total_exposure = sum(current_positions.values()) + trade_notional
        if portfolio_value > 0:
            leverage = total_exposure / portfolio_value
            if leverage > self.config.max_leverage:
                return False, f"Leverage {leverage:.2f}x exceeds max {self.config.max_leverage:.2f}x"

        return True, "OK"

    def get_max_quantity(
        self,
        symbol: str,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
        asset_class: Optional[str] = None,
    ) -> float:
        """Calculate maximum allowed quantity for a symbol.

        Returns the maximum quantity that can be bought within limits.
        """
        limit = self.get_limit(symbol, asset_class)
        current_notional = current_positions.get(symbol, 0.0)

        # Calculate remaining room under each limit
        remaining_notional = limit.max_notional - current_notional
        remaining_pct = (limit.max_pct_portfolio * portfolio_value) - current_notional

        # Calculate remaining leverage room
        total_exposure = sum(current_positions.values())
        remaining_leverage = (self.config.max_leverage * portfolio_value) - total_exposure

        # Take the minimum of all limits
        max_notional = min(remaining_notional, remaining_pct, remaining_leverage)
        max_notional = max(0, max_notional)

        # Convert to quantity
        if price > 0:
            return max_notional / price
        return 0.0


# Global instance with default configuration
ASSET_LIMITS = AssetLimitsManager()
