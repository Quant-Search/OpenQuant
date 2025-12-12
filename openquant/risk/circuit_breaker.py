"""
Circuit Breaker Module.

Provides automatic trading halt mechanisms based on risk thresholds:
- Daily loss limit: Halt if daily P&L drops below threshold
- Drawdown limit: Halt if portfolio drawdown exceeds threshold
- Volatility spike: Halt if market volatility exceeds threshold

Mathematical basis:
- Daily loss = (current_equity - start_of_day_equity) / start_of_day_equity
- Drawdown = (peak_equity - current_equity) / peak_equity
- Volatility = standard deviation of returns over lookback period

The circuit breaker persists state to a JSON file so it survives restarts.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict

from openquant.utils.logging import get_logger
from openquant.utils.validation import (
    validate_params,
    validate_range_param
)

LOGGER = get_logger(__name__)


@dataclass
class CircuitBreakerState:
    """Persistent state for circuit breaker tracking."""
    # Daily tracking
    current_date: str = ""  # YYYY-MM-DD format
    start_of_day_equity: float = 0.0  # Equity at start of trading day

    # Peak tracking for drawdown
    peak_equity: float = 0.0  # Highest equity ever recorded

    # Breaker status
    daily_loss_breaker_tripped: bool = False  # True if daily loss limit hit
    drawdown_breaker_tripped: bool = False  # True if drawdown limit hit
    volatility_breaker_tripped: bool = False  # True if volatility spike detected

    # Timestamps
    last_trip_time: str = ""  # ISO format timestamp of last trip
    last_reset_time: str = ""  # ISO format timestamp of last reset


class CircuitBreaker:
    """
    Monitors portfolio metrics and trips breakers when thresholds exceeded.

    Usage:
        breaker = CircuitBreaker(
            daily_loss_limit=0.02,  # 2% daily loss limit
            drawdown_limit=0.10,    # 10% max drawdown
            volatility_limit=0.05   # 5% volatility spike threshold
        )

        # Check before trading
        if breaker.is_tripped():
            print("Trading halted - circuit breaker active")
            return

        # Update with current equity
        breaker.update(current_equity=100000.0)
    """

    @validate_params(
        daily_loss_limit=validate_range_param('daily_loss_limit', min_val=0.0, max_val=1.0),
        drawdown_limit=validate_range_param('drawdown_limit', min_val=0.0, max_val=1.0),
        volatility_limit=validate_range_param('volatility_limit', min_val=0.0)
    )
    def __init__(
        self,
        daily_loss_limit: float = 0.02,  # 2% default
        drawdown_limit: float = 0.10,  # 10% default
        volatility_limit: float = 0.05,  # 5% default
        state_file: str = "data/circuit_breaker_state.json",
    ):
        """Initialize circuit breaker with thresholds.

        Args:
            daily_loss_limit: Maximum allowed daily loss as fraction (0.02 = 2%)
            drawdown_limit: Maximum allowed drawdown from peak as fraction
            volatility_limit: Maximum allowed volatility before halt
            state_file: Path to persist state across restarts
        """
        self.daily_loss_limit = daily_loss_limit
        self.drawdown_limit = drawdown_limit
        self.volatility_limit = volatility_limit
        self.state_file = Path(state_file)

        # Load or initialize state
        self.state = self._load_state()

    def _load_state(self) -> CircuitBreakerState:
        """Load state from file or create new state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return CircuitBreakerState(**data)
            except Exception as e:
                LOGGER.warning(f"Failed to load circuit breaker state: {e}")
        return CircuitBreakerState()

    def _save_state(self) -> None:
        """Persist state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, indent=2)

    def is_tripped(self) -> bool:
        """Check if any circuit breaker is currently tripped."""
        return (
            self.state.daily_loss_breaker_tripped or
            self.state.drawdown_breaker_tripped or
            self.state.volatility_breaker_tripped
        )

    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of all breakers."""
        return {
            "is_tripped": self.is_tripped(),
            "daily_loss_tripped": self.state.daily_loss_breaker_tripped,
            "drawdown_tripped": self.state.drawdown_breaker_tripped,
            "volatility_tripped": self.state.volatility_breaker_tripped,
            "current_date": self.state.current_date,
            "start_of_day_equity": self.state.start_of_day_equity,
            "peak_equity": self.state.peak_equity,
            "last_trip_time": self.state.last_trip_time,
            "thresholds": {
                "daily_loss_limit": self.daily_loss_limit,
                "drawdown_limit": self.drawdown_limit,
                "volatility_limit": self.volatility_limit,
            }
        }

    def update(self, current_equity: float, volatility: Optional[float] = None) -> bool:
        """Update circuit breaker with current portfolio state.

        Args:
            current_equity: Current portfolio equity value
            volatility: Optional current volatility measure (e.g., from GARCH)

        Returns:
            True if any breaker was tripped by this update
        """
        today = date.today().isoformat()
        tripped = False

        # Check for new day - reset daily breaker
        if self.state.current_date != today:
            LOGGER.info(f"New trading day: {today}. Resetting daily breaker.")
            self.state.current_date = today
            self.state.start_of_day_equity = current_equity
            self.state.daily_loss_breaker_tripped = False
            self.state.last_reset_time = datetime.now().isoformat()

        # Update peak equity for drawdown calculation
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity

        # Calculate daily loss
        if self.state.start_of_day_equity > 0:
            daily_loss = (self.state.start_of_day_equity - current_equity) / self.state.start_of_day_equity
            if daily_loss >= self.daily_loss_limit:
                if not self.state.daily_loss_breaker_tripped:
                    LOGGER.critical(
                        f"CIRCUIT BREAKER TRIPPED: Daily loss {daily_loss:.2%} exceeds limit {self.daily_loss_limit:.2%}"
                    )
                    self.state.daily_loss_breaker_tripped = True
                    self.state.last_trip_time = datetime.now().isoformat()
                    tripped = True

        # Calculate drawdown from peak
        if self.state.peak_equity > 0:
            drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
            if drawdown >= self.drawdown_limit:
                if not self.state.drawdown_breaker_tripped:
                    LOGGER.critical(
                        f"CIRCUIT BREAKER TRIPPED: Drawdown {drawdown:.2%} exceeds limit {self.drawdown_limit:.2%}"
                    )
                    self.state.drawdown_breaker_tripped = True
                    self.state.last_trip_time = datetime.now().isoformat()
                    tripped = True

        # Check volatility if provided
        if volatility is not None and volatility >= self.volatility_limit:
            if not self.state.volatility_breaker_tripped:
                LOGGER.critical(
                    f"CIRCUIT BREAKER TRIPPED: Volatility {volatility:.2%} exceeds limit {self.volatility_limit:.2%}"
                )
                self.state.volatility_breaker_tripped = True
                self.state.last_trip_time = datetime.now().isoformat()
                tripped = True

        # Persist state
        self._save_state()
        return tripped

    def reset(self, breaker: Optional[str] = None) -> None:
        """Reset circuit breaker(s).

        Args:
            breaker: Specific breaker to reset ('daily', 'drawdown', 'volatility')
                    If None, resets all breakers.
        """
        if breaker is None or breaker == "daily":
            self.state.daily_loss_breaker_tripped = False
        if breaker is None or breaker == "drawdown":
            self.state.drawdown_breaker_tripped = False
        if breaker is None or breaker == "volatility":
            self.state.volatility_breaker_tripped = False

        self.state.last_reset_time = datetime.now().isoformat()
        self._save_state()
        LOGGER.info(f"Circuit breaker reset: {breaker or 'all'}")

    def reset_peak(self, new_peak: float) -> None:
        """Reset peak equity to a new value (e.g., after capital injection)."""
        self.state.peak_equity = new_peak
        self.state.drawdown_breaker_tripped = False
        self._save_state()
        LOGGER.info(f"Peak equity reset to {new_peak}")


# Global instance with default thresholds
CIRCUIT_BREAKER = CircuitBreaker()
