"""
Periodic Retrain Scheduler Module.

Provides automated Walk-Forward Optimization (WFO) retraining:
- Configurable schedule (daily, weekly, monthly)
- Automatic model refresh based on new data
- State persistence for tracking last retrain time
- Integration with universe_runner for full pipeline

Mathematical basis:
- WFO splits data into in-sample (training) and out-of-sample (validation)
- Periodic retraining ensures model adapts to regime changes
- Recommended frequency: weekly for daily strategies, monthly for longer horizons

The scheduler can run as:
1. A standalone daemon process
2. A cron job triggered externally
3. Integrated into the main trading loop
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time

from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


class RetrainFrequency(Enum):
    """Retraining frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"


@dataclass
class RetrainState:
    """Persistent state for retrain scheduler."""
    last_retrain_time: str = ""  # ISO format timestamp
    last_retrain_status: str = ""  # "success", "failed", "running"
    last_retrain_duration_seconds: float = 0.0
    total_retrains: int = 0
    successful_retrains: int = 0
    failed_retrains: int = 0


class RetrainScheduler:
    """
    Manages periodic retraining of trading strategies.

    Usage:
        scheduler = RetrainScheduler(
            frequency=RetrainFrequency.WEEKLY,
            retrain_func=my_retrain_function
        )

        # Check if retrain is due
        if scheduler.is_retrain_due():
            scheduler.run_retrain()

        # Or run as daemon
        scheduler.start_daemon()
    """

    def __init__(
        self,
        frequency: RetrainFrequency = RetrainFrequency.WEEKLY,
        retrain_func: Optional[Callable[[], bool]] = None,
        state_file: str = "data/retrain_state.json",
        retrain_hour: int = 2,  # Hour of day to run retrain (UTC)
        retrain_day: int = 0,  # Day of week for weekly (0=Monday)
    ):
        """Initialize retrain scheduler.

        Args:
            frequency: How often to retrain
            retrain_func: Function to call for retraining (returns True on success)
            state_file: Path to persist state
            retrain_hour: Hour of day to run retrain (0-23, UTC)
            retrain_day: Day of week for weekly retrain (0=Monday, 6=Sunday)
        """
        self.frequency = frequency
        self.retrain_func = retrain_func
        self.state_file = Path(state_file)
        self.retrain_hour = retrain_hour
        self.retrain_day = retrain_day

        # Load or initialize state
        self.state = self._load_state()

        # Daemon control
        self._daemon_running = False
        self._daemon_thread: Optional[threading.Thread] = None

    def _load_state(self) -> RetrainState:
        """Load state from file or create new state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return RetrainState(**data)
            except Exception as e:
                LOGGER.warning(f"Failed to load retrain state: {e}")
        return RetrainState()

    def _save_state(self) -> None:
        """Persist state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, indent=2)

    def _get_next_retrain_time(self) -> datetime:
        """Calculate the next scheduled retrain time."""
        now = datetime.utcnow()

        if self.frequency == RetrainFrequency.DAILY:
            # Next occurrence of retrain_hour
            next_time = now.replace(hour=self.retrain_hour, minute=0, second=0, microsecond=0)
            if next_time <= now:
                next_time += timedelta(days=1)
            return next_time

        elif self.frequency == RetrainFrequency.WEEKLY:
            # Next occurrence of retrain_day at retrain_hour
            days_ahead = self.retrain_day - now.weekday()
            if days_ahead < 0:
                days_ahead += 7
            next_time = now.replace(hour=self.retrain_hour, minute=0, second=0, microsecond=0)
            next_time += timedelta(days=days_ahead)
            if next_time <= now:
                next_time += timedelta(weeks=1)
            return next_time

        elif self.frequency == RetrainFrequency.BIWEEKLY:
            # Every 2 weeks
            next_time = self._get_next_retrain_time_for_weekly()
            # Check if we should skip a week
            if self.state.last_retrain_time:
                last = datetime.fromisoformat(self.state.last_retrain_time)
                if (next_time - last).days < 14:
                    next_time += timedelta(weeks=1)
            return next_time


    def _get_next_retrain_time_for_weekly(self) -> datetime:
        """Helper for weekly calculation."""
        now = datetime.utcnow()
        days_ahead = self.retrain_day - now.weekday()
        if days_ahead < 0:
            days_ahead += 7
        next_time = now.replace(hour=self.retrain_hour, minute=0, second=0, microsecond=0)
        next_time += timedelta(days=days_ahead)
        if next_time <= now:
            next_time += timedelta(weeks=1)
        return next_time

    def is_retrain_due(self) -> bool:
        """Check if a retrain is due based on schedule."""
        if not self.state.last_retrain_time:
            # Never retrained, so it's due
            return True

        try:
            last_retrain = datetime.fromisoformat(self.state.last_retrain_time)
        except ValueError:
            return True

        now = datetime.utcnow()

        if self.frequency == RetrainFrequency.DAILY:
            return (now - last_retrain) >= timedelta(days=1)
        elif self.frequency == RetrainFrequency.WEEKLY:
            return (now - last_retrain) >= timedelta(weeks=1)
        elif self.frequency == RetrainFrequency.BIWEEKLY:
            return (now - last_retrain) >= timedelta(weeks=2)
        elif self.frequency == RetrainFrequency.MONTHLY:
            return (now - last_retrain) >= timedelta(days=28)

        return False

    def run_retrain(self) -> bool:
        """Execute the retrain function and update state.

        Returns:
            True if retrain was successful, False otherwise
        """
        if self.retrain_func is None:
            LOGGER.error("No retrain function configured")
            return False

        LOGGER.info(f"Starting scheduled retrain (frequency: {self.frequency.value})")

        self.state.last_retrain_status = "running"
        self.state.total_retrains += 1
        self._save_state()

        start_time = time.time()
        success = False

        try:
            success = self.retrain_func()
            self.state.last_retrain_status = "success" if success else "failed"
            if success:
                self.state.successful_retrains += 1
            else:
                self.state.failed_retrains += 1
        except Exception as e:
            LOGGER.error(f"Retrain failed with exception: {e}")
            self.state.last_retrain_status = "failed"
            self.state.failed_retrains += 1

        self.state.last_retrain_time = datetime.utcnow().isoformat()
        self.state.last_retrain_duration_seconds = time.time() - start_time
        self._save_state()

        LOGGER.info(
            f"Retrain completed: status={self.state.last_retrain_status}, "
            f"duration={self.state.last_retrain_duration_seconds:.1f}s"
        )

        return success

    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "frequency": self.frequency.value,
            "last_retrain_time": self.state.last_retrain_time,
            "last_retrain_status": self.state.last_retrain_status,
            "last_retrain_duration_seconds": self.state.last_retrain_duration_seconds,
            "total_retrains": self.state.total_retrains,
            "successful_retrains": self.state.successful_retrains,
            "failed_retrains": self.state.failed_retrains,
            "is_retrain_due": self.is_retrain_due(),
            "next_retrain_time": self._get_next_retrain_time().isoformat(),
        }

    def start_daemon(self, check_interval_seconds: int = 3600) -> None:
        """Start the scheduler as a background daemon.

        Args:
            check_interval_seconds: How often to check if retrain is due
        """
        if self._daemon_running:
            LOGGER.warning("Daemon already running")
            return

        def daemon_loop():
            LOGGER.info(f"Retrain daemon started (frequency: {self.frequency.value})")
            while self._daemon_running:
                if self.is_retrain_due():
                    self.run_retrain()
                time.sleep(check_interval_seconds)
            LOGGER.info("Retrain daemon stopped")

        self._daemon_running = True
        self._daemon_thread = threading.Thread(target=daemon_loop, daemon=True)
        self._daemon_thread.start()

    def stop_daemon(self) -> None:
        """Stop the background daemon."""
        self._daemon_running = False
        if self._daemon_thread:
            self._daemon_thread.join(timeout=5)
            self._daemon_thread = None


def create_default_retrain_func(
    exchange: str = "binance",
    strategies: tuple = ("kalman", "hurst", "stat_arb", "liquidity"),
) -> Callable[[], bool]:
    """Create a default retrain function using universe_runner.

    Args:
        exchange: Exchange to run research on
        strategies: Strategies to optimize

    Returns:
        A callable that runs the full research pipeline
    """
    def retrain() -> bool:
        try:
            from openquant.research.universe_runner import run_universe

            LOGGER.info(f"Running universe research for {exchange}")
            rows = run_universe(
                exchange=exchange,
                strategies=strategies,
                run_wfo=True,
                optimize=True,
            )

            # Check if we got valid results
            ok_count = sum(1 for r in rows if (r.get("metrics") or {}).get("ok"))
            LOGGER.info(f"Retrain complete: {ok_count}/{len(rows)} strategies passed")

            return ok_count > 0
        except Exception as e:
            LOGGER.error(f"Retrain failed: {e}")
            return False

    return retrain


# Global instance with default configuration
RETRAIN_SCHEDULER = RetrainScheduler()
