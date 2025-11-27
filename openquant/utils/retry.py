"""Simple retry/backoff utility functions.

Provides:
- retry_call: Generic retry with exponential backoff
- ConnectionRetry: Specialized retry for broker connections (MT5, Alpaca)
"""
from __future__ import annotations

import time
import functools
from typing import Callable, Type, Tuple, Any, Optional
from dataclasses import dataclass

from .logging import get_logger

LOGGER = get_logger(__name__)


def retry_call(
    fn: Callable[[], Any],
    retries: int = 3,
    backoff: float = 1.5,
    base_delay: float = 0.5,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
) -> Any:
    """Call fn() with exponential backoff.

    Args:
        fn: Function to call (no arguments)
        retries: Max attempts (total calls = retries)
        backoff: Multiplier for delay between retries
        base_delay: Initial delay in seconds
        retry_on: Exception types to retry on

    Returns:
        Result of fn() if successful

    Raises:
        Last exception if all retries exhausted
    """
    attempt = 0
    delay = base_delay
    while True:
        try:
            return fn()
        except retry_on as e:
            attempt += 1
            if attempt >= retries:
                raise
            time.sleep(delay)
            delay *= backoff


@dataclass
class RetryConfig:
    """Configuration for connection retry behavior."""
    max_retries: int = 5           # Maximum retry attempts
    base_delay: float = 1.0        # Initial delay in seconds
    max_delay: float = 60.0        # Maximum delay between retries
    backoff_factor: float = 2.0    # Exponential backoff multiplier
    jitter: float = 0.1            # Random jitter factor (0-1)


class ConnectionRetry:
    """
    Specialized retry handler for broker connections.

    Features:
    - Exponential backoff with jitter to prevent thundering herd
    - Connection state tracking
    - Automatic reconnection attempts
    - Logging of retry attempts

    Usage:
        retry = ConnectionRetry(name="MT5")

        @retry.with_retry
        def connect_to_mt5():
            mt5.initialize()
            return mt5.login(...)

        # Or use context manager
        with retry.connection(connect_fn, disconnect_fn):
            # Use connection
            pass
    """

    def __init__(
        self,
        name: str = "Broker",
        config: Optional[RetryConfig] = None
    ):
        """Initialize connection retry handler.

        Args:
            name: Name for logging (e.g., "MT5", "Alpaca")
            config: Retry configuration (uses defaults if None)
        """
        self.name = name
        self.config = config or RetryConfig()
        self._connected = False
        self._retry_count = 0
        self._last_error: Optional[Exception] = None

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected

    @property
    def last_error(self) -> Optional[Exception]:
        """Get last connection error."""
        return self._last_error

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        import random

        # Exponential backoff: delay = base * (factor ^ attempt)
        delay = self.config.base_delay * (self.config.backoff_factor ** attempt)

        # Cap at max_delay
        delay = min(delay, self.config.max_delay)

        # Add jitter to prevent thundering herd
        jitter = delay * self.config.jitter * random.random()
        return delay + jitter

    def with_retry(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator to add retry logic to a connection function.

        Args:
            fn: Function to wrap with retry logic

        Returns:
            Wrapped function with retry behavior
        """
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            self._retry_count = 0
            self._last_error = None

            for attempt in range(self.config.max_retries):
                try:
                    result = fn(*args, **kwargs)
                    self._connected = True
                    self._retry_count = attempt
                    LOGGER.info(f"{self.name}: Connected successfully")
                    return result

                except Exception as e:
                    self._last_error = e
                    self._connected = False

                    if attempt < self.config.max_retries - 1:
                        delay = self._calculate_delay(attempt)
                        LOGGER.warning(
                            f"{self.name}: Connection failed (attempt {attempt + 1}/"
                            f"{self.config.max_retries}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        LOGGER.error(
                            f"{self.name}: Connection failed after "
                            f"{self.config.max_retries} attempts: {e}"
                        )
                        raise

            # Should not reach here, but just in case
            raise RuntimeError(f"{self.name}: Connection failed")

        return wrapper

    def execute_with_retry(
        self,
        fn: Callable[[], Any],
        retry_on: Tuple[Type[BaseException], ...] = (Exception,)
    ) -> Any:
        """Execute a function with retry logic.

        Args:
            fn: Function to execute
            retry_on: Exception types to retry on

        Returns:
            Result of fn() if successful
        """
        return self.with_retry(fn)()

    def reset(self):
        """Reset connection state."""
        self._connected = False
        self._retry_count = 0
        self._last_error = None


# Pre-configured retry handlers for common brokers
MT5_RETRY = ConnectionRetry(
    name="MT5",
    config=RetryConfig(max_retries=5, base_delay=2.0, max_delay=30.0)
)

ALPACA_RETRY = ConnectionRetry(
    name="Alpaca",
    config=RetryConfig(max_retries=3, base_delay=1.0, max_delay=10.0)
)

CCXT_RETRY = ConnectionRetry(
    name="CCXT",
    config=RetryConfig(max_retries=3, base_delay=0.5, max_delay=5.0)
)

