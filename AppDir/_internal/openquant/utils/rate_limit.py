"""Simple thread-safe token bucket rate limiter.

Usage:
    limiter = get_rate_limiter("binance", rate_per_sec=8, capacity=8)
    limiter.acquire()  # blocks until 1 token is available

For tests, you can pass a custom time function to RateLimiter.
"""
from __future__ import annotations
import time
import threading
from typing import Callable, Dict


class RateLimiter:
    def __init__(self, rate_per_sec: float, capacity: int, time_fn: Callable[[], float] | None = None):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.rate = float(rate_per_sec)
        self.capacity = int(capacity)
        self._tokens = float(capacity)
        self._time = time_fn or time.monotonic
        self._last = self._time()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        if now <= self._last:
            return
        delta = now - self._last
        self._tokens = min(self.capacity, self._tokens + delta * self.rate)
        self._last = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Attempt to take tokens without blocking; return True if successful."""
        with self._lock:
            now = self._time()
            self._refill(now)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def acquire(self, tokens: float = 1.0) -> None:
        """Block until the requested tokens are available, then take them."""
        while True:
            with self._lock:
                now = self._time()
                self._refill(now)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                needed = tokens - self._tokens
                wait = needed / self.rate
            # Sleep outside lock
            time.sleep(max(0.0, wait))


_LIMITERS: Dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_rate_limiter(key: str, rate_per_sec: float = 8.0, capacity: int = 8) -> RateLimiter:
    """Return a process-wide rate limiter for the given key (e.g., exchange name)."""
    k = key.lower()
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(k)
        if lim is None:
            lim = RateLimiter(rate_per_sec=rate_per_sec, capacity=capacity)
            _LIMITERS[k] = lim
        return lim

