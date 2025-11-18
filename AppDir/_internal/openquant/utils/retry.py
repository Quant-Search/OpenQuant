"""Simple retry/backoff utility functions."""
from __future__ import annotations
import time
from typing import Callable, Type, Tuple, Any


def retry_call(
    fn: Callable[[], Any],
    retries: int = 3,
    backoff: float = 1.5,
    base_delay: float = 0.5,
    retry_on: Tuple[Type[BaseException], ...] = (Exception,),
) -> Any:
    """Call fn() with exponential backoff.
    - retries: max attempts (total calls = retries)
    - backoff: multiplier for delay
    - base_delay: initial delay in seconds
    - retry_on: exception types to retry
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

