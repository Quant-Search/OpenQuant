"""Base classes for strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base strategy: produce position signals given OHLCV DataFrame.
    Signals convention: -1 short, 0 flat, +1 long (position units, not weights).
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a signal series indexed like df with values in {-1,0,1}."""
        raise NotImplementedError
