"""Purged K-Fold cross-validation splits for time series (minimal).

Creates train/test indices with a purge gap around test folds to avoid leakage.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def purged_kfold_indices(n: int, k: int = 5, purge: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, test_idx) with purge around test.
    Args:
        n: length of the series
        k: number of folds
        purge: number of points to purge before and after the test fold
    """
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_size = max(1, n // k)
    for i in range(k):
        test_start = i * fold_size
        test_end = min(n, (i + 1) * fold_size)
        test_idx = np.arange(test_start, test_end)
        # Purge region bounds
        left = max(0, test_start - purge)
        right = min(n, test_end + purge)
        train_idx = np.concatenate([np.arange(0, left), np.arange(right, n)])
        folds.append((train_idx, test_idx))
    return folds

