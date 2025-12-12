"""Combinatorially Purged Cross-Validation (CPCV).

Implements advanced cross-validation that prevents data leakage in time series
by purging training samples close to test samples and using embargo periods.

Based on "Advances in Financial Machine Learning" by Marcos López de Prado.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Iterator
from itertools import combinations

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

class CombinatorialPurgedCV:
    """
    Combinatorially Purged Cross-Validation for time series.
    
    Prevents overfitting by:
    1. Purging training samples close to test samples (temporal leakage)
    2. Adding embargo period after test set
    3. Testing multiple train/test combinations
    """
    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_pct: float = 0.02,
        embargo_pct: float = 0.01
    ):
        """
        Args:
            n_splits: Number of splits to create.
            n_test_splits: Number of splits to use as test in each combination.
            purge_pct: Percentage of data before test set to purge.
            embargo_pct: Percentage of data after test set to embargo.
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        
    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices with purging and embargo.
        
        Args:
            X: Input data (must have DatetimeIndex or similar).
            
        Yields:
            (train_indices, test_indices) tuples.
        """
        n = len(X)
        split_size = n // self.n_splits
        
        # Create splits
        splits = []
        for i in range(self.n_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < self.n_splits - 1 else n
            splits.append((start, end))
            
        # Generate combinations of test splits
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))
        
        LOGGER.info(f"CPCV: {len(test_combinations)} combinations with {self.n_splits} splits")
        
        for test_split_ids in test_combinations:
            # Gather test indices
            test_indices = []
            for split_id in test_split_ids:
                start, end = splits[split_id]
                test_indices.extend(range(start, end))
                
            test_indices = sorted(test_indices)
            
            # Calculate purge and embargo zones
            purge_size = int(n * self.purge_pct)
            embargo_size = int(n * self.embargo_pct)
            
            # Find train indices (all splits not in test)
            train_indices = []
            for i in range(self.n_splits):
                if i not in test_split_ids:
                    start, end = splits[i]
                    train_indices.extend(range(start, end))
                    
            # Purge training samples close to test samples
            train_indices_purged = []
            for idx in train_indices:
                # Check if this training sample is too close to any test sample
                min_dist_to_test = min([abs(idx - t) for t in test_indices])
                
                if min_dist_to_test > purge_size:
                    train_indices_purged.append(idx)
                    
            # Apply embargo: remove training samples right after test set
            if test_indices:
                test_end = max(test_indices)
                embargo_zone = range(test_end + 1, min(test_end + 1 + embargo_size, n))
                train_indices_purged = [idx for idx in train_indices_purged if idx not in embargo_zone]
                
            yield (np.array(train_indices_purged), np.array(test_indices))
            
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)
