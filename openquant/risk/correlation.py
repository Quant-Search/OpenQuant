from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

def filter_correlated_candidates(
    candidates: List[Tuple[int, Dict[str, Any]]], 
    threshold: float = 0.8
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Filter out candidates that are highly correlated with higher-ranked candidates.
    
    Args:
        candidates: List of (index, row) tuples, already sorted by score (descending).
        threshold: Correlation threshold (0.0 to 1.0). If corr > threshold, the lower-ranked one is dropped.
        
    Returns:
        Filtered list of candidates.
    """
    if not candidates:
        return []
        
    accepted = []
    accepted_returns = {} # map index -> pd.Series (returns)
    
    for idx, row in candidates:
        metrics = row.get("metrics") or {}
        returns = metrics.get("returns")
        
        # If no returns data, we can't check correlation. 
        # Decide whether to keep or drop. Let's keep but warn? 
        # Or just keep as "uncorrelated" by default.
        if returns is None or not isinstance(returns, pd.Series) or returns.empty:
            accepted.append((idx, row))
            continue
            
        # Align returns with already accepted candidates
        # We need a common index. 
        # Optimization: Only check against candidates with same symbol? 
        # NO, we want to filter correlation ACROSS symbols too (e.g. BTC vs ETH strategies).
        # But aligning daily returns across different symbols requires a common date index.
        # Assuming returns are daily and indexed by datetime.
        
        is_correlated = False
        for acc_idx, acc_row in accepted:
            acc_ret = accepted_returns.get(acc_idx)
            if acc_ret is None:
                continue
                
            # Calculate correlation
            # We need to align indices.
            # Inner join?
            common_idx = returns.index.intersection(acc_ret.index)
            if len(common_idx) < 10: # Not enough overlap
                continue
                
            s1 = returns.loc[common_idx]
            s2 = acc_ret.loc[common_idx]
            
            corr = s1.corr(s2)
            
            if corr > threshold:
                is_correlated = True
                # Log or mark as dropped?
                # For now just drop
                break
        
        if not is_correlated:
            accepted.append((idx, row))
            accepted_returns[idx] = returns
            
    return accepted
