"""Walk-Forward Optimization (WFO) evaluator (minimal version).

For a given dataframe df, a strategy_factory(params)->strategy, and a parameter grid,
this performs rolling train/test splits. For each split:
- Choose params that maximize Sharpe on train
- Evaluate on test and record Sharpe
Aggregate test Sharpe across splits.
"""
from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import itertools
import numpy as np
import pandas as pd

from ..backtest.engine import backtest_signals
from ..backtest.metrics import sharpe


@dataclass
class WFOSpec:
    n_splits: int = 4
    train_frac: float = 0.7  # fraction of window used for training


def _split_windows(index: pd.DatetimeIndex, n_splits: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate rolling window boundaries over the full index."""
    if len(index) < n_splits + 2:
        return [(index.min(), index.max())]
    bounds: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    step = int(len(index) / (n_splits + 1))
    for i in range(n_splits):
        start = index[i * step]
        end = index[(i + 1) * step]
        bounds.append((start, end))
    return bounds


def walk_forward_evaluate(
    df: pd.DataFrame,
    strategy_factory,
    param_grid: Dict[str, Iterable[Any]],
    fee_bps: float = 2.0,
    weight: float = 1.0,
    wfo: WFOSpec = WFOSpec(),
) -> Dict[str, Any]:
    """Run minimal WFO evaluation and return aggregated metrics.
    Returns dict with keys: test_sharpes (list), mean_test_sharpe, best_params_per_split
    """
    idx = df.index
    splits = _split_windows(idx, wfo.n_splits)
    best_params: List[Dict[str, Any]] = []
    test_sharpes: List[float] = []

    # Iterate through rolling windows
    for (start, end) in splits:
        df_win = df[(df.index >= start) & (df.index <= end)]
        if df_win.empty:
            continue
        # Train/Test split within window
        n = len(df_win)
        cut = int(n * wfo.train_frac)
        df_train = df_win.iloc[:cut]
        df_test = df_win.iloc[cut:]
        if len(df_test) < 5:
            continue

        # Grid search on train
        candidates = list(itertools.product(*param_grid.values()))
        keys = list(param_grid.keys())
        best_s = -np.inf
        best_p = None
        for combo in candidates:
            params = dict(zip(keys, combo))
            try:
                strat = strategy_factory(**params)
                sig = strat.generate_signals(df_train)
                res = backtest_signals(df_train, sig, fee_bps=fee_bps, weight=weight)
                s = sharpe(res.returns, freq="1d")  # freq approximation; could infer from index spacing
                if s > best_s:
                    best_s = s
                    best_p = params
            except Exception:
                continue
        if best_p is None:
            continue
        best_params.append(best_p)

        # Evaluate on test with best params
        strat = strategy_factory(**best_p)
        sig = strat.generate_signals(df_test)
        res = backtest_signals(df_test, sig, fee_bps=fee_bps, weight=weight)
        test_s = sharpe(res.returns, freq="1d")
        test_sharpes.append(float(test_s))

    mean_test_sharpe = float(np.mean(test_sharpes)) if test_sharpes else 0.0
    return {
        "test_sharpes": test_sharpes,
        "mean_test_sharpe": mean_test_sharpe,
        "best_params_per_split": best_params,
    }

