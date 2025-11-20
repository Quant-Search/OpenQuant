"""Optuna-based parameter search utilities.

We use categorical suggestions derived from the provided param grid.
Objective maximizes Deflated Sharpe Ratio (DSR) computed on full sample.
"""
from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, List

import optuna  # type: ignore
import numpy as np
import pandas as pd

from ..strategies.registry import make_strategy
from ..backtest.engine import backtest_signals
from ..backtest.metrics import sharpe
from ..evaluation.deflated_sharpe import deflated_sharpe_ratio


def _freq_from_timeframe(tf: str) -> str:
    return tf.lower()


def _grid_keys_values(grid: Dict[str, Iterable[Any]]) -> Tuple[List[str], List[List[Any]]]:
    keys = list(grid.keys())
    vals = [list(grid[k]) for k in keys]
    return keys, vals


def optuna_best_params(
    strat_name: str,
    df: pd.DataFrame,
    grid: Dict[str, Iterable[Any]],
    fee_bps: float,
    weight: float,
    timeframe: str,
    n_trials: int = 20,  # Reduced from 50 for speed
) -> Dict[str, Any]:
    """Run an Optuna study to pick best params from the (categorical) grid.
    Returns best parameter dict.
    """
    keys, vals = _grid_keys_values(grid)
    freq = _freq_from_timeframe(timeframe)

    def objective(trial: optuna.trial.Trial) -> float:  # type: ignore
        params = {k: trial.suggest_categorical(k, v) for k, v in zip(keys, vals)}
        try:
            strat = make_strategy(strat_name, **params)
            sig = strat.generate_signals(df)
            res = backtest_signals(df, sig, fee_bps=fee_bps, weight=weight)
            s = sharpe(res.returns, freq=freq)
            r = res.returns.dropna().values
            T = int(r.size) if r.size else 1
            dsr = float(deflated_sharpe_ratio(s, T=T, trials=max(n_trials, 1)))
            # We maximize DSR
            return dsr
        except Exception:
            # Penalize invalid configurations
            return -1e6

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params if study.best_trial is not None else ({k: vals[i][0] for i, k in enumerate(keys)})

