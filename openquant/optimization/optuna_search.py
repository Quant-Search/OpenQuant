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
    symbol: str = "unknown",
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run an Optuna study to pick best params from the (categorical) grid.
    Returns best parameter dict with optional caching.
    
    Args:
        strat_name: Strategy name
        df: OHLCV DataFrame
        grid: Parameter grid
        fee_bps: Fee in basis points
        weight: Position weight
        timeframe: Data timeframe
        n_trials: Number of optimization trials
        symbol: Symbol identifier for caching
        use_cache: Whether to use cache
    """
    cache = get_cache() if use_cache else None
    
    # Try cache first
    if cache:
        cached = cache.get_optimization(
            strat_name, symbol, timeframe,
            fee_bps=fee_bps, weight=weight, n_trials=n_trials, grid_hash=str(hash(str(grid)))
        )
        if cached is not None:
            return cached
    
    keys, vals = _grid_keys_values(grid)
    freq = _freq_from_timeframe(timeframe)

    def objective(trial: optuna.trial.Trial) -> float:  # type: ignore
        params = {}
        for k, v_list in zip(keys, vals):
            # Dynamic type detection for search space
            # For small grids (<=5 values), always use categorical to ensure
            # returned values are exactly from the grid
            if not v_list:
                continue

            # Small grids: use categorical to guarantee exact grid values
            if len(v_list) <= 5:
                params[k] = trial.suggest_categorical(k, v_list)
                continue

            first_val = v_list[0]
            if isinstance(first_val, bool):
                params[k] = trial.suggest_categorical(k, v_list)
            elif isinstance(first_val, int):
                # Use int range if list looks like a range (>5 values)
                if all(isinstance(x, int) for x in v_list):
                    min_v, max_v = min(v_list), max(v_list)
                    # Step detection (heuristic)
                    step = 1
                    sorted_v = sorted(list(set(v_list)))
                    if len(sorted_v) > 2:
                        diffs = [sorted_v[i+1] - sorted_v[i] for i in range(len(sorted_v)-1)]
                        if len(set(diffs)) == 1:
                            step = diffs[0]
                    params[k] = trial.suggest_int(k, min_v, max_v, step=step)
                else:
                    params[k] = trial.suggest_categorical(k, v_list)
            elif isinstance(first_val, float):
                # Use float range for large grids
                if all(isinstance(x, (int, float)) for x in v_list):
                    min_v, max_v = min(v_list), max(v_list)
                    params[k] = trial.suggest_float(k, min_v, max_v)
                else:
                    params[k] = trial.suggest_categorical(k, v_list)
            else:
                params[k] = trial.suggest_categorical(k, v_list)

        try:
            strat = make_strategy(strat_name, **params)
            sig = strat.generate_signals(df)
            res = backtest_signals(df, sig, fee_bps=fee_bps, weight=weight)
            s = sharpe(res.returns, freq=freq)
            r = res.returns.dropna().values
            T = int(r.size) if r.size else 1
            dsr = float(deflated_sharpe_ratio(s, T=T, trials=max(n_trials, 1)))
            return dsr
        except Exception:
            return -1e6

    # Use TPE Sampler with multivariate enabled for better exploration
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params if study.best_trial is not None else ({k: vals[i][0] for i, k in enumerate(keys)})

