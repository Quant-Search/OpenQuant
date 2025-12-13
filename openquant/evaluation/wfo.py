"""Walk-Forward Optimization (WFO) evaluator (minimal version).

For a given dataframe df, a strategy_factory(params)->strategy, and a parameter grid,
this performs rolling train/test splits. For each split:
- Choose params that maximize Sharpe on train
- Evaluate on test and record Sharpe
Aggregate test Sharpe across splits.
"""
from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, List, Optional, Callable
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
    use_cpcv: bool = False  # Use Combinatorially Purged CV
    cpcv_n_test_splits: int = 2
    cpcv_purge_pct: float = 0.02
    cpcv_embargo_pct: float = 0.01
    use_monte_carlo: bool = False  # Run Monte Carlo robustness testing
    mc_n_simulations: int = 100  # Number of MC simulations (reduced from default for WFO)
    mc_block_size: int = 20  # Block size for path-dependent bootstrap
    mc_param_perturbation_pct: float = 0.1  # Parameter perturbation percentage
    anchored: bool = False  # Use anchored walk-forward with expanding window
    mc_param_noise: bool = False  # Apply Monte Carlo parameter perturbation to best params
    sharpe_stability_check: bool = False  # Check Sharpe stability across folds


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


def _split_windows_anchored(index: pd.DatetimeIndex, n_splits: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate anchored window boundaries with expanding training window.
    
    In anchored WFO, the training window starts from the beginning and expands
    as we move forward, while the test window moves forward in time.
    """
    if len(index) < n_splits + 2:
        return [(index.min(), index.max())]
    
    bounds: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    anchor_start = index.min()
    step = int(len(index) / (n_splits + 1))
    
    for i in range(n_splits):
        # Window always starts from anchor_start (expanding train window)
        start = anchor_start
        end = index[(i + 1) * step]
        bounds.append((start, end))
    
    return bounds


def _perturb_params(params: Dict[str, Any], perturbation_pct: float = 0.1) -> Dict[str, Any]:
    """Apply Monte Carlo parameter perturbation with ±perturbation_pct noise.
    
    Args:
        params: Dictionary of parameters
        perturbation_pct: Perturbation percentage (default 0.1 for ±10%)
    
    Returns:
        Perturbed parameters dictionary
    """
    perturbed = {}
    for key, value in params.items():
        if isinstance(value, (int, float)):
            # Apply random perturbation: value * (1 + uniform(-pct, +pct))
            noise = np.random.uniform(-perturbation_pct, perturbation_pct)
            new_value = value * (1.0 + noise)
            
            # Preserve int/float type
            if isinstance(value, int):
                perturbed[key] = int(round(new_value))
            else:
                perturbed[key] = float(new_value)
        else:
            # Keep non-numeric parameters unchanged
            perturbed[key] = value
    
    return perturbed


def _calculate_sharpe_stability(fold_sharpes: List[float]) -> Dict[str, float]:
    """Calculate Sharpe stability metrics across folds.
    
    Args:
        fold_sharpes: List of Sharpe ratios from each fold
    
    Returns:
        Dictionary with stability metrics:
        - mean: Mean Sharpe
        - std: Standard deviation of Sharpe
        - cv: Coefficient of variation (std/mean)
        - min: Minimum Sharpe
        - max: Maximum Sharpe
        - stability_ratio: mean / std (higher is better)
    """
    if not fold_sharpes:
        return {
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "min": 0.0,
            "max": 0.0,
            "stability_ratio": 0.0
        }
    
    sharpes = np.array(fold_sharpes)
    mean_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    
    # Coefficient of variation
    cv = abs(std_sharpe / mean_sharpe) if abs(mean_sharpe) > 1e-6 else float('inf')
    
    # Stability ratio (inverse of CV, higher is better)
    stability_ratio = abs(mean_sharpe / std_sharpe) if std_sharpe > 1e-6 else 0.0
    
    return {
        "mean": mean_sharpe,
        "std": std_sharpe,
        "cv": cv,
        "min": float(np.min(sharpes)),
        "max": float(np.max(sharpes)),
        "stability_ratio": stability_ratio
    }


def walk_forward_evaluate(
    df: pd.DataFrame,
    strategy_factory,
    param_grid: Dict[str, Iterable[Any]],
    fee_bps: float = 2.0,
    weight: float = 1.0,
    wfo: WFOSpec = WFOSpec(),
) -> Dict[str, Any]:
    """Run WFO evaluation with optional CPCV and Monte Carlo robustness testing.
    
    Enhanced features:
    - Anchored walk-forward: Expanding training window when wfo.anchored=True
    - Monte Carlo parameter perturbation: Apply ±10% noise to best params when wfo.mc_param_noise=True
    - Sharpe stability checks: Track out-of-sample Sharpe stability across folds
    - Expanded results: Fold-level metrics including returns, equity, and per-fold Sharpe
    
    Returns dict with keys:
        - test_sharpes: List of test Sharpe ratios per fold
        - mean_test_sharpe: Mean test Sharpe
        - best_params_per_split: List of best parameters per split
        - fold_metrics: List of per-fold detailed metrics
        - sharpe_stability: Sharpe stability metrics across folds
        - monte_carlo_results: MC results (if wfo.use_monte_carlo=True)
        - robustness_evaluation: Robustness evaluation (if wfo.use_monte_carlo=True)
    """
    # Use CPCV if enabled
    if wfo.use_cpcv:
        from ..validation.combinatorial_cv import CombinatorialPurgedCV
        return _walk_forward_cpcv(df, strategy_factory, param_grid, fee_bps, weight, wfo)
        
    # Original WFO implementation with enhancements
    idx = df.index
    
    # Choose splitting strategy: anchored or rolling
    if wfo.anchored:
        splits = _split_windows_anchored(idx, wfo.n_splits)
    else:
        splits = _split_windows(idx, wfo.n_splits)
    
    best_params: List[Dict[str, Any]] = []
    test_sharpes: List[float] = []
    fold_metrics: List[Dict[str, Any]] = []

    # Iterate through windows
    for fold_idx, (start, end) in enumerate(splits):
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
        
        # Apply Monte Carlo parameter perturbation if enabled
        if wfo.mc_param_noise:
            test_params = _perturb_params(best_p, wfo.mc_param_perturbation_pct)
        else:
            test_params = best_p
        
        best_params.append(best_p)

        # Evaluate on test with best params (potentially perturbed)
        try:
            strat = strategy_factory(**test_params)
            sig = strat.generate_signals(df_test)
            res = backtest_signals(df_test, sig, fee_bps=fee_bps, weight=weight)
            test_s = sharpe(res.returns, freq="1d")
            test_sharpes.append(float(test_s))
            
            # Collect detailed fold-level metrics
            fold_metric = {
                "fold": fold_idx,
                "train_start": start,
                "train_end": df_train.index[-1],
                "test_start": df_test.index[0],
                "test_end": end,
                "train_size": len(df_train),
                "test_size": len(df_test),
                "best_params": best_p,
                "test_params": test_params if wfo.mc_param_noise else best_p,
                "train_sharpe": float(best_s),
                "test_sharpe": float(test_s),
                "test_mean_return": float(res.returns.mean()),
                "test_std_return": float(res.returns.std()),
                "test_total_return": float(res.equity_curve.iloc[-1] - 1.0),
                "test_max_equity": float(res.equity_curve.max()),
                "test_min_equity": float(res.equity_curve.min()),
            }
            fold_metrics.append(fold_metric)
            
        except Exception:
            continue

    mean_test_sharpe = float(np.mean(test_sharpes)) if test_sharpes else 0.0
    
    result = {
        "test_sharpes": test_sharpes,
        "mean_test_sharpe": mean_test_sharpe,
        "best_params_per_split": best_params,
        "fold_metrics": fold_metrics,
    }
    
    # Calculate Sharpe stability metrics if enabled
    if wfo.sharpe_stability_check:
        stability = _calculate_sharpe_stability(test_sharpes)
        result["sharpe_stability"] = stability
    
    # Run Monte Carlo robustness testing if enabled
    if wfo.use_monte_carlo and best_params:
        from .monte_carlo import run_comprehensive_mc, MonteCarloConfig, evaluate_robustness
        from ..utils.logging import get_logger
        
        LOGGER = get_logger(__name__)
        LOGGER.info("Running Monte Carlo robustness testing on best parameters")
        
        # Use the most recent best params (or average if numeric)
        final_params = best_params[-1]
        
        mc_config = MonteCarloConfig(
            n_simulations=wfo.mc_n_simulations,
            block_size=wfo.mc_block_size,
            param_perturbation_pct=wfo.mc_param_perturbation_pct
        )
        
        try:
            mc_results = run_comprehensive_mc(
                df=df,
                strategy_factory=strategy_factory,
                params=final_params,
                config=mc_config,
                fee_bps=fee_bps,
                weight=weight
            )
            
            robustness_eval = evaluate_robustness(mc_results)
            
            result["monte_carlo_results"] = mc_results
            result["robustness_evaluation"] = robustness_eval
            
            LOGGER.info(f"Monte Carlo testing completed: Robustness rating = {robustness_eval['rating']}")
            
        except Exception as e:
            LOGGER.error(f"Monte Carlo testing failed: {e}")
            result["monte_carlo_error"] = str(e)
    
    return result


def _walk_forward_cpcv(
    df: pd.DataFrame,
    strategy_factory,
    param_grid: Dict[str, Iterable[Any]],
    fee_bps: float,
    weight: float,
    wfo: WFOSpec
) -> Dict[str, Any]:
    """WFO using Combinatorially Purged CV."""
    from ..validation.combinatorial_cv import CombinatorialPurgedCV
    from ..utils.logging import get_logger
    
    LOGGER = get_logger(__name__)
    
    cv = CombinatorialPurgedCV(
        n_splits=wfo.n_splits,
        n_test_splits=wfo.cpcv_n_test_splits,
        purge_pct=wfo.cpcv_purge_pct,
        embargo_pct=wfo.cpcv_embargo_pct
    )
    
    test_sharpes: List[float] = []
    best_params: List[Dict[str, Any]] = []
    fold_metrics: List[Dict[str, Any]] = []
    
    LOGGER.info("Running CPCV Walk-Forward Optimization...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df)):
        if len(train_idx) < 20 or len(test_idx) < 5:
            continue
            
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]
        
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
                s = sharpe(res.returns, freq="1d")
                if s > best_s:
                    best_s = s
                    best_p = params
            except Exception:
                continue
                
        if best_p is None:
            continue
        
        # Apply Monte Carlo parameter perturbation if enabled
        if wfo.mc_param_noise:
            test_params = _perturb_params(best_p, wfo.mc_param_perturbation_pct)
        else:
            test_params = best_p
            
        best_params.append(best_p)
        
        # Evaluate on test
        try:
            strat = strategy_factory(**test_params)
            sig = strat.generate_signals(df_test)
            res = backtest_signals(df_test, sig, fee_bps=fee_bps, weight=weight)
            test_s = sharpe(res.returns, freq="1d")
            test_sharpes.append(float(test_s))
            
            # Collect detailed fold-level metrics
            fold_metric = {
                "fold": fold_idx,
                "train_start": df_train.index[0],
                "train_end": df_train.index[-1],
                "test_start": df_test.index[0],
                "test_end": df_test.index[-1],
                "train_size": len(df_train),
                "test_size": len(df_test),
                "best_params": best_p,
                "test_params": test_params if wfo.mc_param_noise else best_p,
                "train_sharpe": float(best_s),
                "test_sharpe": float(test_s),
                "test_mean_return": float(res.returns.mean()),
                "test_std_return": float(res.returns.std()),
                "test_total_return": float(res.equity_curve.iloc[-1] - 1.0),
                "test_max_equity": float(res.equity_curve.max()),
                "test_min_equity": float(res.equity_curve.min()),
            }
            fold_metrics.append(fold_metric)
            
        except Exception:
            continue
        
    mean_test_sharpe = float(np.mean(test_sharpes)) if test_sharpes else 0.0
    
    LOGGER.info(f"CPCV completed: {len(test_sharpes)} folds, mean Sharpe: {mean_test_sharpe:.2f}")
    
    result = {
        "test_sharpes": test_sharpes,
        "mean_test_sharpe": mean_test_sharpe,
        "best_params_per_split": best_params,
        "fold_metrics": fold_metrics,
    }
    
    # Calculate Sharpe stability metrics if enabled
    if wfo.sharpe_stability_check:
        stability = _calculate_sharpe_stability(test_sharpes)
        result["sharpe_stability"] = stability
    
    # Run Monte Carlo robustness testing if enabled
    if wfo.use_monte_carlo and best_params:
        from .monte_carlo import run_comprehensive_mc, MonteCarloConfig, evaluate_robustness
        
        LOGGER.info("Running Monte Carlo robustness testing on best parameters")
        
        final_params = best_params[-1]
        
        mc_config = MonteCarloConfig(
            n_simulations=wfo.mc_n_simulations,
            block_size=wfo.mc_block_size,
            param_perturbation_pct=wfo.mc_param_perturbation_pct
        )
        
        try:
            mc_results = run_comprehensive_mc(
                df=df,
                strategy_factory=strategy_factory,
                params=final_params,
                config=mc_config,
                fee_bps=fee_bps,
                weight=weight
            )
            
            robustness_eval = evaluate_robustness(mc_results)
            
            result["monte_carlo_results"] = mc_results
            result["robustness_evaluation"] = robustness_eval
            
            LOGGER.info(f"Monte Carlo testing completed: Robustness rating = {robustness_eval['rating']}")
            
        except Exception as e:
            LOGGER.error(f"Monte Carlo testing failed: {e}")
            result["monte_carlo_error"] = str(e)
    
    return result


def walk_forward_evaluate_regime_specific(
    df: pd.DataFrame,
    strategy_factory,
    param_grid: Dict[str, Iterable[Any]],
    fee_bps: float = 2.0,
    weight: float = 1.0,
    wfo: WFOSpec = WFOSpec(),
    regime_classifier: Optional[Callable[[pd.DataFrame], str]] = None,
) -> Dict[str, Any]:
    """
    Run WFO evaluation with regime-specific performance tracking.
    
    This enhanced version of walk_forward_evaluate tracks performance separately
    for different market regimes (trending, mean-reverting, volatile).
    
    Args:
        df: OHLCV DataFrame
        strategy_factory: Function that creates strategy instances from parameters
        param_grid: Dictionary of parameter ranges to optimize
        fee_bps: Trading fee in basis points
        weight: Position sizing weight
        wfo: WFO specification
        regime_classifier: Optional function that takes a DataFrame and returns
                          regime label ('trending', 'mean_reverting', 'volatile', 'neutral')
                          If None, will use RegimeDetector internally
    
    Returns:
        Dictionary with overall metrics plus regime-specific breakdowns:
            - test_sharpes: List of test Sharpe ratios per split
            - mean_test_sharpe: Mean test Sharpe
            - best_params_per_split: List of best parameters per split
            - regime_performance: Dict mapping regime -> performance metrics
            - regime_distribution: Dict mapping regime -> count/percentage
    """
    from ..quant.regime_detector import RegimeDetector, RegimeType
    from ..utils.logging import get_logger
    
    LOGGER = get_logger(__name__)
    
    if regime_classifier is None:
        regime_detector = RegimeDetector(lookback=100)
        def default_classifier(df_window: pd.DataFrame) -> str:
            regime_info = regime_detector.detect_regime(df_window)
            trend = regime_info['trend_regime']
            vol = regime_info['volatility_regime']
            hurst = regime_info['hurst_exponent']
            
            if vol == RegimeType.HIGH_VOLATILITY:
                return 'volatile'
            elif hurst > 0.55:
                return 'trending'
            elif hurst < 0.45:
                return 'mean_reverting'
            else:
                return 'neutral'
        regime_classifier = default_classifier
    
    idx = df.index
    splits = _split_windows(idx, wfo.n_splits)
    best_params: List[Dict[str, Any]] = []
    test_sharpes: List[float] = []
    
    regime_results = {
        'trending': {'sharpes': [], 'returns': [], 'periods': 0},
        'mean_reverting': {'sharpes': [], 'returns': [], 'periods': 0},
        'volatile': {'sharpes': [], 'returns': [], 'periods': 0},
        'neutral': {'sharpes': [], 'returns': [], 'periods': 0}
    }
    
    for (start, end) in splits:
        df_win = df[(df.index >= start) & (df.index <= end)]
        if df_win.empty:
            continue
            
        n = len(df_win)
        cut = int(n * wfo.train_frac)
        df_train = df_win.iloc[:cut]
        df_test = df_win.iloc[cut:]
        if len(df_test) < 5:
            continue
        
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
                s = sharpe(res.returns, freq="1d")
                if s > best_s:
                    best_s = s
                    best_p = params
            except Exception as e:
                LOGGER.debug(f"Strategy evaluation failed: {e}")
                continue
                
        if best_p is None:
            continue
            
        best_params.append(best_p)
        
        strat = strategy_factory(**best_p)
        sig = strat.generate_signals(df_test)
        res = backtest_signals(df_test, sig, fee_bps=fee_bps, weight=weight)
        test_s = sharpe(res.returns, freq="1d")
        test_sharpes.append(float(test_s))
        
        for i in range(len(df_test)):
            df_window = df_test.iloc[:i+1]
            if len(df_window) < 20:
                continue
                
            try:
                regime = regime_classifier(df_window)
                if regime in regime_results:
                    regime_results[regime]['periods'] += 1
                    ret = res.returns.iloc[i]
                    regime_results[regime]['returns'].append(ret)
            except Exception as e:
                LOGGER.debug(f"Regime classification failed: {e}")
                continue
    
    for regime, data in regime_results.items():
        if data['returns']:
            returns_series = pd.Series(data['returns'])
            regime_sharpe = sharpe(returns_series, freq="1d")
            data['sharpes'].append(regime_sharpe)
            data['mean_return'] = float(returns_series.mean())
            data['std_return'] = float(returns_series.std())
            data['sharpe'] = float(regime_sharpe)
        else:
            data['mean_return'] = 0.0
            data['std_return'] = 0.0
            data['sharpe'] = 0.0
    
    total_periods = sum(r['periods'] for r in regime_results.values())
    regime_distribution = {
        regime: {
            'count': data['periods'],
            'percentage': 100.0 * data['periods'] / total_periods if total_periods > 0 else 0.0
        }
        for regime, data in regime_results.items()
    }
    
    regime_performance = {
        regime: {
            'sharpe': data.get('sharpe', 0.0),
            'mean_return': data.get('mean_return', 0.0),
            'std_return': data.get('std_return', 0.0),
            'num_periods': data['periods']
        }
        for regime, data in regime_results.items()
    }
    
    mean_test_sharpe = float(np.mean(test_sharpes)) if test_sharpes else 0.0
    
    LOGGER.info(f"WFO completed: {len(test_sharpes)} folds, mean Sharpe: {mean_test_sharpe:.2f}")
    LOGGER.info(f"Regime distribution: {regime_distribution}")
    
    return {
        "test_sharpes": test_sharpes,
        "mean_test_sharpe": mean_test_sharpe,
        "best_params_per_split": best_params,
        "regime_performance": regime_performance,
        "regime_distribution": regime_distribution,
    }


def compare_strategies_by_regime(
    df: pd.DataFrame,
    strategies: Dict[str, Any],
    fee_bps: float = 2.0,
    weight: float = 1.0,
    regime_classifier: Optional[Callable[[pd.DataFrame], str]] = None,
) -> pd.DataFrame:
    """
    Compare multiple strategies and analyze their performance by regime.
    
    Args:
        df: OHLCV DataFrame
        strategies: Dict mapping strategy_name -> strategy_instance
        fee_bps: Trading fee in basis points
        weight: Position sizing weight
        regime_classifier: Optional function to classify regimes
    
    Returns:
        DataFrame with performance metrics by strategy and regime
    """
    from ..quant.regime_detector import RegimeDetector, RegimeType
    from ..utils.logging import get_logger
    
    LOGGER = get_logger(__name__)
    
    if regime_classifier is None:
        regime_detector = RegimeDetector(lookback=100)
        def default_classifier(df_window: pd.DataFrame) -> str:
            regime_info = regime_detector.detect_regime(df_window)
            trend = regime_info['trend_regime']
            vol = regime_info['volatility_regime']
            hurst = regime_info['hurst_exponent']
            
            if vol == RegimeType.HIGH_VOLATILITY:
                return 'volatile'
            elif hurst > 0.55:
                return 'trending'
            elif hurst < 0.45:
                return 'mean_reverting'
            else:
                return 'neutral'
        regime_classifier = default_classifier
    
    results = []
    
    for strat_name, strategy in strategies.items():
        try:
            signals = strategy.generate_signals(df)
            backtest_res = backtest_signals(df, signals, fee_bps=fee_bps, weight=weight)
            
            regime_returns = {
                'trending': [],
                'mean_reverting': [],
                'volatile': [],
                'neutral': []
            }
            
            for i in range(100, len(df)):
                df_window = df.iloc[:i+1]
                try:
                    regime = regime_classifier(df_window)
                    if regime in regime_returns:
                        regime_returns[regime].append(backtest_res.returns.iloc[i])
                except Exception:
                    continue
            
            overall_sharpe = sharpe(backtest_res.returns, freq="1d")
            
            results.append({
                'strategy': strat_name,
                'regime': 'overall',
                'sharpe': float(overall_sharpe),
                'mean_return': float(backtest_res.returns.mean()),
                'std_return': float(backtest_res.returns.std()),
                'num_periods': len(backtest_res.returns)
            })
            
            for regime, returns_list in regime_returns.items():
                if returns_list:
                    returns_series = pd.Series(returns_list)
                    regime_sharpe = sharpe(returns_series, freq="1d")
                    results.append({
                        'strategy': strat_name,
                        'regime': regime,
                        'sharpe': float(regime_sharpe),
                        'mean_return': float(returns_series.mean()),
                        'std_return': float(returns_series.std()),
                        'num_periods': len(returns_list)
                    })
                    
        except Exception as e:
            LOGGER.error(f"Strategy {strat_name} failed: {e}")
            continue
    
    return pd.DataFrame(results)
