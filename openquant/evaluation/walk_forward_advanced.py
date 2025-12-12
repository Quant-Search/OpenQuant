"""Advanced Walk-Forward Optimization with Anchored Expanding Window.

Implements sophisticated walk-forward optimization techniques including:
- Anchored expanding training window (accumulates more data over time)
- Combinatorial Purged Cross-Validation integration
- Monte Carlo parameter perturbation for robustness testing
- Out-of-sample Sharpe ratio stability analysis across folds
"""
from __future__ import annotations
from typing import Dict, Any, Iterable, List, Tuple, Callable, Optional
from dataclasses import dataclass, field
import itertools
import numpy as np
import pandas as pd

from ..backtest.engine import backtest_signals
from ..backtest.metrics import sharpe
from ..validation.combinatorial_cv import CombinatorialPurgedCV
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AdvancedWFOConfig:
    """Configuration for advanced walk-forward optimization."""
    
    # Window configuration
    n_splits: int = 6
    min_train_pct: float = 0.3  # Minimum training data percentage
    test_pct: float = 0.15  # Test window percentage
    expanding: bool = True  # Use expanding window (vs rolling)
    
    # CPCV configuration
    use_cpcv: bool = True
    cpcv_n_test_splits: int = 2
    cpcv_purge_pct: float = 0.02
    cpcv_embargo_pct: float = 0.01
    
    # Monte Carlo perturbation
    mc_perturbations: int = 10  # Number of parameter perturbations
    mc_noise_pct: float = 0.10  # ±10% noise on parameters
    
    # Stability analysis
    stability_threshold: float = 0.5  # Min Sharpe correlation across folds
    min_sharpe_ratio: float = 0.5  # Minimum acceptable Sharpe
    
    # Backtest settings
    fee_bps: float = 2.0
    weight: float = 1.0


@dataclass
class WFOFoldResult:
    """Results from a single walk-forward fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    mc_test_sharpes: List[float] = field(default_factory=list)
    mc_params: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AdvancedWFOResult:
    """Comprehensive results from advanced walk-forward optimization."""
    fold_results: List[WFOFoldResult]
    mean_test_sharpe: float
    median_test_sharpe: float
    std_test_sharpe: float
    sharpe_stability: float  # Coefficient of variation
    mc_mean_sharpe: float
    mc_std_sharpe: float
    mc_min_sharpe: float
    mc_max_sharpe: float
    is_stable: bool
    best_params_consensus: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "n_folds": len(self.fold_results),
            "mean_test_sharpe": self.mean_test_sharpe,
            "median_test_sharpe": self.median_test_sharpe,
            "std_test_sharpe": self.std_test_sharpe,
            "sharpe_stability": self.sharpe_stability,
            "mc_mean_sharpe": self.mc_mean_sharpe,
            "mc_std_sharpe": self.mc_std_sharpe,
            "mc_sharpe_range": (self.mc_min_sharpe, self.mc_max_sharpe),
            "is_stable": self.is_stable,
            "consensus_params": self.best_params_consensus,
            "warnings": self.warnings,
        }


def _generate_expanding_windows(
    index: pd.DatetimeIndex,
    n_splits: int,
    min_train_pct: float,
    test_pct: float
) -> List[Tuple[int, int, int, int]]:
    """
    Generate anchored expanding windows for walk-forward optimization.
    
    Args:
        index: Time series index
        n_splits: Number of walk-forward splits
        min_train_pct: Minimum training data percentage
        test_pct: Test window percentage
        
    Returns:
        List of (train_start, train_end, test_start, test_end) index tuples
    """
    n = len(index)
    min_train_size = int(n * min_train_pct)
    test_size = int(n * test_pct)
    
    if test_size < 5:
        test_size = min(5, n // 4)
    
    windows = []
    
    # Calculate total available data for walk-forward
    total_wf_size = n - min_train_size
    step_size = max(1, total_wf_size // n_splits)
    
    for i in range(n_splits):
        # Anchored start (always from beginning)
        train_start = 0
        
        # Expanding train end
        train_end = min_train_size + (i * step_size)
        
        # Test window follows immediately
        test_start = train_end
        test_end = min(test_start + test_size, n)
        
        # Ensure we have valid windows
        if test_end <= test_start or train_end <= train_start:
            continue
            
        if test_end - test_start < 5:  # Minimum test samples
            continue
            
        windows.append((train_start, train_end, test_start, test_end))
    
    return windows


def _perturb_params(
    params: Dict[str, Any],
    noise_pct: float,
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Apply Monte Carlo noise to parameters.
    
    Args:
        params: Original parameters
        noise_pct: Noise percentage (e.g., 0.10 for ±10%)
        rng: Random number generator
        
    Returns:
        Perturbed parameters
    """
    perturbed = {}
    
    for key, value in params.items():
        if isinstance(value, (int, float, np.number)):
            # Apply multiplicative noise
            noise_factor = 1.0 + rng.uniform(-noise_pct, noise_pct)
            new_value = value * noise_factor
            
            # Preserve type (int vs float)
            if isinstance(value, int):
                new_value = int(round(new_value))
                # Ensure at least 1 for positive integers
                if value > 0:
                    new_value = max(1, new_value)
            
            perturbed[key] = new_value
        else:
            # Keep non-numeric parameters unchanged
            perturbed[key] = value
    
    return perturbed


def _compute_sharpe_stability(sharpes: List[float]) -> float:
    """
    Compute stability metric for Sharpe ratios across folds.
    
    Uses coefficient of variation (inverse stability indicator).
    Lower CV = higher stability.
    
    Args:
        sharpes: List of Sharpe ratios
        
    Returns:
        Stability score (0-1, higher is more stable)
    """
    if not sharpes or len(sharpes) < 2:
        return 0.0
    
    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes, ddof=1)
    
    if abs(mean_sharpe) < 1e-6:
        return 0.0
    
    # Coefficient of variation
    cv = std_sharpe / abs(mean_sharpe)
    
    # Convert to stability score (lower CV = higher stability)
    # Use exponential decay: stability = exp(-cv)
    stability = np.exp(-cv)
    
    return float(stability)


def _find_consensus_params(
    fold_results: List[WFOFoldResult]
) -> Dict[str, Any]:
    """
    Find consensus best parameters across folds.
    
    Uses mode for categorical, median for numeric parameters.
    
    Args:
        fold_results: Results from all folds
        
    Returns:
        Consensus parameters
    """
    if not fold_results:
        return {}
    
    # Collect all parameter keys
    all_keys = set()
    for result in fold_results:
        all_keys.update(result.best_params.keys())
    
    consensus = {}
    
    for key in all_keys:
        values = [r.best_params.get(key) for r in fold_results if key in r.best_params]
        
        if not values:
            continue
        
        # Check if numeric
        if all(isinstance(v, (int, float, np.number)) for v in values):
            # Use median for numeric
            consensus[key] = np.median(values)
            
            # Convert back to int if all were int
            if all(isinstance(v, int) for v in values):
                consensus[key] = int(consensus[key])
        else:
            # Use mode for categorical
            from collections import Counter
            counter = Counter(values)
            consensus[key] = counter.most_common(1)[0][0]
    
    return consensus


def advanced_walk_forward_optimize(
    df: pd.DataFrame,
    strategy_factory: Callable,
    param_grid: Dict[str, Iterable[Any]],
    config: AdvancedWFOConfig = AdvancedWFOConfig(),
) -> AdvancedWFOResult:
    """
    Perform advanced walk-forward optimization with expanding window and stability checks.
    
    Args:
        df: OHLCV DataFrame with DatetimeIndex
        strategy_factory: Callable that takes **params and returns strategy instance
        param_grid: Dictionary of parameter names to iterable of values
        config: Configuration for optimization
        
    Returns:
        AdvancedWFOResult with comprehensive metrics and stability analysis
    """
    LOGGER.info("Starting Advanced Walk-Forward Optimization")
    LOGGER.info(f"Config: n_splits={config.n_splits}, expanding={config.expanding}, "
                f"use_cpcv={config.use_cpcv}, mc_perturbations={config.mc_perturbations}")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    fold_results: List[WFOFoldResult] = []
    all_test_sharpes: List[float] = []
    all_mc_sharpes: List[float] = []
    warnings: List[str] = []
    
    rng = np.random.default_rng(42)  # Reproducible random state
    
    # Generate expanding windows
    windows = _generate_expanding_windows(
        df.index,
        config.n_splits,
        config.min_train_pct,
        config.test_pct
    )
    
    LOGGER.info(f"Generated {len(windows)} walk-forward windows")
    
    if not windows:
        warnings.append("No valid windows generated - check data size and configuration")
        return AdvancedWFOResult(
            fold_results=[],
            mean_test_sharpe=0.0,
            median_test_sharpe=0.0,
            std_test_sharpe=0.0,
            sharpe_stability=0.0,
            mc_mean_sharpe=0.0,
            mc_std_sharpe=0.0,
            mc_min_sharpe=0.0,
            mc_max_sharpe=0.0,
            is_stable=False,
            best_params_consensus={},
            warnings=warnings
        )
    
    # Process each fold
    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(windows):
        LOGGER.info(f"Processing fold {fold_id + 1}/{len(windows)}")
        
        df_train = df.iloc[train_start:train_end]
        df_test = df.iloc[test_start:test_end]
        
        train_start_ts = df.index[train_start]
        train_end_ts = df.index[train_end - 1]
        test_start_ts = df.index[test_start]
        test_end_ts = df.index[test_end - 1]
        
        LOGGER.info(f"Train: {train_start_ts} to {train_end_ts} ({len(df_train)} samples)")
        LOGGER.info(f"Test:  {test_start_ts} to {test_end_ts} ({len(df_test)} samples)")
        
        # Optimize on training data
        if config.use_cpcv:
            best_params, train_sharpe = _optimize_with_cpcv(
                df_train, strategy_factory, param_grid, config
            )
        else:
            best_params, train_sharpe = _optimize_grid_search(
                df_train, strategy_factory, param_grid, config
            )
        
        if best_params is None:
            warnings.append(f"Fold {fold_id}: No valid parameters found during optimization")
            continue
        
        LOGGER.info(f"Best params: {best_params}, train Sharpe: {train_sharpe:.3f}")
        
        # Evaluate on test data
        try:
            strategy = strategy_factory(**best_params)
            signals = strategy.generate_signals(df_test)
            result = backtest_signals(df_test, signals, fee_bps=config.fee_bps, weight=config.weight)
            test_sharpe = sharpe(result.returns, freq="1d")
        except Exception as e:
            LOGGER.warning(f"Fold {fold_id}: Test evaluation failed: {e}")
            warnings.append(f"Fold {fold_id}: Test evaluation failed")
            continue
        
        LOGGER.info(f"Test Sharpe: {test_sharpe:.3f}")
        all_test_sharpes.append(test_sharpe)
        
        # Monte Carlo parameter perturbation
        mc_test_sharpes = []
        mc_params_list = []
        
        if config.mc_perturbations > 0:
            LOGGER.info(f"Running {config.mc_perturbations} Monte Carlo perturbations")
            
            for mc_iter in range(config.mc_perturbations):
                perturbed_params = _perturb_params(best_params, config.mc_noise_pct, rng)
                
                try:
                    strategy = strategy_factory(**perturbed_params)
                    signals = strategy.generate_signals(df_test)
                    result = backtest_signals(df_test, signals, fee_bps=config.fee_bps, weight=config.weight)
                    mc_sharpe = sharpe(result.returns, freq="1d")
                    
                    mc_test_sharpes.append(mc_sharpe)
                    mc_params_list.append(perturbed_params)
                    all_mc_sharpes.append(mc_sharpe)
                except Exception:
                    # Perturbed params might be invalid, skip silently
                    pass
            
            if mc_test_sharpes:
                LOGGER.info(f"MC Sharpe: mean={np.mean(mc_test_sharpes):.3f}, "
                           f"std={np.std(mc_test_sharpes):.3f}")
        
        # Store fold result
        fold_result = WFOFoldResult(
            fold_id=fold_id,
            train_start=train_start_ts,
            train_end=train_end_ts,
            test_start=test_start_ts,
            test_end=test_end_ts,
            best_params=best_params,
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            mc_test_sharpes=mc_test_sharpes,
            mc_params=mc_params_list
        )
        fold_results.append(fold_result)
    
    # Aggregate results
    if not all_test_sharpes:
        warnings.append("No successful fold evaluations - results are invalid")
        return AdvancedWFOResult(
            fold_results=fold_results,
            mean_test_sharpe=0.0,
            median_test_sharpe=0.0,
            std_test_sharpe=0.0,
            sharpe_stability=0.0,
            mc_mean_sharpe=0.0,
            mc_std_sharpe=0.0,
            mc_min_sharpe=0.0,
            mc_max_sharpe=0.0,
            is_stable=False,
            best_params_consensus={},
            warnings=warnings
        )
    
    mean_test_sharpe = float(np.mean(all_test_sharpes))
    median_test_sharpe = float(np.median(all_test_sharpes))
    std_test_sharpe = float(np.std(all_test_sharpes, ddof=1)) if len(all_test_sharpes) > 1 else 0.0
    
    # Stability analysis
    sharpe_stability = _compute_sharpe_stability(all_test_sharpes)
    
    # MC statistics
    mc_mean = float(np.mean(all_mc_sharpes)) if all_mc_sharpes else 0.0
    mc_std = float(np.std(all_mc_sharpes, ddof=1)) if len(all_mc_sharpes) > 1 else 0.0
    mc_min = float(np.min(all_mc_sharpes)) if all_mc_sharpes else 0.0
    mc_max = float(np.max(all_mc_sharpes)) if all_mc_sharpes else 0.0
    
    # Determine stability
    is_stable = (
        sharpe_stability >= config.stability_threshold
        and mean_test_sharpe >= config.min_sharpe_ratio
    )
    
    if not is_stable:
        if sharpe_stability < config.stability_threshold:
            warnings.append(f"Low Sharpe stability: {sharpe_stability:.3f} < {config.stability_threshold}")
        if mean_test_sharpe < config.min_sharpe_ratio:
            warnings.append(f"Low mean Sharpe: {mean_test_sharpe:.3f} < {config.min_sharpe_ratio}")
    
    # Find consensus parameters
    consensus_params = _find_consensus_params(fold_results)
    
    LOGGER.info(f"WFO Complete: mean_sharpe={mean_test_sharpe:.3f}, "
                f"stability={sharpe_stability:.3f}, is_stable={is_stable}")
    
    return AdvancedWFOResult(
        fold_results=fold_results,
        mean_test_sharpe=mean_test_sharpe,
        median_test_sharpe=median_test_sharpe,
        std_test_sharpe=std_test_sharpe,
        sharpe_stability=sharpe_stability,
        mc_mean_sharpe=mc_mean,
        mc_std_sharpe=mc_std,
        mc_min_sharpe=mc_min,
        mc_max_sharpe=mc_max,
        is_stable=is_stable,
        best_params_consensus=consensus_params,
        warnings=warnings
    )


def _optimize_grid_search(
    df_train: pd.DataFrame,
    strategy_factory: Callable,
    param_grid: Dict[str, Iterable[Any]],
    config: AdvancedWFOConfig
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Simple grid search optimization on training data.
    
    Returns:
        (best_params, best_sharpe) tuple
    """
    candidates = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())
    
    best_sharpe = -np.inf
    best_params = None
    
    for combo in candidates:
        params = dict(zip(keys, combo))
        
        try:
            strategy = strategy_factory(**params)
            signals = strategy.generate_signals(df_train)
            result = backtest_signals(df_train, signals, fee_bps=config.fee_bps, weight=config.weight)
            current_sharpe = sharpe(result.returns, freq="1d")
            
            if current_sharpe > best_sharpe:
                best_sharpe = current_sharpe
                best_params = params
        except Exception:
            continue
    
    return best_params, float(best_sharpe) if best_params else 0.0


def _optimize_with_cpcv(
    df_train: pd.DataFrame,
    strategy_factory: Callable,
    param_grid: Dict[str, Iterable[Any]],
    config: AdvancedWFOConfig
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Optimize using Combinatorial Purged Cross-Validation on training data.
    
    Returns:
        (best_params, best_mean_cv_sharpe) tuple
    """
    cv = CombinatorialPurgedCV(
        n_splits=min(5, len(df_train) // 20),  # Adaptive splits
        n_test_splits=config.cpcv_n_test_splits,
        purge_pct=config.cpcv_purge_pct,
        embargo_pct=config.cpcv_embargo_pct
    )
    
    candidates = list(itertools.product(*param_grid.values()))
    keys = list(param_grid.keys())
    
    best_mean_sharpe = -np.inf
    best_params = None
    
    for combo in candidates:
        params = dict(zip(keys, combo))
        cv_sharpes = []
        
        try:
            for train_idx, val_idx in cv.split(df_train):
                if len(train_idx) < 10 or len(val_idx) < 5:
                    continue
                
                df_cv_train = df_train.iloc[train_idx]
                df_cv_val = df_train.iloc[val_idx]
                
                strategy = strategy_factory(**params)
                signals = strategy.generate_signals(df_cv_val)
                result = backtest_signals(df_cv_val, signals, fee_bps=config.fee_bps, weight=config.weight)
                cv_sharpe = sharpe(result.returns, freq="1d")
                cv_sharpes.append(cv_sharpe)
            
            if cv_sharpes:
                mean_cv_sharpe = np.mean(cv_sharpes)
                
                if mean_cv_sharpe > best_mean_sharpe:
                    best_mean_sharpe = mean_cv_sharpe
                    best_params = params
        except Exception:
            continue
    
    return best_params, float(best_mean_sharpe) if best_params else 0.0


def plot_wfo_results(result: AdvancedWFOResult, save_path: Optional[str] = None):
    """
    Plot walk-forward optimization results.
    
    Args:
        result: WFO result object
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available - skipping plot")
        return
    
    if not result.fold_results:
        LOGGER.warning("No fold results to plot")
        return
    
    fold_ids = [r.fold_id for r in result.fold_results]
    train_sharpes = [r.train_sharpe for r in result.fold_results]
    test_sharpes = [r.test_sharpe for r in result.fold_results]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Sharpe ratios by fold
    axes[0].plot(fold_ids, train_sharpes, 'o-', label='Train Sharpe', alpha=0.7)
    axes[0].plot(fold_ids, test_sharpes, 's-', label='Test Sharpe', alpha=0.7)
    axes[0].axhline(y=result.mean_test_sharpe, color='r', linestyle='--', 
                    label=f'Mean Test Sharpe: {result.mean_test_sharpe:.2f}')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].set_title('Walk-Forward Sharpe Ratios by Fold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Monte Carlo distribution
    all_mc = []
    for fr in result.fold_results:
        all_mc.extend(fr.mc_test_sharpes)
    
    if all_mc:
        axes[1].hist(all_mc, bins=30, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=result.mc_mean_sharpe, color='r', linestyle='--',
                       label=f'Mean: {result.mc_mean_sharpe:.2f}')
        axes[1].axvline(x=result.mean_test_sharpe, color='g', linestyle='--',
                       label=f'Base: {result.mean_test_sharpe:.2f}')
        axes[1].set_xlabel('Sharpe Ratio')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Monte Carlo Parameter Perturbation Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        LOGGER.info(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
