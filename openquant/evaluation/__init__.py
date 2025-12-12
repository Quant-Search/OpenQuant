"""Evaluation module for strategy testing and validation."""

from .wfo import (
    walk_forward_evaluate,
    WFOSpec,
    walk_forward_evaluate_regime_specific,
    compare_strategies_by_regime,
)
from .monte_carlo import (
    run_comprehensive_mc,
    run_path_dependent_mc,
    run_parameter_perturbation_mc,
    run_regime_shift_mc,
    evaluate_robustness,
    MonteCarloConfig,
    MonteCarloResult,
    PathDependentBootstrap,
    ParameterPerturbation,
    RegimeShiftSimulator,
)
from .regime import compute_regime_features
from .deflated_sharpe import deflated_sharpe_ratio
from .purged_kfold import purged_kfold_indices

__all__ = [
    "walk_forward_evaluate",
    "WFOSpec",
    "walk_forward_evaluate_regime_specific",
    "compare_strategies_by_regime",
    "run_comprehensive_mc",
    "run_path_dependent_mc",
    "run_parameter_perturbation_mc",
    "run_regime_shift_mc",
    "evaluate_robustness",
    "MonteCarloConfig",
    "MonteCarloResult",
    "PathDependentBootstrap",
    "ParameterPerturbation",
    "RegimeShiftSimulator",
    "compute_regime_features",
    "deflated_sharpe_ratio",
    "purged_kfold_indices",
]
