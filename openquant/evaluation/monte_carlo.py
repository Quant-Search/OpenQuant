"""Monte Carlo Simulation for Strategy Robustness Testing.

Implements three types of Monte Carlo analysis:
1. Path-Dependent Randomization: Bootstrap with block structure preservation
2. Parameter Perturbation: Test strategy sensitivity to parameter changes
3. Regime Shift Simulation: Test strategy under different market conditions

Integrates with WFO pipeline for comprehensive robustness validation.
"""
from __future__ import annotations
from typing import Dict, Any, Iterable, List, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from copy import deepcopy

from ..backtest.engine import backtest_signals
from ..backtest.metrics import sharpe, sortino, max_drawdown
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class MonteCarloConfig:
    n_simulations: int = 500
    block_size: int = 20
    confidence_level: float = 0.95
    param_perturbation_pct: float = 0.1
    regime_shift_volatility_multipliers: List[float] = None
    regime_shift_trend_multipliers: List[float] = None
    
    def __post_init__(self):
        if self.regime_shift_volatility_multipliers is None:
            self.regime_shift_volatility_multipliers = [0.5, 1.0, 1.5, 2.0]
        if self.regime_shift_trend_multipliers is None:
            self.regime_shift_trend_multipliers = [0.0, 0.5, 1.0, 1.5]


@dataclass
class MonteCarloResult:
    metric: str
    mean: float
    median: float
    std: float
    percentile_5: float
    percentile_95: float
    min: float
    max: float
    simulations: List[float]
    
    def is_robust(self, threshold: float = 0.0) -> bool:
        return self.percentile_5 > threshold


class PathDependentBootstrap:
    """Path-dependent bootstrap preserving temporal structure using block bootstrap."""
    
    def __init__(self, block_size: int = 20, preserve_trends: bool = True):
        self.block_size = block_size
        self.preserve_trends = preserve_trends
        
    def resample_returns(self, returns: pd.Series, seed: Optional[int] = None) -> pd.Series:
        """Resample returns using block bootstrap to preserve autocorrelation."""
        if seed is not None:
            np.random.seed(seed)
            
        ret_arr = returns.dropna().values
        n = len(ret_arr)
        
        if n == 0:
            return returns
            
        n_blocks = max(1, n // self.block_size)
        resampled = []
        
        for _ in range(n_blocks):
            if len(ret_arr) < self.block_size:
                start_idx = 0
                block = ret_arr[start_idx:]
            else:
                start_idx = np.random.randint(0, len(ret_arr) - self.block_size + 1)
                block = ret_arr[start_idx:start_idx + self.block_size]
            resampled.extend(block)
            
        resampled = np.array(resampled[:n])
        return pd.Series(resampled, index=returns.index)
        
    def resample_prices(self, df: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
        """Resample OHLCV data using block bootstrap on returns, then reconstruct prices."""
        if seed is not None:
            np.random.seed(seed)
            
        df_out = df.copy()
        
        close = df['Close'].values
        returns = np.diff(np.log(close))
        returns = np.concatenate([[0.0], returns])
        
        n = len(returns)
        n_blocks = max(1, n // self.block_size)
        resampled_returns = []
        
        for _ in range(n_blocks):
            if len(returns) < self.block_size:
                start_idx = 0
                block = returns[start_idx:]
            else:
                start_idx = np.random.randint(0, len(returns) - self.block_size + 1)
                block = returns[start_idx:start_idx + self.block_size]
            resampled_returns.extend(block)
            
        resampled_returns = np.array(resampled_returns[:n])
        
        resampled_prices = close[0] * np.exp(np.cumsum(resampled_returns))
        
        ratio = resampled_prices / close
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_out.columns:
                df_out[col] = df_out[col] * ratio
                
        return df_out


class ParameterPerturbation:
    """Perturb strategy parameters to test sensitivity and robustness."""
    
    def __init__(self, perturbation_pct: float = 0.1):
        self.perturbation_pct = perturbation_pct
        
    def perturb_params(self, params: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
        """Add random noise to numeric parameters."""
        if seed is not None:
            np.random.seed(seed)
            
        perturbed = deepcopy(params)
        
        for key, value in params.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                noise = np.random.uniform(-self.perturbation_pct, self.perturbation_pct)
                new_value = value * (1.0 + noise)
                
                if isinstance(value, int):
                    new_value = max(1, int(round(new_value)))
                    
                perturbed[key] = new_value
                
        return perturbed
        
    def generate_perturbation_grid(
        self, 
        params: Dict[str, Any], 
        n_perturbations: int = 50
    ) -> List[Dict[str, Any]]:
        """Generate multiple parameter perturbations."""
        perturbations = []
        for i in range(n_perturbations):
            perturbations.append(self.perturb_params(params, seed=i))
        return perturbations


class RegimeShiftSimulator:
    """Simulate strategy performance under different market regimes."""
    
    def __init__(
        self, 
        volatility_multipliers: List[float] = None,
        trend_multipliers: List[float] = None
    ):
        self.volatility_multipliers = volatility_multipliers or [0.5, 1.0, 1.5, 2.0]
        self.trend_multipliers = trend_multipliers or [0.0, 0.5, 1.0, 1.5]
        
    def apply_volatility_regime(
        self, 
        df: pd.DataFrame, 
        multiplier: float
    ) -> pd.DataFrame:
        """Scale volatility by multiplier while preserving mean returns."""
        df_out = df.copy()
        close = df['Close'].values
        returns = np.diff(np.log(close))
        returns = np.concatenate([[0.0], returns])
        
        mean_return = np.mean(returns)
        demeaned = returns - mean_return
        scaled = demeaned * multiplier
        scaled_returns = scaled + mean_return
        
        new_prices = close[0] * np.exp(np.cumsum(scaled_returns))
        ratio = new_prices / close
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_out.columns:
                df_out[col] = df_out[col] * ratio
                
        return df_out
        
    def apply_trend_regime(
        self,
        df: pd.DataFrame,
        multiplier: float
    ) -> pd.DataFrame:
        """Add synthetic trend to prices."""
        df_out = df.copy()
        n = len(df)
        
        trend = np.linspace(0, multiplier * 0.01, n)
        trend_factor = np.exp(trend)
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df_out.columns:
                df_out[col] = df_out[col] * trend_factor
                
        return df_out
        
    def generate_regime_variants(
        self,
        df: pd.DataFrame
    ) -> List[Tuple[str, pd.DataFrame]]:
        """Generate multiple regime-shifted datasets."""
        variants = []
        
        for vol_mult in self.volatility_multipliers:
            df_vol = self.apply_volatility_regime(df, vol_mult)
            variants.append((f"vol_{vol_mult:.1f}x", df_vol))
            
        for trend_mult in self.trend_multipliers:
            df_trend = self.apply_trend_regime(df, trend_mult)
            variants.append((f"trend_{trend_mult:.1f}x", df_trend))
            
        for vol_mult in [0.5, 2.0]:
            for trend_mult in [0.5, 1.5]:
                df_combined = self.apply_volatility_regime(df, vol_mult)
                df_combined = self.apply_trend_regime(df_combined, trend_mult)
                variants.append((f"vol_{vol_mult:.1f}x_trend_{trend_mult:.1f}x", df_combined))
                
        return variants


def run_path_dependent_mc(
    df: pd.DataFrame,
    strategy_factory: Callable,
    params: Dict[str, Any],
    config: MonteCarloConfig,
    fee_bps: float = 2.0,
    weight: float = 1.0
) -> Dict[str, MonteCarloResult]:
    """Run path-dependent Monte Carlo simulation."""
    LOGGER.info(f"Running path-dependent MC with {config.n_simulations} simulations")
    
    bootstrap = PathDependentBootstrap(block_size=config.block_size)
    
    sharpe_results = []
    sortino_results = []
    max_dd_results = []
    
    for i in range(config.n_simulations):
        try:
            df_resampled = bootstrap.resample_prices(df, seed=i)
            
            strategy = strategy_factory(**params)
            signals = strategy.generate_signals(df_resampled)
            result = backtest_signals(df_resampled, signals, fee_bps=fee_bps, weight=weight)
            
            sharpe_results.append(sharpe(result.returns, freq="1d"))
            sortino_results.append(sortino(result.returns, freq="1d"))
            max_dd_results.append(abs(max_drawdown(result.equity_curve)))
            
        except Exception as e:
            LOGGER.warning(f"Simulation {i} failed: {e}")
            continue
            
    results = {}
    
    for metric_name, values in [
        ("sharpe", sharpe_results),
        ("sortino", sortino_results),
        ("max_drawdown", max_dd_results)
    ]:
        if values:
            results[metric_name] = MonteCarloResult(
                metric=metric_name,
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                std=float(np.std(values)),
                percentile_5=float(np.percentile(values, 5)),
                percentile_95=float(np.percentile(values, 95)),
                min=float(np.min(values)),
                max=float(np.max(values)),
                simulations=values
            )
            
    LOGGER.info(f"Path-dependent MC completed: {len(sharpe_results)} successful simulations")
    return results


def run_parameter_perturbation_mc(
    df: pd.DataFrame,
    strategy_factory: Callable,
    params: Dict[str, Any],
    config: MonteCarloConfig,
    fee_bps: float = 2.0,
    weight: float = 1.0
) -> Dict[str, MonteCarloResult]:
    """Run parameter perturbation Monte Carlo simulation."""
    LOGGER.info(f"Running parameter perturbation MC with {config.n_simulations} simulations")
    
    perturbator = ParameterPerturbation(perturbation_pct=config.param_perturbation_pct)
    perturbations = perturbator.generate_perturbation_grid(params, config.n_simulations)
    
    sharpe_results = []
    sortino_results = []
    max_dd_results = []
    
    for i, perturbed_params in enumerate(perturbations):
        try:
            strategy = strategy_factory(**perturbed_params)
            signals = strategy.generate_signals(df)
            result = backtest_signals(df, signals, fee_bps=fee_bps, weight=weight)
            
            sharpe_results.append(sharpe(result.returns, freq="1d"))
            sortino_results.append(sortino(result.returns, freq="1d"))
            max_dd_results.append(abs(max_drawdown(result.equity_curve)))
            
        except Exception as e:
            LOGGER.warning(f"Perturbation {i} failed: {e}")
            continue
            
    results = {}
    
    for metric_name, values in [
        ("sharpe", sharpe_results),
        ("sortino", sortino_results),
        ("max_drawdown", max_dd_results)
    ]:
        if values:
            results[metric_name] = MonteCarloResult(
                metric=metric_name,
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                std=float(np.std(values)),
                percentile_5=float(np.percentile(values, 5)),
                percentile_95=float(np.percentile(values, 95)),
                min=float(np.min(values)),
                max=float(np.max(values)),
                simulations=values
            )
            
    LOGGER.info(f"Parameter perturbation MC completed: {len(sharpe_results)} successful simulations")
    return results


def run_regime_shift_mc(
    df: pd.DataFrame,
    strategy_factory: Callable,
    params: Dict[str, Any],
    config: MonteCarloConfig,
    fee_bps: float = 2.0,
    weight: float = 1.0
) -> Dict[str, Dict[str, MonteCarloResult]]:
    """Run regime shift Monte Carlo simulation."""
    LOGGER.info("Running regime shift MC across multiple regimes")
    
    simulator = RegimeShiftSimulator(
        volatility_multipliers=config.regime_shift_volatility_multipliers,
        trend_multipliers=config.regime_shift_trend_multipliers
    )
    
    regime_variants = simulator.generate_regime_variants(df)
    all_results = {}
    
    for regime_name, df_regime in regime_variants:
        sharpe_results = []
        sortino_results = []
        max_dd_results = []
        
        try:
            strategy = strategy_factory(**params)
            signals = strategy.generate_signals(df_regime)
            result = backtest_signals(df_regime, signals, fee_bps=fee_bps, weight=weight)
            
            sharpe_val = sharpe(result.returns, freq="1d")
            sortino_val = sortino(result.returns, freq="1d")
            max_dd_val = abs(max_drawdown(result.equity_curve))
            
            all_results[regime_name] = {
                "sharpe": MonteCarloResult(
                    metric="sharpe",
                    mean=sharpe_val,
                    median=sharpe_val,
                    std=0.0,
                    percentile_5=sharpe_val,
                    percentile_95=sharpe_val,
                    min=sharpe_val,
                    max=sharpe_val,
                    simulations=[sharpe_val]
                ),
                "sortino": MonteCarloResult(
                    metric="sortino",
                    mean=sortino_val,
                    median=sortino_val,
                    std=0.0,
                    percentile_5=sortino_val,
                    percentile_95=sortino_val,
                    min=sortino_val,
                    max=sortino_val,
                    simulations=[sortino_val]
                ),
                "max_drawdown": MonteCarloResult(
                    metric="max_drawdown",
                    mean=max_dd_val,
                    median=max_dd_val,
                    std=0.0,
                    percentile_5=max_dd_val,
                    percentile_95=max_dd_val,
                    min=max_dd_val,
                    max=max_dd_val,
                    simulations=[max_dd_val]
                )
            }
            
        except Exception as e:
            LOGGER.warning(f"Regime {regime_name} failed: {e}")
            continue
            
    LOGGER.info(f"Regime shift MC completed: {len(all_results)} regimes tested")
    return all_results


def run_comprehensive_mc(
    df: pd.DataFrame,
    strategy_factory: Callable,
    params: Dict[str, Any],
    config: MonteCarloConfig = None,
    fee_bps: float = 2.0,
    weight: float = 1.0,
    run_path_dependent: bool = True,
    run_param_perturbation: bool = True,
    run_regime_shift: bool = True
) -> Dict[str, Any]:
    """Run comprehensive Monte Carlo analysis combining all three methods."""
    if config is None:
        config = MonteCarloConfig()
        
    LOGGER.info("Starting comprehensive Monte Carlo robustness testing")
    
    results = {}
    
    if run_path_dependent:
        results["path_dependent"] = run_path_dependent_mc(
            df, strategy_factory, params, config, fee_bps, weight
        )
        
    if run_param_perturbation:
        results["parameter_perturbation"] = run_parameter_perturbation_mc(
            df, strategy_factory, params, config, fee_bps, weight
        )
        
    if run_regime_shift:
        results["regime_shift"] = run_regime_shift_mc(
            df, strategy_factory, params, config, fee_bps, weight
        )
        
    overall_sharpe_values = []
    if "path_dependent" in results and "sharpe" in results["path_dependent"]:
        overall_sharpe_values.extend(results["path_dependent"]["sharpe"].simulations)
    if "parameter_perturbation" in results and "sharpe" in results["parameter_perturbation"]:
        overall_sharpe_values.extend(results["parameter_perturbation"]["sharpe"].simulations)
    if "regime_shift" in results:
        for regime_results in results["regime_shift"].values():
            if "sharpe" in regime_results:
                overall_sharpe_values.extend(regime_results["sharpe"].simulations)
                
    if overall_sharpe_values:
        results["overall_summary"] = {
            "sharpe_mean": float(np.mean(overall_sharpe_values)),
            "sharpe_5th_percentile": float(np.percentile(overall_sharpe_values, 5)),
            "sharpe_95th_percentile": float(np.percentile(overall_sharpe_values, 95)),
            "is_robust": float(np.percentile(overall_sharpe_values, 5)) > 0.0,
            "total_simulations": len(overall_sharpe_values)
        }
        
    LOGGER.info("Comprehensive Monte Carlo testing completed")
    return results


def evaluate_robustness(mc_results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate overall strategy robustness from MC results."""
    robustness_score = 0.0
    max_score = 0.0
    details = {}
    
    if "path_dependent" in mc_results and "sharpe" in mc_results["path_dependent"]:
        path_result = mc_results["path_dependent"]["sharpe"]
        if path_result.percentile_5 > 0.5:
            robustness_score += 3
        elif path_result.percentile_5 > 0.0:
            robustness_score += 2
        elif path_result.mean > 0.0:
            robustness_score += 1
        max_score += 3
        details["path_dependent_robust"] = path_result.is_robust(threshold=0.0)
        
    if "parameter_perturbation" in mc_results and "sharpe" in mc_results["parameter_perturbation"]:
        param_result = mc_results["parameter_perturbation"]["sharpe"]
        if param_result.percentile_5 > 0.5:
            robustness_score += 3
        elif param_result.percentile_5 > 0.0:
            robustness_score += 2
        elif param_result.mean > 0.0:
            robustness_score += 1
        max_score += 3
        details["parameter_sensitivity_robust"] = param_result.is_robust(threshold=0.0)
        
    if "regime_shift" in mc_results:
        regime_count = 0
        positive_regimes = 0
        for regime_results in mc_results["regime_shift"].values():
            if "sharpe" in regime_results:
                regime_count += 1
                if regime_results["sharpe"].mean > 0.0:
                    positive_regimes += 1
        if regime_count > 0:
            regime_ratio = positive_regimes / regime_count
            if regime_ratio > 0.8:
                robustness_score += 3
            elif regime_ratio > 0.6:
                robustness_score += 2
            elif regime_ratio > 0.4:
                robustness_score += 1
            max_score += 3
            details["regime_adaptability_score"] = regime_ratio
            
    normalized_score = (robustness_score / max_score) if max_score > 0 else 0.0
    
    if normalized_score >= 0.8:
        rating = "HIGHLY_ROBUST"
    elif normalized_score >= 0.6:
        rating = "ROBUST"
    elif normalized_score >= 0.4:
        rating = "MODERATE"
    else:
        rating = "FRAGILE"
        
    return {
        "robustness_score": normalized_score,
        "rating": rating,
        "details": details
    }
