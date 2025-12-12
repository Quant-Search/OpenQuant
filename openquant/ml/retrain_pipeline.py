"""Automated Model Retraining Pipeline.

Monitors live strategy performance using deflated Sharpe ratio, triggers retraining
when DSR drops below threshold, runs walk-forward optimization with latest data,
validates new model against holdout set, and deploys only if improvement >10%.
"""
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import threading
import numpy as np
import pandas as pd

from ..evaluation.deflated_sharpe import deflated_sharpe_ratio
from ..evaluation.wfo import walk_forward_evaluate, WFOSpec
from ..backtest.engine import backtest_signals
from ..backtest.metrics import sharpe
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining pipeline."""
    dsr_threshold: float = 1.0
    improvement_threshold: float = 0.10
    min_samples_retrain: int = 500
    lookback_window: int = 1000
    holdout_fraction: float = 0.15
    monitoring_interval_hours: int = 6
    max_retrain_per_day: int = 4
    trials_per_strategy: int = 50
    wfo_n_splits: int = 4
    wfo_train_frac: float = 0.7
    enable_cpcv: bool = False
    model_save_dir: Path = Path("data/models")
    metrics_save_dir: Path = Path("data/retrain_metrics")


@dataclass
class PerformanceMetrics:
    """Performance metrics for a strategy."""
    timestamp: datetime
    sharpe: float
    dsr: float
    num_observations: int
    equity_curve: pd.Series
    returns: pd.Series


@dataclass
class RetrainingEvent:
    """Record of a retraining event."""
    timestamp: datetime
    trigger_reason: str
    old_dsr: float
    old_sharpe: float
    new_dsr: float
    new_sharpe: float
    improvement: float
    deployed: bool
    validation_metrics: Dict[str, Any]
    new_params: Dict[str, Any]


class PerformanceMonitor:
    """Monitors live strategy performance and calculates DSR."""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.performance_history: Dict[str, list[PerformanceMetrics]] = {}
        self._lock = threading.Lock()
        
    def update_performance(
        self,
        strategy_id: str,
        equity_curve: pd.Series,
        returns: pd.Series,
        num_trials: int = 1
    ) -> PerformanceMetrics:
        """Update performance metrics for a strategy."""
        with self._lock:
            # Calculate metrics
            s = float(sharpe(returns, freq="1h"))
            T = len(returns.dropna())
            dsr = float(deflated_sharpe_ratio(s, T=T, trials=num_trials))
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                sharpe=s,
                dsr=dsr,
                num_observations=T,
                equity_curve=equity_curve.copy(),
                returns=returns.copy()
            )
            
            if strategy_id not in self.performance_history:
                self.performance_history[strategy_id] = []
            self.performance_history[strategy_id].append(metrics)
            
            LOGGER.info(
                f"Performance updated for {strategy_id}: "
                f"Sharpe={s:.2f}, DSR={dsr:.2f}, Samples={T}"
            )
            
            return metrics
    
    def get_latest_metrics(self, strategy_id: str) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        with self._lock:
            if strategy_id not in self.performance_history:
                return None
            if not self.performance_history[strategy_id]:
                return None
            return self.performance_history[strategy_id][-1]
    
    def should_retrain(
        self,
        strategy_id: str,
        last_retrain_time: Optional[datetime] = None
    ) -> tuple[bool, str]:
        """Check if strategy should be retrained."""
        metrics = self.get_latest_metrics(strategy_id)
        if metrics is None:
            return False, "No metrics available"
        
        # Check minimum sample size
        if metrics.num_observations < self.config.min_samples_retrain:
            return False, f"Insufficient samples ({metrics.num_observations} < {self.config.min_samples_retrain})"
        
        # Check DSR threshold
        if metrics.dsr < self.config.dsr_threshold:
            return True, f"DSR below threshold ({metrics.dsr:.2f} < {self.config.dsr_threshold})"
        
        # Check rate limiting
        if last_retrain_time is not None:
            hours_since_retrain = (datetime.now() - last_retrain_time).total_seconds() / 3600
            if hours_since_retrain < self.config.monitoring_interval_hours:
                return False, f"Too soon since last retrain ({hours_since_retrain:.1f}h < {self.config.monitoring_interval_hours}h)"
        
        return False, "Performance acceptable"


class ModelValidator:
    """Validates new models against holdout data."""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
    
    def validate_model(
        self,
        df: pd.DataFrame,
        strategy_factory: Callable,
        old_params: Dict[str, Any],
        new_params: Dict[str, Any],
        fee_bps: float = 2.0,
        weight: float = 1.0
    ) -> tuple[bool, Dict[str, Any]]:
        """Validate new model against holdout set.
        
        Returns:
            (should_deploy, validation_metrics)
        """
        # Split data: train/val for new model, holdout for final validation
        n = len(df)
        holdout_size = int(n * self.config.holdout_fraction)
        df_train_val = df.iloc[:-holdout_size]
        df_holdout = df.iloc[-holdout_size:]
        
        LOGGER.info(
            f"Validation split: train/val={len(df_train_val)}, holdout={len(df_holdout)}"
        )
        
        # Evaluate old model on holdout
        try:
            old_strat = strategy_factory(**old_params)
            old_signals = old_strat.generate_signals(df_holdout)
            old_result = backtest_signals(df_holdout, old_signals, fee_bps=fee_bps, weight=weight)
            old_sharpe = float(sharpe(old_result.returns, freq="1h"))
            old_T = len(old_result.returns.dropna())
            old_dsr = float(deflated_sharpe_ratio(old_sharpe, T=old_T, trials=1))
        except Exception as e:
            LOGGER.error(f"Old model evaluation failed: {e}")
            old_sharpe = 0.0
            old_dsr = 0.0
        
        # Evaluate new model on holdout
        try:
            new_strat = strategy_factory(**new_params)
            new_signals = new_strat.generate_signals(df_holdout)
            new_result = backtest_signals(df_holdout, new_signals, fee_bps=fee_bps, weight=weight)
            new_sharpe = float(sharpe(new_result.returns, freq="1h"))
            new_T = len(new_result.returns.dropna())
            new_dsr = float(deflated_sharpe_ratio(new_sharpe, T=new_T, trials=1))
        except Exception as e:
            LOGGER.error(f"New model evaluation failed: {e}")
            return False, {
                "error": str(e),
                "old_sharpe": old_sharpe,
                "old_dsr": old_dsr
            }
        
        # Calculate improvement
        if old_sharpe <= 0:
            improvement = float('inf') if new_sharpe > 0 else 0.0
        else:
            improvement = (new_sharpe - old_sharpe) / abs(old_sharpe)
        
        validation_metrics = {
            "old_sharpe": old_sharpe,
            "old_dsr": old_dsr,
            "new_sharpe": new_sharpe,
            "new_dsr": new_dsr,
            "improvement": improvement,
            "holdout_samples": len(df_holdout),
            "threshold": self.config.improvement_threshold
        }
        
        should_deploy = improvement > self.config.improvement_threshold
        
        LOGGER.info(
            f"Validation results: old_sharpe={old_sharpe:.2f}, new_sharpe={new_sharpe:.2f}, "
            f"improvement={improvement:.2%}, deploy={should_deploy}"
        )
        
        return should_deploy, validation_metrics


class ModelRetrainer:
    """Handles model retraining using walk-forward optimization."""
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
    
    def retrain(
        self,
        df: pd.DataFrame,
        strategy_factory: Callable,
        param_grid: Dict[str, Iterable[Any]],
        fee_bps: float = 2.0,
        weight: float = 1.0
    ) -> Dict[str, Any]:
        """Retrain model using walk-forward optimization.
        
        Returns:
            Dictionary with best_params and wfo_results
        """
        # Use only recent data for retraining
        if len(df) > self.config.lookback_window:
            df_retrain = df.iloc[-self.config.lookback_window:]
            LOGGER.info(f"Using last {self.config.lookback_window} samples for retraining")
        else:
            df_retrain = df
        
        # Reserve holdout data
        holdout_size = int(len(df_retrain) * self.config.holdout_fraction)
        df_wfo = df_retrain.iloc[:-holdout_size] if holdout_size > 0 else df_retrain
        
        LOGGER.info(f"WFO on {len(df_wfo)} samples with {self.config.wfo_n_splits} splits")
        
        # Run walk-forward optimization
        wfo_spec = WFOSpec(
            n_splits=self.config.wfo_n_splits,
            train_frac=self.config.wfo_train_frac,
            use_cpcv=self.config.enable_cpcv
        )
        
        try:
            wfo_results = walk_forward_evaluate(
                df=df_wfo,
                strategy_factory=strategy_factory,
                param_grid=param_grid,
                fee_bps=fee_bps,
                weight=weight,
                wfo=wfo_spec
            )
        except Exception as e:
            LOGGER.error(f"WFO failed: {e}")
            raise
        
        # Get best params (most common across splits)
        best_params_list = wfo_results.get("best_params_per_split", [])
        if not best_params_list:
            raise ValueError("No valid parameters found during WFO")
        
        # Vote for most common parameter combination
        best_params = self._vote_best_params(best_params_list)
        
        LOGGER.info(
            f"WFO completed: mean_test_sharpe={wfo_results['mean_test_sharpe']:.2f}, "
            f"best_params={best_params}"
        )
        
        return {
            "best_params": best_params,
            "wfo_results": wfo_results
        }
    
    def _vote_best_params(self, params_list: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Vote for best parameter combination across WFO splits."""
        if not params_list:
            return {}
        
        # Convert to hashable form for voting
        param_tuples = [tuple(sorted(p.items())) for p in params_list]
        
        # Count occurrences
        from collections import Counter
        counter = Counter(param_tuples)
        most_common = counter.most_common(1)[0][0]
        
        # Convert back to dict
        return dict(most_common)


class RetrainingPipeline:
    """Automated model retraining pipeline.
    
    Main orchestrator that combines monitoring, retraining, and validation.
    """
    
    def __init__(self, config: Optional[RetrainingConfig] = None):
        self.config = config or RetrainingConfig()
        self.monitor = PerformanceMonitor(self.config)
        self.validator = ModelValidator(self.config)
        self.retrainer = ModelRetrainer(self.config)
        
        self.last_retrain_time: Dict[str, datetime] = {}
        self.retrain_history: Dict[str, list[RetrainingEvent]] = {}
        self._lock = threading.Lock()
        
        # Ensure directories exist
        self.config.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.config.metrics_save_dir.mkdir(parents=True, exist_ok=True)
    
    def update_and_check(
        self,
        strategy_id: str,
        equity_curve: pd.Series,
        returns: pd.Series,
        num_trials: int = 1
    ) -> tuple[bool, str]:
        """Update performance and check if retraining is needed.
        
        Returns:
            (should_retrain, reason)
        """
        # Update performance metrics
        self.monitor.update_performance(strategy_id, equity_curve, returns, num_trials)
        
        # Check if retraining is needed
        last_retrain = self.last_retrain_time.get(strategy_id)
        should_retrain, reason = self.monitor.should_retrain(strategy_id, last_retrain)
        
        return should_retrain, reason
    
    def run_retrain_cycle(
        self,
        strategy_id: str,
        df: pd.DataFrame,
        strategy_factory: Callable,
        current_params: Dict[str, Any],
        param_grid: Dict[str, Iterable[Any]],
        fee_bps: float = 2.0,
        weight: float = 1.0
    ) -> tuple[bool, Dict[str, Any]]:
        """Run complete retraining cycle.
        
        Returns:
            (deployed, new_params_if_deployed)
        """
        with self._lock:
            # Check rate limiting
            today = datetime.now().date()
            today_retrains = sum(
                1 for event in self.retrain_history.get(strategy_id, [])
                if event.timestamp.date() == today
            )
            
            if today_retrains >= self.config.max_retrain_per_day:
                LOGGER.warning(
                    f"Max retrains per day reached for {strategy_id} "
                    f"({today_retrains}/{self.config.max_retrain_per_day})"
                )
                return False, current_params
        
        LOGGER.info(f"Starting retraining cycle for {strategy_id}")
        
        # Get current performance
        current_metrics = self.monitor.get_latest_metrics(strategy_id)
        if current_metrics is None:
            LOGGER.error(f"No current metrics for {strategy_id}")
            return False, current_params
        
        # Run retraining
        try:
            retrain_result = self.retrainer.retrain(
                df=df,
                strategy_factory=strategy_factory,
                param_grid=param_grid,
                fee_bps=fee_bps,
                weight=weight
            )
        except Exception as e:
            LOGGER.error(f"Retraining failed for {strategy_id}: {e}")
            return False, current_params
        
        new_params = retrain_result["best_params"]
        
        # Validate new model
        should_deploy, validation_metrics = self.validator.validate_model(
            df=df,
            strategy_factory=strategy_factory,
            old_params=current_params,
            new_params=new_params,
            fee_bps=fee_bps,
            weight=weight
        )
        
        # Record event
        event = RetrainingEvent(
            timestamp=datetime.now(),
            trigger_reason=f"DSR={current_metrics.dsr:.2f}",
            old_dsr=current_metrics.dsr,
            old_sharpe=current_metrics.sharpe,
            new_dsr=validation_metrics.get("new_dsr", 0.0),
            new_sharpe=validation_metrics.get("new_sharpe", 0.0),
            improvement=validation_metrics.get("improvement", 0.0),
            deployed=should_deploy,
            validation_metrics=validation_metrics,
            new_params=new_params
        )
        
        with self._lock:
            if strategy_id not in self.retrain_history:
                self.retrain_history[strategy_id] = []
            self.retrain_history[strategy_id].append(event)
            self.last_retrain_time[strategy_id] = datetime.now()
        
        # Save event
        self._save_event(strategy_id, event)
        
        if should_deploy:
            LOGGER.info(
                f"Deploying new model for {strategy_id}: "
                f"improvement={event.improvement:.2%}"
            )
            self._save_model(strategy_id, new_params)
            return True, new_params
        else:
            LOGGER.info(
                f"New model for {strategy_id} did not meet deployment threshold: "
                f"improvement={event.improvement:.2%} < {self.config.improvement_threshold:.2%}"
            )
            return False, current_params
    
    def _save_model(self, strategy_id: str, params: Dict[str, Any]):
        """Save model parameters to disk."""
        model_path = self.config.model_save_dir / f"{strategy_id}_params.json"
        try:
            with open(model_path, "w") as f:
                json.dump({
                    "strategy_id": strategy_id,
                    "params": params,
                    "timestamp": datetime.now().isoformat(),
                    "version": self._get_next_version(strategy_id)
                }, f, indent=2)
            LOGGER.info(f"Model saved to {model_path}")
        except Exception as e:
            LOGGER.error(f"Failed to save model: {e}")
    
    def _save_event(self, strategy_id: str, event: RetrainingEvent):
        """Save retraining event to disk."""
        event_path = self.config.metrics_save_dir / f"{strategy_id}_events.jsonl"
        try:
            with open(event_path, "a") as f:
                event_data = {
                    "timestamp": event.timestamp.isoformat(),
                    "trigger_reason": event.trigger_reason,
                    "old_dsr": event.old_dsr,
                    "old_sharpe": event.old_sharpe,
                    "new_dsr": event.new_dsr,
                    "new_sharpe": event.new_sharpe,
                    "improvement": event.improvement,
                    "deployed": event.deployed,
                    "validation_metrics": event.validation_metrics,
                    "new_params": event.new_params
                }
                f.write(json.dumps(event_data) + "\n")
            LOGGER.info(f"Event saved to {event_path}")
        except Exception as e:
            LOGGER.error(f"Failed to save event: {e}")
    
    def _get_next_version(self, strategy_id: str) -> int:
        """Get next version number for strategy."""
        with self._lock:
            return len(self.retrain_history.get(strategy_id, [])) + 1
    
    def load_latest_params(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Load latest saved parameters for a strategy."""
        model_path = self.config.model_save_dir / f"{strategy_id}_params.json"
        if not model_path.exists():
            return None
        
        try:
            with open(model_path, "r") as f:
                data = json.load(f)
                return data.get("params")
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")
            return None
    
    def get_retrain_history(self, strategy_id: str) -> list[RetrainingEvent]:
        """Get retraining history for a strategy."""
        with self._lock:
            return self.retrain_history.get(strategy_id, []).copy()
    
    def get_stats(self, strategy_id: str) -> Dict[str, Any]:
        """Get statistics about retraining history."""
        events = self.get_retrain_history(strategy_id)
        if not events:
            return {
                "total_retrains": 0,
                "deployments": 0,
                "avg_improvement": 0.0,
                "last_retrain": None
            }
        
        deployments = [e for e in events if e.deployed]
        improvements = [e.improvement for e in deployments if e.improvement != float('inf')]
        
        return {
            "total_retrains": len(events),
            "deployments": len(deployments),
            "deployment_rate": len(deployments) / len(events) if events else 0.0,
            "avg_improvement": float(np.mean(improvements)) if improvements else 0.0,
            "max_improvement": float(np.max(improvements)) if improvements else 0.0,
            "last_retrain": events[-1].timestamp.isoformat() if events else None,
            "last_deployed": deployments[-1].timestamp.isoformat() if deployments else None
        }
