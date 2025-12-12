"""Real-time performance monitoring for quantitative trading strategies.

This module provides comprehensive performance tracking with:
- Exponentially-weighted rolling Sharpe ratio (configurable half-life)
- Drawdown tracking from peak equity
- Correlation drift detection (live vs backtest comparison using Frobenius norm)
- Automated alert triggering when metrics degrade beyond threshold

Integrates with openquant.utils.alerts for multi-channel notifications.
Thread-safe for concurrent trading systems.

Example:
    >>> from openquant.monitoring import PerformanceTracker
    >>> import numpy as np
    >>> 
    >>> # Initialize with backtest reference metrics
    >>> tracker = PerformanceTracker(
    ...     backtest_sharpe=2.5,
    ...     backtest_max_drawdown=0.15,
    ...     alert_threshold=0.20,
    ...     sharpe_halflife_days=30
    ... )
    >>> 
    >>> # Update with trading results
    >>> tracker.update(equity=105000.0, returns=0.02)
    >>> 
    >>> # Get current metrics
    >>> metrics = tracker.get_current_metrics()
    >>> print(f"Sharpe: {metrics['sharpe_ratio']:.3f}")
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

from ..utils.logging import get_logger
from ..utils.alerts import send_alert

LOGGER = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Snapshot of performance metrics at a point in time."""
    timestamp: datetime
    sharpe_ratio: float
    drawdown: float
    drawdown_from_peak: float
    peak_value: float
    current_value: float
    correlation_drift: Optional[float] = None
    

class PerformanceTracker:
    """Real-time performance monitoring with exponentially-weighted metrics.
    
    Features:
    - Exponentially-weighted rolling Sharpe (30-day half-life)
    - Drawdown tracking from peak
    - Correlation drift detection (live vs backtest)
    - Alert triggering when metrics degrade >20%
    """
    
    def __init__(
        self,
        backtest_correlation_matrix: Optional[np.ndarray] = None,
        backtest_sharpe: Optional[float] = None,
        backtest_max_drawdown: Optional[float] = None,
        alert_threshold: float = 0.20,
        sharpe_halflife_days: int = 30,
        freq: str = "1d"
    ):
        """Initialize performance tracker.
        
        Args:
            backtest_correlation_matrix: Reference correlation matrix from backtest
            backtest_sharpe: Reference Sharpe ratio from backtest
            backtest_max_drawdown: Reference max drawdown from backtest (positive value)
            alert_threshold: Threshold for triggering alerts (e.g., 0.20 for 20% degradation)
            sharpe_halflife_days: Half-life for exponential weighting (default: 30 days)
            freq: Frequency of returns ("1d", "1h", etc.)
        """
        self.backtest_corr_matrix = backtest_correlation_matrix
        self.backtest_sharpe = backtest_sharpe
        self.backtest_max_drawdown = backtest_max_drawdown
        self.alert_threshold = alert_threshold
        self.sharpe_halflife_days = sharpe_halflife_days
        self.freq = freq
        
        # Real-time tracking
        self.returns: List[float] = []
        self.timestamps: List[datetime] = []
        self.equity_values: List[float] = []
        self.peak_equity = 0.0
        self.current_equity = 0.0
        
        # For correlation tracking
        self.symbol_returns: Dict[str, List[float]] = {}
        self.symbol_timestamps: Dict[str, List[datetime]] = {}
        
        # Metrics history
        self.metrics_history: List[PerformanceMetrics] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Alert state tracking (to avoid spam)
        self._last_alert_time: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(hours=1)
        
    def _annualization_factor(self) -> float:
        """Get annualization factor based on frequency."""
        freq = self.freq.lower()
        if freq in {"d", "1d", "daily"}:
            return 252.0
        if freq in {"h", "1h"}:
            return 252.0 * 6.5
        if freq in {"4h"}:
            return 252.0 * (6.5 / 4.0)
        if freq in {"30m", "15m"}:
            return 252.0 * (6.5 * 60 / 30.0) if freq == "30m" else 252.0 * (6.5 * 60 / 15.0)
        return 252.0
    
    def _compute_ew_sharpe(self) -> float:
        """Compute exponentially-weighted Sharpe ratio with specified half-life.
        
        Uses exponential weighting where more recent returns get higher weights.
        Half-life determines how quickly weights decay.
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns_array = np.array(self.returns)
        
        # Compute decay parameter from half-life
        # alpha = 1 - exp(-ln(2) / half_life)
        # For simplicity in daily context: alpha ≈ ln(2) / half_life for small alpha
        decay = np.exp(-np.log(2) / self.sharpe_halflife_days)
        
        # Create weights: most recent observation gets weight (1-decay),
        # previous gets weight (1-decay)*decay, etc.
        n = len(returns_array)
        weights = np.array([(1 - decay) * (decay ** i) for i in range(n)])
        weights = weights[::-1]  # Reverse so most recent is last
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Weighted mean and variance
        mean_return = np.sum(weights * returns_array)
        variance = np.sum(weights * (returns_array - mean_return) ** 2)
        std_return = np.sqrt(variance)
        
        if std_return < 1e-12:
            return 0.0
        
        # Annualize
        sharpe = (mean_return / std_return) * np.sqrt(self._annualization_factor())
        return float(sharpe)
    
    def _compute_drawdown_from_peak(self) -> float:
        """Compute current drawdown from peak equity."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def _compute_correlation_drift(self) -> Optional[float]:
        """Compute correlation drift using Frobenius norm.
        
        Compares live correlation matrix with backtest reference.
        Returns None if insufficient data or no reference matrix.
        """
        if self.backtest_corr_matrix is None:
            return None
        
        # Need at least 2 symbols and sufficient data points
        if len(self.symbol_returns) < 2:
            return None
        
        # Find common timestamps across all symbols
        common_timestamps = None
        for symbol, timestamps in self.symbol_timestamps.items():
            if common_timestamps is None:
                common_timestamps = set(timestamps)
            else:
                common_timestamps = common_timestamps.intersection(set(timestamps))
        
        if common_timestamps is None or len(common_timestamps) < 20:
            return None
        
        # Build aligned returns matrix
        common_timestamps_sorted = sorted(list(common_timestamps))
        symbols = sorted(self.symbol_returns.keys())
        
        returns_matrix = []
        for symbol in symbols:
            symbol_ts = self.symbol_timestamps[symbol]
            symbol_ret = self.symbol_returns[symbol]
            
            # Create dict for fast lookup
            ts_to_ret = dict(zip(symbol_ts, symbol_ret))
            
            # Get returns for common timestamps
            aligned_returns = [ts_to_ret.get(ts, 0.0) for ts in common_timestamps_sorted]
            returns_matrix.append(aligned_returns)
        
        returns_df = pd.DataFrame(returns_matrix).T
        
        # Compute live correlation matrix
        live_corr_matrix = returns_df.corr().values
        
        # Check dimensions match
        if live_corr_matrix.shape != self.backtest_corr_matrix.shape:
            LOGGER.warning(
                f"Correlation matrix dimension mismatch: "
                f"live {live_corr_matrix.shape} vs backtest {self.backtest_corr_matrix.shape}"
            )
            return None
        
        # Compute Frobenius norm of difference
        diff_matrix = live_corr_matrix - self.backtest_corr_matrix
        frobenius_norm = np.linalg.norm(diff_matrix, ord='fro')
        
        return float(frobenius_norm)
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type."""
        if alert_type not in self._last_alert_time:
            return True
        
        elapsed = datetime.now() - self._last_alert_time[alert_type]
        return elapsed >= self._alert_cooldown
    
    def _trigger_alert_if_needed(self, metric_name: str, current: float, reference: float, higher_is_better: bool = True):
        """Trigger alert if metric degrades beyond threshold.
        
        Args:
            metric_name: Name of the metric (for alert message)
            current: Current metric value
            reference: Reference/baseline metric value
            higher_is_better: If True, alert when current < reference * (1 - threshold)
                            If False, alert when current > reference * (1 + threshold)
        """
        if reference == 0 or not np.isfinite(reference) or not np.isfinite(current):
            return
        
        if higher_is_better:
            # Sharpe, profit factor, etc: alert when drops below threshold
            degradation = (reference - current) / abs(reference)
            if degradation > self.alert_threshold:
                alert_type = f"{metric_name}_degradation"
                if self._should_send_alert(alert_type):
                    pct_change = degradation * 100
                    send_alert(
                        subject=f"Performance Alert: {metric_name} Degraded",
                        body=f"{metric_name} has degraded by {pct_change:.1f}%\n"
                             f"Current: {current:.3f}\n"
                             f"Reference: {reference:.3f}\n"
                             f"Threshold: {self.alert_threshold * 100:.0f}%",
                        severity="WARNING"
                    )
                    self._last_alert_time[alert_type] = datetime.now()
        else:
            # Drawdown, correlation drift, etc: alert when increases above threshold
            degradation = (current - reference) / abs(reference)
            if degradation > self.alert_threshold:
                alert_type = f"{metric_name}_degradation"
                if self._should_send_alert(alert_type):
                    pct_change = degradation * 100
                    send_alert(
                        subject=f"Performance Alert: {metric_name} Increased",
                        body=f"{metric_name} has increased by {pct_change:.1f}%\n"
                             f"Current: {current:.3f}\n"
                             f"Reference: {reference:.3f}\n"
                             f"Threshold: {self.alert_threshold * 100:.0f}%",
                        severity="WARNING"
                    )
                    self._last_alert_time[alert_type] = datetime.now()
    
    def update(
        self,
        equity: float,
        returns: Optional[float] = None,
        symbol_returns: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None
    ):
        """Update performance metrics with new data.
        
        Args:
            equity: Current equity value
            returns: Portfolio return for this period (optional, computed from equity if not provided)
            symbol_returns: Dict of {symbol: return} for correlation tracking (optional)
            timestamp: Timestamp for this update (default: now)
        """
        with self._lock:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Update equity tracking
            prev_equity = self.current_equity if self.current_equity > 0 else equity
            self.current_equity = equity
            self.equity_values.append(equity)
            
            # Update peak
            if equity > self.peak_equity:
                self.peak_equity = equity
            
            # Compute return if not provided
            if returns is None:
                returns = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            
            self.returns.append(returns)
            self.timestamps.append(timestamp)
            
            # Update symbol-level returns for correlation tracking
            if symbol_returns:
                for symbol, ret in symbol_returns.items():
                    if symbol not in self.symbol_returns:
                        self.symbol_returns[symbol] = []
                        self.symbol_timestamps[symbol] = []
                    self.symbol_returns[symbol].append(ret)
                    self.symbol_timestamps[symbol].append(timestamp)
            
            # Compute current metrics
            sharpe = self._compute_ew_sharpe()
            drawdown = self._compute_drawdown_from_peak()
            corr_drift = self._compute_correlation_drift()
            
            # Store metrics snapshot
            metrics = PerformanceMetrics(
                timestamp=timestamp,
                sharpe_ratio=sharpe,
                drawdown=drawdown,
                drawdown_from_peak=drawdown,
                peak_value=self.peak_equity,
                current_value=self.current_equity,
                correlation_drift=corr_drift
            )
            self.metrics_history.append(metrics)
            
            # Check for alerts
            if self.backtest_sharpe is not None:
                self._trigger_alert_if_needed("Sharpe Ratio", sharpe, self.backtest_sharpe, higher_is_better=True)
            
            if self.backtest_max_drawdown is not None:
                self._trigger_alert_if_needed("Drawdown", drawdown, self.backtest_max_drawdown, higher_is_better=False)
            
            if corr_drift is not None:
                # For correlation drift, we alert if it increases significantly from 0
                # Use a small reference value (0.1) as baseline
                if corr_drift > 0.5:  # Absolute threshold for correlation drift
                    alert_type = "correlation_drift"
                    if self._should_send_alert(alert_type):
                        send_alert(
                            subject="Performance Alert: Correlation Drift Detected",
                            body=f"Correlation structure has drifted significantly from backtest\n"
                                 f"Frobenius norm: {corr_drift:.3f}\n"
                                 f"This may indicate strategy behavior has changed.",
                            severity="WARNING"
                        )
                        self._last_alert_time[alert_type] = datetime.now()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dict containing current Sharpe, drawdown, correlation drift, etc.
        """
        with self._lock:
            if not self.metrics_history:
                return {
                    "sharpe_ratio": 0.0,
                    "drawdown": 0.0,
                    "peak_equity": 0.0,
                    "current_equity": 0.0,
                    "correlation_drift": None,
                    "num_observations": 0,
                    "backtest_sharpe": self.backtest_sharpe,
                    "backtest_max_drawdown": self.backtest_max_drawdown
                }
            
            latest = self.metrics_history[-1]
            
            return {
                "timestamp": latest.timestamp.isoformat(),
                "sharpe_ratio": latest.sharpe_ratio,
                "drawdown": latest.drawdown,
                "peak_equity": latest.peak_value,
                "current_equity": latest.current_value,
                "correlation_drift": latest.correlation_drift,
                "num_observations": len(self.returns),
                "backtest_sharpe": self.backtest_sharpe,
                "backtest_max_drawdown": self.backtest_max_drawdown,
                "sharpe_degradation_pct": (
                    ((self.backtest_sharpe - latest.sharpe_ratio) / abs(self.backtest_sharpe) * 100)
                    if self.backtest_sharpe and self.backtest_sharpe != 0 else None
                ),
                "drawdown_increase_pct": (
                    ((latest.drawdown - self.backtest_max_drawdown) / abs(self.backtest_max_drawdown) * 100)
                    if self.backtest_max_drawdown and self.backtest_max_drawdown != 0 else None
                )
            }
    
    def get_metrics_history(self, lookback_days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get historical metrics.
        
        Args:
            lookback_days: If specified, only return metrics from last N days
            
        Returns:
            List of metric snapshots
        """
        with self._lock:
            if lookback_days is None:
                metrics_to_return = self.metrics_history
            else:
                cutoff = datetime.now() - timedelta(days=lookback_days)
                metrics_to_return = [m for m in self.metrics_history if m.timestamp >= cutoff]
            
            return [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "sharpe_ratio": m.sharpe_ratio,
                    "drawdown": m.drawdown,
                    "peak_value": m.peak_value,
                    "current_value": m.current_value,
                    "correlation_drift": m.correlation_drift
                }
                for m in metrics_to_return
            ]
    
    def reset_alerts(self):
        """Reset alert cooldowns (useful for testing)."""
        with self._lock:
            self._last_alert_time.clear()
    
    def get_summary(self) -> str:
        """Get human-readable summary of current performance."""
        metrics = self.get_current_metrics()
        
        lines = [
            "=== Performance Tracker Summary ===",
            f"Current Sharpe: {metrics['sharpe_ratio']:.3f}",
            f"Current Drawdown: {metrics['drawdown']:.2%}",
            f"Peak Equity: {metrics['peak_equity']:.2f}",
            f"Current Equity: {metrics['current_equity']:.2f}",
        ]
        
        if metrics['correlation_drift'] is not None:
            lines.append(f"Correlation Drift (Frobenius): {metrics['correlation_drift']:.3f}")
        
        if metrics['backtest_sharpe'] is not None:
            lines.append(f"\nBacktest Sharpe: {metrics['backtest_sharpe']:.3f}")
            if metrics['sharpe_degradation_pct'] is not None:
                lines.append(f"Sharpe Degradation: {metrics['sharpe_degradation_pct']:+.1f}%")
        
        if metrics['backtest_max_drawdown'] is not None:
            lines.append(f"\nBacktest Max DD: {metrics['backtest_max_drawdown']:.2%}")
            if metrics['drawdown_increase_pct'] is not None:
                lines.append(f"Drawdown Change: {metrics['drawdown_increase_pct']:+.1f}%")
        
        lines.append(f"\nObservations: {metrics['num_observations']}")
        
        return "\n".join(lines)
