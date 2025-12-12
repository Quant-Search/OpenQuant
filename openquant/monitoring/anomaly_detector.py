"""Statistical Process Control Anomaly Detection System.

Implements statistical process control (SPC) techniques to detect anomalies:
- 3-Sigma Rules: Detect outliers beyond 3 standard deviations
- CUSUM (Cumulative Sum Control): Detect persistent shifts in process mean
- Volume spike detection
- P&L pattern anomalies
- Connectivity/latency issues

Mathematical foundations:
1. 3-Sigma Rules:
   - Control limits: μ ± 3σ
   - Western Electric rules for early detection
   
2. CUSUM:
   - Upper CUSUM: S_H = max(0, S_H + x - μ - K)
   - Lower CUSUM: S_L = max(0, S_L - x + μ - K)
   - K = slack parameter (typically 0.5σ)
   - H = decision threshold (typically 4-5σ)

3. Volume Detection:
   - Exponentially weighted moving average (EWMA)
   - Alert when volume > μ_volume + k*σ_volume

Usage:
    detector = AnomalyDetector(
        alert_callback=send_alert,
        sigma_threshold=3.0,
        cusum_slack=0.5,
        cusum_threshold=4.0,
    )
    
    # Update with new data
    detector.update_pnl(current_pnl, timestamp)
    detector.update_volume(symbol, volume, timestamp)
    detector.update_latency(latency_ms, timestamp)
    
    # Check for anomalies
    anomalies = detector.get_recent_anomalies(hours=1)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import json
from pathlib import Path
import numpy as np

from openquant.utils.logging import get_logger
from openquant.utils.alerts import send_alert

LOGGER = get_logger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected by the system."""
    PNL_SPIKE = "pnl_spike"
    PNL_DROP = "pnl_drop"
    PNL_DRIFT_UP = "pnl_drift_up"
    PNL_DRIFT_DOWN = "pnl_drift_down"
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DROP = "volume_drop"
    LATENCY_SPIKE = "latency_spike"
    CONNECTIVITY_ISSUE = "connectivity_issue"
    SIGMA_RULE_1 = "sigma_rule_1"  # 1 point > 3σ
    SIGMA_RULE_2 = "sigma_rule_2"  # 2 of 3 points > 2σ
    SIGMA_RULE_3 = "sigma_rule_3"  # 4 of 5 points > 1σ
    SIGMA_RULE_4 = "sigma_rule_4"  # 8 consecutive points on same side


@dataclass
class AnomalyEvent:
    """Record of an anomaly detection event."""
    timestamp: str
    anomaly_type: AnomalyType
    metric: str  # "pnl", "volume", "latency", etc.
    value: float
    expected_value: float
    deviation: float  # in units of σ
    severity: str  # "INFO", "WARNING", "CRITICAL"
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CUSUMDetector:
    """CUSUM (Cumulative Sum Control) detector for persistent shifts.
    
    Detects gradual drifts in process mean that might be missed by
    simple threshold checks.
    """
    
    def __init__(
        self,
        target_mean: float = 0.0,
        slack: float = 0.5,
        threshold: float = 4.0,
        std_dev: float = 1.0,
    ):
        """Initialize CUSUM detector.
        
        Args:
            target_mean: Target process mean (μ)
            slack: Slack parameter K (typically 0.5σ)
            threshold: Decision threshold H (typically 4-5σ)
            std_dev: Process standard deviation (σ)
        """
        self.target_mean = target_mean
        self.slack = slack
        self.threshold = threshold
        self.std_dev = max(std_dev, 1e-6)
        
        self.s_high = 0.0  # Upper CUSUM
        self.s_low = 0.0   # Lower CUSUM
        
        self.history: deque = deque(maxlen=1000)
        
    def update(self, value: float) -> Tuple[bool, Optional[str]]:
        """Update CUSUM with new observation.
        
        Returns:
            (is_anomaly, direction) where direction is "up", "down", or None
        """
        self.history.append(value)
        
        # Update adaptive parameters if we have enough data
        if len(self.history) >= 30:
            recent = np.array(list(self.history)[-100:])
            self.std_dev = max(np.std(recent), 1e-6)
            
        # Normalize value
        z = (value - self.target_mean) / self.std_dev
        k = self.slack
        
        # Update CUSUM statistics
        self.s_high = max(0, self.s_high + z - k)
        self.s_low = max(0, self.s_low - z - k)
        
        # Check thresholds
        if self.s_high > self.threshold:
            return True, "up"
        elif self.s_low > self.threshold:
            return True, "down"
        
        return False, None
    
    def reset(self) -> None:
        """Reset CUSUM statistics."""
        self.s_high = 0.0
        self.s_low = 0.0


class SigmaRuleDetector:
    """Detector implementing Western Electric Rules (3-sigma rules).
    
    Rules:
    1. One point > 3σ from center line
    2. Two out of three consecutive points > 2σ from center line (same side)
    3. Four out of five consecutive points > 1σ from center line (same side)
    4. Eight consecutive points on same side of center line
    """
    
    def __init__(self, window_size: int = 100):
        """Initialize sigma rule detector.
        
        Args:
            window_size: Rolling window for calculating statistics
        """
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.mean = 0.0
        self.std_dev = 1.0
        
    def update(self, value: float) -> List[AnomalyType]:
        """Update with new observation and check Western Electric rules.
        
        Returns:
            List of triggered rule types
        """
        self.history.append(value)
        
        # Need at least 10 points to establish baseline
        if len(self.history) < 10:
            return []
        
        # Update statistics
        data = np.array(self.history)
        self.mean = np.mean(data)
        self.std_dev = max(np.std(data), 1e-6)
        
        z_score = (value - self.mean) / self.std_dev
        
        violations = []
        
        # Rule 1: One point > 3σ
        if abs(z_score) > 3.0:
            violations.append(AnomalyType.SIGMA_RULE_1)
        
        # Rule 2: Two out of three consecutive points > 2σ (same side)
        if len(self.history) >= 3:
            recent_3 = list(self.history)[-3:]
            z_scores_3 = [(v - self.mean) / self.std_dev for v in recent_3]
            
            positive_2sigma = sum(1 for z in z_scores_3 if z > 2.0)
            negative_2sigma = sum(1 for z in z_scores_3 if z < -2.0)
            
            if positive_2sigma >= 2 or negative_2sigma >= 2:
                violations.append(AnomalyType.SIGMA_RULE_2)
        
        # Rule 3: Four out of five consecutive points > 1σ (same side)
        if len(self.history) >= 5:
            recent_5 = list(self.history)[-5:]
            z_scores_5 = [(v - self.mean) / self.std_dev for v in recent_5]
            
            positive_1sigma = sum(1 for z in z_scores_5 if z > 1.0)
            negative_1sigma = sum(1 for z in z_scores_5 if z < -1.0)
            
            if positive_1sigma >= 4 or negative_1sigma >= 4:
                violations.append(AnomalyType.SIGMA_RULE_3)
        
        # Rule 4: Eight consecutive points on same side of center line
        if len(self.history) >= 8:
            recent_8 = list(self.history)[-8:]
            z_scores_8 = [(v - self.mean) / self.std_dev for v in recent_8]
            
            all_positive = all(z > 0 for z in z_scores_8)
            all_negative = all(z < 0 for z in z_scores_8)
            
            if all_positive or all_negative:
                violations.append(AnomalyType.SIGMA_RULE_4)
        
        return violations


class AnomalyDetector:
    """Comprehensive anomaly detection system using statistical process control.
    
    Monitors multiple metrics (P&L, volume, latency) and detects anomalies using:
    - 3-sigma rules (Western Electric rules)
    - CUSUM for drift detection
    - Volume spike detection
    - Connectivity monitoring
    """
    
    def __init__(
        self,
        alert_callback: Optional[Callable[[str, str, str], None]] = None,
        sigma_threshold: float = 3.0,
        cusum_slack: float = 0.5,
        cusum_threshold: float = 4.0,
        volume_spike_threshold: float = 3.0,
        latency_threshold_ms: float = 1000.0,
        max_history: int = 10000,
        state_file: str = "data/anomaly_detector_state.json",
    ):
        """Initialize anomaly detector.
        
        Args:
            alert_callback: Function to call when anomaly detected (subject, body, severity)
            sigma_threshold: Number of standard deviations for outlier detection
            cusum_slack: CUSUM slack parameter (K)
            cusum_threshold: CUSUM decision threshold (H)
            volume_spike_threshold: Threshold for volume spike detection (in σ)
            latency_threshold_ms: Absolute latency threshold in milliseconds
            max_history: Maximum number of events to keep in memory
            state_file: Path to persist detector state
        """
        self.alert_callback = alert_callback or self._default_alert
        self.sigma_threshold = sigma_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.max_history = max_history
        self.state_file = Path(state_file)
        
        # Detectors for different metrics
        self.pnl_cusum = CUSUMDetector(
            target_mean=0.0,
            slack=cusum_slack,
            threshold=cusum_threshold,
        )
        self.pnl_sigma = SigmaRuleDetector(window_size=100)
        
        # Volume tracking per symbol
        self.volume_history: Dict[str, deque] = {}
        self.volume_stats: Dict[str, Dict[str, float]] = {}
        
        # Latency tracking
        self.latency_history: deque = deque(maxlen=1000)
        self.latency_sigma = SigmaRuleDetector(window_size=100)
        
        # P&L tracking
        self.pnl_history: deque = deque(maxlen=1000)
        self.last_pnl: Optional[float] = None
        self.last_update_time: Optional[datetime] = None
        
        # Anomaly event history
        self.anomaly_events: deque = deque(maxlen=max_history)
        
        # Connectivity tracking
        self.connectivity_failures = 0
        self.last_connectivity_check = datetime.now()
        
        # Load previous state if exists
        self._load_state()
        
    def _default_alert(self, subject: str, body: str, severity: str) -> None:
        """Default alert handler using built-in alert system."""
        send_alert(subject, body, severity=severity)
        
    def _load_state(self) -> None:
        """Load detector state from file."""
        if not self.state_file.exists():
            return
            
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                
            # Restore history
            if "pnl_history" in state:
                self.pnl_history = deque(state["pnl_history"], maxlen=1000)
                if len(self.pnl_history) > 0:
                    self.last_pnl = self.pnl_history[-1]
                    
            if "latency_history" in state:
                self.latency_history = deque(state["latency_history"], maxlen=1000)
                
            if "volume_stats" in state:
                self.volume_stats = state["volume_stats"]
                
            LOGGER.info("Loaded anomaly detector state from file")
        except Exception as e:
            LOGGER.warning(f"Failed to load anomaly detector state: {e}")
            
    def _save_state(self) -> None:
        """Persist detector state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                "pnl_history": list(self.pnl_history),
                "latency_history": list(self.latency_history),
                "volume_stats": self.volume_stats,
                "last_update": datetime.now().isoformat(),
            }
            
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            LOGGER.warning(f"Failed to save anomaly detector state: {e}")
            
    def update_pnl(
        self,
        current_pnl: float,
        timestamp: Optional[datetime] = None,
    ) -> List[AnomalyEvent]:
        """Update P&L and check for anomalies.
        
        Args:
            current_pnl: Current profit/loss value
            timestamp: Timestamp of observation (default: now)
            
        Returns:
            List of detected anomalies
        """
        timestamp = timestamp or datetime.now()
        self.last_update_time = timestamp
        
        anomalies = []
        
        # Calculate P&L change if we have previous data
        if self.last_pnl is not None:
            pnl_change = current_pnl - self.last_pnl
            self.pnl_history.append(pnl_change)
            
            # CUSUM detection for drift
            is_anomaly, direction = self.pnl_cusum.update(pnl_change)
            if is_anomaly:
                anomaly_type = (
                    AnomalyType.PNL_DRIFT_UP if direction == "up"
                    else AnomalyType.PNL_DRIFT_DOWN
                )
                
                anomaly = AnomalyEvent(
                    timestamp=timestamp.isoformat(),
                    anomaly_type=anomaly_type,
                    metric="pnl",
                    value=current_pnl,
                    expected_value=self.last_pnl,
                    deviation=abs(pnl_change) / max(self.pnl_cusum.std_dev, 1e-6),
                    severity="WARNING",
                    description=f"P&L drift detected: {direction}. Change: {pnl_change:.2f}",
                    metadata={
                        "cusum_high": self.pnl_cusum.s_high,
                        "cusum_low": self.pnl_cusum.s_low,
                    }
                )
                anomalies.append(anomaly)
                self._record_anomaly(anomaly)
                self.pnl_cusum.reset()
                
            # Sigma rule detection
            sigma_violations = self.pnl_sigma.update(pnl_change)
            for rule_type in sigma_violations:
                z_score = (pnl_change - self.pnl_sigma.mean) / self.pnl_sigma.std_dev
                
                severity = "CRITICAL" if rule_type == AnomalyType.SIGMA_RULE_1 else "WARNING"
                
                anomaly = AnomalyEvent(
                    timestamp=timestamp.isoformat(),
                    anomaly_type=rule_type,
                    metric="pnl",
                    value=current_pnl,
                    expected_value=self.last_pnl + self.pnl_sigma.mean,
                    deviation=abs(z_score),
                    severity=severity,
                    description=f"P&L {rule_type.value} violated. Z-score: {z_score:.2f}",
                    metadata={
                        "z_score": z_score,
                        "mean": self.pnl_sigma.mean,
                        "std_dev": self.pnl_sigma.std_dev,
                    }
                )
                anomalies.append(anomaly)
                self._record_anomaly(anomaly)
                
            # Simple threshold check for large changes
            if len(self.pnl_history) >= 10:
                mean = np.mean(self.pnl_history)
                std = np.std(self.pnl_history)
                z_score = abs(pnl_change - mean) / max(std, 1e-6)
                
                if z_score > self.sigma_threshold:
                    anomaly_type = (
                        AnomalyType.PNL_SPIKE if pnl_change > 0
                        else AnomalyType.PNL_DROP
                    )
                    
                    anomaly = AnomalyEvent(
                        timestamp=timestamp.isoformat(),
                        anomaly_type=anomaly_type,
                        metric="pnl",
                        value=current_pnl,
                        expected_value=self.last_pnl + mean,
                        deviation=z_score,
                        severity="CRITICAL" if z_score > 5 else "WARNING",
                        description=f"Unexpected P&L change: {pnl_change:.2f} ({z_score:.2f}σ)",
                        metadata={"z_score": z_score}
                    )
                    anomalies.append(anomaly)
                    self._record_anomaly(anomaly)
        
        self.last_pnl = current_pnl
        self._save_state()
        
        return anomalies
    
    def update_volume(
        self,
        symbol: str,
        volume: float,
        timestamp: Optional[datetime] = None,
    ) -> List[AnomalyEvent]:
        """Update volume for a symbol and check for anomalies.
        
        Args:
            symbol: Trading symbol
            volume: Current volume
            timestamp: Timestamp of observation
            
        Returns:
            List of detected anomalies
        """
        timestamp = timestamp or datetime.now()
        
        anomalies = []
        
        # Initialize tracking for new symbols
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=1000)
            self.volume_stats[symbol] = {"mean": 0.0, "std": 1.0}
        
        history = self.volume_history[symbol]
        history.append(volume)
        
        # Need enough data to establish baseline
        if len(history) < 30:
            return anomalies
        
        # Update statistics with exponential weighting
        alpha = 0.1  # EWMA smoothing factor
        data = np.array(history)
        current_mean = np.mean(data[-100:])  # Use recent window
        current_std = max(np.std(data[-100:]), 1e-6)
        
        # Update stats with EWMA
        old_mean = self.volume_stats[symbol]["mean"]
        old_std = self.volume_stats[symbol]["std"]
        
        self.volume_stats[symbol]["mean"] = alpha * current_mean + (1 - alpha) * old_mean
        self.volume_stats[symbol]["std"] = alpha * current_std + (1 - alpha) * old_std
        
        mean = self.volume_stats[symbol]["mean"]
        std = self.volume_stats[symbol]["std"]
        
        # Check for volume spikes
        z_score = (volume - mean) / std
        
        if z_score > self.volume_spike_threshold:
            anomaly = AnomalyEvent(
                timestamp=timestamp.isoformat(),
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                metric=f"volume_{symbol}",
                value=volume,
                expected_value=mean,
                deviation=z_score,
                severity="WARNING" if z_score < 5 else "CRITICAL",
                description=f"Volume spike for {symbol}: {volume:.0f} ({z_score:.2f}σ above mean)",
                metadata={
                    "symbol": symbol,
                    "z_score": z_score,
                    "mean": mean,
                    "std": std,
                }
            )
            anomalies.append(anomaly)
            self._record_anomaly(anomaly)
        
        # Check for volume drops (potential liquidity issues)
        if z_score < -self.volume_spike_threshold:
            anomaly = AnomalyEvent(
                timestamp=timestamp.isoformat(),
                anomaly_type=AnomalyType.VOLUME_DROP,
                metric=f"volume_{symbol}",
                value=volume,
                expected_value=mean,
                deviation=abs(z_score),
                severity="WARNING",
                description=f"Volume drop for {symbol}: {volume:.0f} ({abs(z_score):.2f}σ below mean)",
                metadata={
                    "symbol": symbol,
                    "z_score": z_score,
                    "mean": mean,
                    "std": std,
                }
            )
            anomalies.append(anomaly)
            self._record_anomaly(anomaly)
        
        self._save_state()
        return anomalies
    
    def update_latency(
        self,
        latency_ms: float,
        timestamp: Optional[datetime] = None,
    ) -> List[AnomalyEvent]:
        """Update latency measurement and check for anomalies.
        
        Args:
            latency_ms: Latency in milliseconds
            timestamp: Timestamp of observation
            
        Returns:
            List of detected anomalies
        """
        timestamp = timestamp or datetime.now()
        
        anomalies = []
        
        self.latency_history.append(latency_ms)
        
        # Absolute threshold check
        if latency_ms > self.latency_threshold_ms:
            anomaly = AnomalyEvent(
                timestamp=timestamp.isoformat(),
                anomaly_type=AnomalyType.LATENCY_SPIKE,
                metric="latency",
                value=latency_ms,
                expected_value=self.latency_threshold_ms,
                deviation=latency_ms / self.latency_threshold_ms,
                severity="CRITICAL",
                description=f"High latency detected: {latency_ms:.1f}ms (threshold: {self.latency_threshold_ms:.1f}ms)",
                metadata={"latency_ms": latency_ms}
            )
            anomalies.append(anomaly)
            self._record_anomaly(anomaly)
        
        # Statistical check using sigma rules
        if len(self.latency_history) >= 30:
            sigma_violations = self.latency_sigma.update(latency_ms)
            
            for rule_type in sigma_violations:
                z_score = (latency_ms - self.latency_sigma.mean) / self.latency_sigma.std_dev
                
                anomaly = AnomalyEvent(
                    timestamp=timestamp.isoformat(),
                    anomaly_type=rule_type,
                    metric="latency",
                    value=latency_ms,
                    expected_value=self.latency_sigma.mean,
                    deviation=abs(z_score),
                    severity="WARNING",
                    description=f"Latency {rule_type.value} violated. Latency: {latency_ms:.1f}ms (Z={z_score:.2f})",
                    metadata={
                        "z_score": z_score,
                        "mean": self.latency_sigma.mean,
                        "std_dev": self.latency_sigma.std_dev,
                    }
                )
                anomalies.append(anomaly)
                self._record_anomaly(anomaly)
        
        self._save_state()
        return anomalies
    
    def report_connectivity_failure(
        self,
        reason: str,
        timestamp: Optional[datetime] = None,
    ) -> AnomalyEvent:
        """Report a connectivity failure.
        
        Args:
            reason: Description of the connectivity issue
            timestamp: Timestamp of the failure
            
        Returns:
            Anomaly event
        """
        timestamp = timestamp or datetime.now()
        
        self.connectivity_failures += 1
        self.last_connectivity_check = timestamp
        
        anomaly = AnomalyEvent(
            timestamp=timestamp.isoformat(),
            anomaly_type=AnomalyType.CONNECTIVITY_ISSUE,
            metric="connectivity",
            value=self.connectivity_failures,
            expected_value=0,
            deviation=self.connectivity_failures,
            severity="CRITICAL",
            description=f"Connectivity failure #{self.connectivity_failures}: {reason}",
            metadata={
                "reason": reason,
                "failure_count": self.connectivity_failures,
            }
        )
        
        self._record_anomaly(anomaly)
        return anomaly
    
    def reset_connectivity_failures(self) -> None:
        """Reset connectivity failure counter after successful reconnection."""
        if self.connectivity_failures > 0:
            LOGGER.info(f"Resetting connectivity failures (was {self.connectivity_failures})")
            self.connectivity_failures = 0
    
    def _record_anomaly(self, anomaly: AnomalyEvent) -> None:
        """Record anomaly and trigger alert."""
        self.anomaly_events.append(anomaly)
        
        # Trigger alert
        subject = f"Anomaly Detected: {anomaly.anomaly_type.value}"
        body = (
            f"{anomaly.description}\n\n"
            f"Metric: {anomaly.metric}\n"
            f"Value: {anomaly.value:.2f}\n"
            f"Expected: {anomaly.expected_value:.2f}\n"
            f"Deviation: {anomaly.deviation:.2f}σ\n"
            f"Time: {anomaly.timestamp}"
        )
        
        self.alert_callback(subject, body, anomaly.severity)
        
        LOGGER.warning(
            f"Anomaly detected: {anomaly.anomaly_type.value}",
            extra={
                "anomaly_type": anomaly.anomaly_type.value,
                "metric": anomaly.metric,
                "value": anomaly.value,
                "deviation": anomaly.deviation,
            }
        )
    
    def get_recent_anomalies(
        self,
        hours: float = 24.0,
        anomaly_type: Optional[AnomalyType] = None,
    ) -> List[AnomalyEvent]:
        """Get recent anomalies within time window.
        
        Args:
            hours: Time window in hours
            anomaly_type: Filter by specific anomaly type
            
        Returns:
            List of anomaly events
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = []
        for event in self.anomaly_events:
            event_time = datetime.fromisoformat(event.timestamp)
            if event_time >= cutoff:
                if anomaly_type is None or event.anomaly_type == anomaly_type:
                    recent.append(event)
        
        return recent
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current detector statistics.
        
        Returns:
            Dictionary with statistics for all metrics
        """
        stats = {
            "pnl": {
                "history_size": len(self.pnl_history),
                "last_value": self.last_pnl,
                "cusum_high": self.pnl_cusum.s_high,
                "cusum_low": self.pnl_cusum.s_low,
            },
            "latency": {
                "history_size": len(self.latency_history),
                "current_mean": self.latency_sigma.mean if len(self.latency_history) > 0 else 0,
                "current_std": self.latency_sigma.std_dev if len(self.latency_history) > 0 else 0,
            },
            "volume": {
                "tracked_symbols": len(self.volume_history),
                "stats": self.volume_stats.copy(),
            },
            "connectivity": {
                "failure_count": self.connectivity_failures,
                "last_check": self.last_connectivity_check.isoformat() if self.last_connectivity_check else None,
            },
            "anomalies": {
                "total_count": len(self.anomaly_events),
                "recent_1h": len(self.get_recent_anomalies(hours=1)),
                "recent_24h": len(self.get_recent_anomalies(hours=24)),
            }
        }
        
        if len(self.pnl_history) > 0:
            stats["pnl"]["mean"] = float(np.mean(self.pnl_history))
            stats["pnl"]["std"] = float(np.std(self.pnl_history))
            
        if len(self.latency_history) > 0:
            stats["latency"]["mean"] = float(np.mean(self.latency_history))
            stats["latency"]["std"] = float(np.std(self.latency_history))
            stats["latency"]["p95"] = float(np.percentile(self.latency_history, 95))
            stats["latency"]["p99"] = float(np.percentile(self.latency_history, 99))
        
        return stats
    
    def export_anomalies(self, output_file: str) -> None:
        """Export anomaly history to JSON file.
        
        Args:
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        anomalies_data = []
        for event in self.anomaly_events:
            data = asdict(event)
            data["anomaly_type"] = event.anomaly_type.value
            anomalies_data.append(data)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(anomalies_data, f, indent=2)
        
        LOGGER.info(f"Exported {len(anomalies_data)} anomalies to {output_file}")


# Global detector instance
ANOMALY_DETECTOR = AnomalyDetector()
