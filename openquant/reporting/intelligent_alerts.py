"""Intelligent Alert System.

Monitors robot performance and market conditions to generate
intelligent alerts for anomalies, regime changes, and issues.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

from ..utils.logging import get_logger
from ..utils.alerts import send_alert
import os

LOGGER = get_logger(__name__)

class AlertType:
    """Alert type constants."""
    ANOMALY = "anomaly"
    REGIME_CHANGE = "regime_change"
    PERFORMANCE = "performance"
    SYSTEM = "system"

class Alert:
    """Alert data structure."""
    def __init__(
        self,
        alert_type: str,
        severity: str,  # "info", "warning", "critical"
        message: str,
        details: Optional[Dict] = None
    ):
        self.type = alert_type
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

class IntelligentAlerts:
    """
    Intelligent alert system for monitoring robot health and performance.
    """
    def __init__(self, history_file: str = "data/alerts_history.json"):
        self.history_file = Path(history_file)
        self.alerts_buffer: List[Alert] = []
        
    def check_pnl_anomaly(self, recent_pnl: List[float], window: int = 20) -> Optional[Alert]:
        """Check for unusual PnL."""
        if len(recent_pnl) < window:
            return None
            
        recent = pd.Series(recent_pnl[-window:])
        mean_pnl = recent.mean()
        std_pnl = recent.std()
        
        if std_pnl == 0:
            return None
            
        latest_pnl = recent_pnl[-1]
        z_score = (latest_pnl - mean_pnl) / std_pnl
        
        if abs(z_score) > 2.5:
            severity = "critical" if abs(z_score) > 3 else "warning"
            return Alert(
                alert_type=AlertType.ANOMALY,
                severity=severity,
                message=f"Unusual PnL detected: {latest_pnl:.2f} (z-score: {z_score:.2f})",
                details={"z_score": float(z_score), "pnl": float(latest_pnl)}
            )
        return None
        
    def check_drawdown(self, equity_curve: List[float], threshold: float = 0.15) -> Optional[Alert]:
        """Check for excessive drawdown."""
        if len(equity_curve) < 2:
            return None
            
        equity = pd.Series(equity_curve)
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        current_dd = abs(drawdown.iloc[-1])
        
        if current_dd > threshold:
            return Alert(
                alert_type=AlertType.PERFORMANCE,
                severity="critical",
                message=f"High drawdown: {current_dd:.1%} (threshold: {threshold:.1%})",
                details={"drawdown": float(current_dd), "threshold": threshold}
            )
        return None
        
    def check_signal_quality(self, prob_history: List[float], min_prob: float = 0.55) -> Optional[Alert]:
        """Check if ML signal quality is degrading."""
        if len(prob_history) < 10:
            return None
            
        recent_probs = prob_history[-10:]
        avg_prob = np.mean(recent_probs)
        
        if avg_prob < min_prob:
            return Alert(
                alert_type=AlertType.PERFORMANCE,
                severity="warning",
                message=f"ML signal quality degraded: avg prob {avg_prob:.2%}",
                details={"avg_probability": float(avg_prob), "threshold": min_prob}
            )
        return None
        
    def check_regime_change(
        self,
        previous_regime: str,
        current_regime: str
    ) -> Optional[Alert]:
        """Alert on regime changes."""
        if previous_regime != current_regime:
            return Alert(
                alert_type=AlertType.REGIME_CHANGE,
                severity="info",
                message=f"Market regime changed: {previous_regime} → {current_regime}",
                details={"from": previous_regime, "to": current_regime}
            )
        return None
        
    def check_online_learner_convergence(
        self,
        last_update: datetime,
        max_hours: int = 48
    ) -> Optional[Alert]:
        """Check if online learner has stalled."""
        hours_since = (datetime.now() - last_update).total_seconds() / 3600
        
        if hours_since > max_hours:
            return Alert(
                alert_type=AlertType.SYSTEM,
                severity="warning",
                message=f"Online learner hasn't updated in {hours_since:.1f} hours",
                details={"hours_since_update": hours_since}
            )
        return None
        
    def add_alert(self, alert: Alert):
        
        self.alerts_buffer.append(alert)
        LOGGER.warning(f"[{alert.severity.upper()}] {alert.message}")
        try:
            body = json.dumps(alert.to_dict())
            send_alert(alert.message, body, severity=alert.severity)
        except Exception:
            pass
        
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            a.to_dict() for a in self.alerts_buffer
            if a.timestamp > cutoff
        ]
        
    def save_alerts(self):
        """Save alerts to file."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
                
            # Append new
            for alert in self.alerts_buffer:
                history.append(alert.to_dict())
                
            # Keep last 1000
            history = history[-1000:]
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            self.alerts_buffer.clear()
            
        except Exception as e:
            LOGGER.error(f"Failed to save alerts: {e}")

    def run_diagnostics_on_results(self, db_path: str | Path, run_id: Optional[str] = None,
                                   wfo_drop: float = 0.2,
                                   profit_factor_min: float = 1.2,
                                   mc_dd_p95_max: float = 0.25,
                                   p_value_max: float = 0.10) -> None:
        try:
            cfg_path = Path("data/diagnostics_config.json")
            if cfg_path.exists():
                try:
                    with open(cfg_path, "r") as f:
                        cfg = json.load(f)
                    wfo_drop = float(cfg.get("wfo_drop", wfo_drop))
                    profit_factor_min = float(cfg.get("profit_factor_min", profit_factor_min))
                    mc_dd_p95_max = float(cfg.get("mc_dd_p95_max", mc_dd_p95_max))
                    p_value_max = float(cfg.get("p_value_max", p_value_max))
                except Exception:
                    pass
            import duckdb
            con = duckdb.connect(str(db_path))
            base_sql = "SELECT * FROM results"
            if run_id:
                base_sql += " WHERE run_id = ?"
                df = con.execute(base_sql, (run_id,)).df()
            else:
                df = con.execute(base_sql).df()
            if df.empty:
                return
            # Underperformers
            bad_rows = df[(df.get('ok', True) == False) | (df.get('profit_factor', 2.0) < profit_factor_min) |
                          (df.get('wfo_mts', 0.0) < wfo_drop) | (df.get('mc_dd_p95', 0.0) > mc_dd_p95_max) |
                          (df.get('p_value', 1.0) > p_value_max)]
            for _, r in bad_rows.iterrows():
                details = {
                    "exchange": r.get("exchange"),
                    "symbol": r.get("symbol"),
                    "timeframe": r.get("timeframe"),
                    "strategy": r.get("strategy"),
                    "sharpe": float(r.get("sharpe", 0.0)),
                    "profit_factor": float(r.get("profit_factor", 0.0)),
                    "wfo_mts": float(r.get("wfo_mts", 0.0)),
                    "mc_dd_p95": float(r.get("mc_dd_p95", 0.0)),
                    "p_value": float(r.get("p_value", 1.0))
                }
                sev = "critical" if float(details["mc_dd_p95"]) > mc_dd_p95_max else "warning"
                self.add_alert(Alert(AlertType.PERFORMANCE, sev,
                                     f"Underperforming: {r.get('strategy')} {r.get('symbol')} {r.get('timeframe')}",
                                     details))
                # Recommendations
                recs = []
                if float(details["profit_factor"]) < profit_factor_min:
                    recs.append("Tighten entry/exit thresholds or improve filters")
                if float(details["mc_dd_p95"]) > mc_dd_p95_max:
                    recs.append("Reduce leverage; add volatility guardrails")
                if float(details["wfo_mts"]) < wfo_drop:
                    recs.append("Expand hyperparameter grid; validate with WFO")
                if float(details["p_value"]) > p_value_max:
                    recs.append("Insignificant returns; revisit signal and feature set")
                if recs:
                    self.add_alert(Alert(AlertType.PERFORMANCE, "info",
                                         "Recommendations",
                                         {"suggestions": recs, **details}))

            # Missing strategies/features
            required = {"kalman", "hurst", "stat_arb", "liquidity"}
            present = set(str(x).lower() for x in df.get("strategy", []).tolist())
            missing = list(sorted(required - present))
            if missing:
                self.add_alert(Alert(AlertType.SYSTEM, "warning",
                                     f"Missing strategies in run: {', '.join(missing)}",
                                     {"missing": missing}))

            # Data source suggestions
            sample_ex = str(df.iloc[0].get("exchange", ""))
            if sample_ex.lower() in {"binance", "bybit", "okx"}:
                self.add_alert(Alert(AlertType.SYSTEM, "info",
                                     "Consider adding funding rates and orderbook imbalance features",
                                     {"exchange": sample_ex}))
            if sample_ex.lower() == "mt5":
                self.add_alert(Alert(AlertType.SYSTEM, "info",
                                     "Validate spread/swap settings and economic calendar features",
                                     {"exchange": sample_ex}))

            # Regular health check: stale data
            try:
                last_ts = pd.to_datetime(df.get("ts")).max()
                hours_since = (datetime.now() - last_ts).total_seconds() / 3600.0
                if hours_since > 12.0:
                    self.add_alert(Alert(AlertType.SYSTEM, "warning",
                                         f"Results are stale ({hours_since:.1f}h since last update)",
                                         {"hours_since": hours_since}))
            except Exception:
                pass
        except Exception as e:
            LOGGER.warning(f"Diagnostics error: {e}")

    def generate_diagnostic_report(self, db_path: str | Path, run_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            import duckdb
            con = duckdb.connect(str(db_path))
            sql = "SELECT * FROM results"
            df = con.execute(sql if not run_id else sql + " WHERE run_id = ?", (() if not run_id else (run_id,))).df()
            if df.empty:
                return {}
            perf = {}
            try:
                from .performance_tracker import PERFORMANCE_TRACKER
                perf = PERFORMANCE_TRACKER.get_stats(lookback_days=30)
            except Exception:
                perf = {}
            agg = {
                "rows": int(len(df)),
                "mean_sharpe": float(df.get("sharpe", 0).mean() if "sharpe" in df.columns else 0.0),
                "mean_profit_factor": float(df.get("profit_factor", 0).replace([float('inf')], None).dropna().mean() if "profit_factor" in df.columns else 0.0),
                "mean_win_rate": float(df.get("win_rate", 0).mean() if "win_rate" in df.columns else 0.0),
                "mean_wfo_mts": float(df.get("wfo_mts", 0).mean() if "wfo_mts" in df.columns else 0.0),
                "median_p_value": float(df.get("p_value", 1.0).median() if "p_value" in df.columns else 1.0),
                "mean_mc_dd_p95": float(df.get("mc_dd_p95", 0).mean() if "mc_dd_p95" in df.columns else 0.0),
                "guardrail_violations": int(df[df.get("ok", True) == False].shape[0])
            }
            top = []
            try:
                top = df.sort_values("sharpe", ascending=False).head(10)[["exchange","symbol","timeframe","strategy","sharpe","profit_factor","win_rate","wfo_mts","p_value","mc_dd_p95"]].to_dict(orient="records")
            except Exception:
                top = []
            recs = []
            if agg["mean_profit_factor"] < 1.4:
                recs.append({"recommendation": "Improve filters and exits", "evidence": {"mean_profit_factor": agg["mean_profit_factor"]}})
            if agg["mean_mc_dd_p95"] > 0.25:
                recs.append({"recommendation": "Reduce leverage and add volatility guards", "evidence": {"mean_mc_dd_p95": agg["mean_mc_dd_p95"]}})
            if agg["mean_wfo_mts"] < 0.2:
                recs.append({"recommendation": "Expand hyperparameter search and validate with WFO", "evidence": {"mean_wfo_mts": agg["mean_wfo_mts"]}})
            if agg["median_p_value"] > 0.10:
                recs.append({"recommendation": "Revise signal and feature set", "evidence": {"median_p_value": agg["median_p_value"]}})
            roi = {}
            try:
                pf = float(perf.get("profit_factor", 0))
                target_pf = max(1.2, pf * 1.1) if pf > 0 else 1.2
                total_pnl = float(perf.get("total_pnl", 0))
                uplift_pct = float((target_pf - pf) / pf) if pf > 0 else 0.10
                roi = {"current_profit_factor": pf, "target_profit_factor": target_pf, "projected_pnl_delta_pct": uplift_pct, "projected_pnl_delta_usd": float(total_pnl * uplift_pct)}
            except Exception:
                roi = {}
            report = {
                "run_id": run_id,
                "aggregate": agg,
                "top_results": top,
                "live_performance": perf,
                "recommendations": recs,
                "roi_projection": roi,
                "timestamp": datetime.now().isoformat()
            }
            out_path = Path("data/diagnostic_report.json")
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(report, f, indent=2)
                hist_path = Path("data/diagnostic_reports.json")
                hist = []
                if hist_path.exists():
                    try:
                        with open(hist_path, "r") as hf:
                            hist = json.load(hf)
                    except Exception:
                        hist = []
                hist.append(report)
                hist = hist[-200:]
                with open(hist_path, "w") as hf:
                    json.dump(hist, hf, indent=2)
            except Exception:
                pass
            return report
        except Exception as e:
            LOGGER.warning(f"Report generation error: {e}")
            return {}

    def propose_optimization_actions(self, db_path: str | Path, run_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            import duckdb
            con = duckdb.connect(str(db_path))
            sql = "SELECT strategy, symbol, timeframe, sharpe, profit_factor, wfo_mts, p_value, mc_dd_p95 FROM results"
            df = con.execute(sql if not run_id else sql + " WHERE run_id = ?", (() if not run_id else (run_id,))).df()
            if df.empty:
                return {}
            overrides: Dict[str, Any] = {}
            trials_multiplier = 1.0
            for _, r in df.iterrows():
                strat = str(r.get("strategy", "")).lower()
                pf = float(r.get("profit_factor", 0) or 0)
                wfo = float(r.get("wfo_mts", 0) or 0)
                pval = float(r.get("p_value", 1) or 1)
                mcdd = float(r.get("mc_dd_p95", 0) or 0)
                if pf < 1.2 or wfo < 0.2 or pval > 0.10 or mcdd > 0.25:
                    trials_multiplier = max(trials_multiplier, 1.3)
                    if strat == "kalman":
                        overrides["kalman"] = {"process_noise": [1e-5, 5e-5, 1e-4], "measurement_noise": [1e-3, 5e-3, 1e-2], "threshold": [0.8, 1.0, 1.5]}
                    elif strat == "hurst":
                        overrides["hurst"] = {"lookback": [50, 100, 150], "trend_threshold": [0.55, 0.60], "mr_threshold": [0.45, 0.40]}
                    elif strat == "stat_arb":
                        overrides["stat_arb"] = {"entry_z": [1.8, 2.0, 2.5], "exit_z": [-0.2, 0.0, 0.5]}
                    elif strat == "liquidity":
                        overrides["liquidity"] = {"vpin_threshold": [0.2, 0.3, 0.4, 0.5], "lookback": [10, 20, 30]}
            actions = {"param_grid_overrides": overrides, "optuna_trials_multiplier": trials_multiplier, "timestamp": datetime.now().isoformat()}
            path = Path("data/optimization_actions.json")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    json.dump(actions, f, indent=2)
            except Exception:
                pass
            return actions
        except Exception as e:
            LOGGER.warning(f"Optimization actions error: {e}")
            return {}
