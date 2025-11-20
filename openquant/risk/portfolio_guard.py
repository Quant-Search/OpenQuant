"""Portfolio Risk Guard.
Monitors global portfolio limits (Drawdown, Daily Loss, Exposure) and triggers emergency stops.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Optional, Tuple, List
import numpy as np

from openquant.utils.logging import get_logger
from openquant.utils.alerts import send_alert

LOGGER = get_logger(__name__)

class PortfolioGuard:
    """Monitors portfolio risk metrics and enforces limits."""
    
    def __init__(self, state_file: Path = Path("data/risk_state.json")):
        self.state_file = state_file
        self.high_water_mark = 0.0
        self.start_day_equity = 0.0
        self.current_date = date.today()
        self.limits = {
            "dd_limit": 0.20,          # 20% Max Drawdown
            "daily_loss_cap": 0.05,    # 5% Daily Loss
            "cvar_limit": 0.08,        # 8% CVaR (95% confidence)
            "max_exposure_per_symbol": 0.20 # 20% per symbol
        }
        self.returns_history: List[float] = []
        self._load_state()

    def update_config(self, config: Dict[str, float]):
        """Update risk limits."""
        self.limits.update(config)
        LOGGER.info(f"Risk Limits Updated: {self.limits}")

    def record_daily_return(self, daily_return: float):
        """Record a daily return for CVaR calculation."""
        self.returns_history.append(daily_return)
        # Keep last 252 days (1 trading year)
        if len(self.returns_history) > 252:
            self.returns_history.pop(0)
        self._save_state()

    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (CVaR) from history."""
        if not self.returns_history or len(self.returns_history) < 20:
            return 0.0
        
        losses = [-r for r in self.returns_history]
        var = np.quantile(losses, confidence)
        tail = [l for l in losses if l >= var]
        return np.mean(tail) if tail else var

    def check_exposure(self, positions: Dict[str, float], total_equity: float) -> Tuple[bool, str]:
        """Check if any single position exceeds the exposure limit.
        positions: Dict[symbol, market_value]
        """
        limit_pct = self.limits.get("max_exposure_per_symbol", 1.0)
        if total_equity <= 0:
            return True, "OK"

        for symbol, value in positions.items():
            exposure_pct = abs(value) / total_equity
            if exposure_pct > limit_pct:
                msg = f"EXPOSURE LIMIT BREACHED: {symbol} {exposure_pct:.2%} > {limit_pct:.2%}"
                return False, msg
        return True, "OK"

    def on_cycle_start(self, current_equity: float, holdings: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
        """Call at the start of every cycle to check limits.
        Returns: (is_safe, reason)
        """
        # 1. Daily Reset
        today = date.today()
        if today > self.current_date:
            # Record previous day's return
            if self.start_day_equity > 0:
                daily_ret = (current_equity - self.start_day_equity) / self.start_day_equity
                self.record_daily_return(daily_ret)
                LOGGER.info(f"Recorded Daily Return: {daily_ret:.4%}")

            LOGGER.info(f"New Day: Resetting Daily Loss Counter. Old Start Equity: {self.start_day_equity}, New: {current_equity}")
            self.start_day_equity = current_equity
            self.current_date = today
            self._save_state()
        
        # Initialize if first run
        if self.high_water_mark == 0.0:
            self.high_water_mark = current_equity
            self.start_day_equity = current_equity
            self._save_state()

        # 2. Update HWM
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
            self._save_state()

        # 3. Check Drawdown
        dd_pct = (self.high_water_mark - current_equity) / self.high_water_mark if self.high_water_mark > 0 else 0.0
        if dd_pct > self.limits["dd_limit"]:
            msg = f"MAX DRAWDOWN BREACHED: {dd_pct:.2%} > {self.limits['dd_limit']:.2%}. Stopping Robot."
            send_alert("RISK ALERT: Max Drawdown", msg)
            return False, msg

        # 4. Check Daily Loss
        daily_pl_pct = (current_equity - self.start_day_equity) / self.start_day_equity if self.start_day_equity > 0 else 0.0
        # daily_pl_pct is negative for loss
        if daily_pl_pct < -self.limits["daily_loss_cap"]:
            msg = f"DAILY LOSS CAP BREACHED: {daily_pl_pct:.2%} < -{self.limits['daily_loss_cap']:.2%}. Stopping Robot."
            send_alert("RISK ALERT: Daily Loss", msg)
            return False, msg

        # 5. Check CVaR
        cvar = self.calculate_cvar()
        if cvar > self.limits["cvar_limit"]:
            msg = f"CVaR LIMIT BREACHED: {cvar:.2%} > {self.limits['cvar_limit']:.2%}. Stopping Robot."
            send_alert("RISK ALERT: CVaR", msg)
            return False, msg

        # 6. Check Exposure (if holdings provided)
        if holdings:
            is_safe_exp, msg_exp = self.check_exposure(holdings, current_equity)
            if not is_safe_exp:
                send_alert("RISK ALERT: Exposure Limit", msg_exp)
                return False, msg_exp

        return True, "OK"

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.high_water_mark = data.get("high_water_mark", 0.0)
                    self.start_day_equity = data.get("start_day_equity", 0.0)
                    d_str = data.get("current_date")
                    if d_str:
                        self.current_date = datetime.strptime(d_str, "%Y-%m-%d").date()
                    self.returns_history = data.get("returns_history", [])
            except Exception as e:
                LOGGER.error(f"Failed to load risk state: {e}")

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "high_water_mark": self.high_water_mark,
            "start_day_equity": self.start_day_equity,
            "current_date": self.current_date.strftime("%Y-%m-%d"),
            "returns_history": self.returns_history
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

# Global Instance
GUARD = PortfolioGuard()
