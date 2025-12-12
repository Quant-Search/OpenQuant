"""Performance Tracker for Real-Time Profit Monitoring.

Tracks every trade's P&L and calculates rolling metrics.
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import threading

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

@dataclass
class TradeRecord:
    """Record of a completed trade."""
    timestamp: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_usd: float
    pnl_pct: float
    duration_seconds: float
    strategy: str = "unknown"
    
class PerformanceTracker:
    """
    Tracks trading performance in real-time.
    
    Features:
    - Records all trades
    - Calculates win rate, expectancy, Sharpe
    - Tracks equity curve and drawdown
    - Persists to disk for dashboard
    """
    
    def __init__(self, data_path: Path = Path("data/performance.json")):
        self.data_path = data_path
        self.trades: List[TradeRecord] = []
        self._lock = threading.Lock()
        
        # Equity tracking
        self.initial_equity = 0.0
        self.current_equity = 0.0
        self.peak_equity = 0.0
        self.equity_curve: List[Dict[str, Any]] = []
        
        # Load existing data
        self._load()
        
    def _load(self):
        """Load data from disk."""
        if self.data_path.exists():
            try:
                with open(self.data_path, "r") as f:
                    data = json.load(f)
                    self.trades = [TradeRecord(**t) for t in data.get("trades", [])]
                    self.initial_equity = data.get("initial_equity", 0.0)
                    self.current_equity = data.get("current_equity", 0.0)
                    self.peak_equity = data.get("peak_equity", 0.0)
                    self.equity_curve = data.get("equity_curve", [])
            except Exception as e:
                LOGGER.warning(f"Failed to load performance data: {e}")
                
    def _save(self):
        """Save data to disk."""
        try:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_path, "w") as f:
                json.dump({
                    "trades": [asdict(t) for t in self.trades],
                    "initial_equity": self.initial_equity,
                    "current_equity": self.current_equity,
                    "peak_equity": self.peak_equity,
                    "equity_curve": self.equity_curve[-1000:],  # Keep last 1000 points
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            LOGGER.error(f"Failed to save performance data: {e}")
            
    def set_initial_equity(self, equity: float):
        """Set initial equity (call once at start)."""
        with self._lock:
            if self.initial_equity == 0:
                self.initial_equity = equity
                self.current_equity = equity
                self.peak_equity = equity
                self._save()
                
    def update_equity(self, equity: float):
        """Update current equity."""
        with self._lock:
            self.current_equity = equity
            if equity > self.peak_equity:
                self.peak_equity = equity
                
            # Add to equity curve
            self.equity_curve.append({
                "timestamp": datetime.now().isoformat(),
                "equity": equity,
                "drawdown": self.get_current_drawdown()
            })
            self._save()
            
    def record_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        duration_seconds: float = 0,
        strategy: str = "unknown"
    ):
        """Record a completed trade."""
        with self._lock:
            # Calculate P&L
            if side.upper() == "LONG":
                pnl_usd = (exit_price - entry_price) * quantity
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_usd = (entry_price - exit_price) * quantity
                pnl_pct = (entry_price - exit_price) / entry_price
                
            trade = TradeRecord(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                duration_seconds=duration_seconds,
                strategy=strategy
            )
            
            self.trades.append(trade)
            LOGGER.info(f"Trade recorded: {symbol} {side} P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2%})")
            self._save()
            
    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity
        
    def get_daily_pnl(self) -> float:
        """Get today's P&L in USD."""
        today = datetime.now().date()
        daily_pnl = 0.0
        
        for trade in self.trades:
            try:
                trade_date = datetime.fromisoformat(trade.timestamp).date()
                if trade_date == today:
                    daily_pnl += trade.pnl_usd
            except Exception:
                pass
                
        return daily_pnl
        
    def get_stats(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Get performance statistics."""
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        recent_trades = []
        for trade in self.trades:
            try:
                trade_date = datetime.fromisoformat(trade.timestamp)
                if trade_date >= cutoff:
                    recent_trades.append(trade)
            except Exception:
                pass
                
        if not recent_trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "expectancy": 0.0,
                "total_pnl": 0.0,
                "daily_avg_pnl": 0.0,
                "max_drawdown": 0.0,
                "current_drawdown": self.get_current_drawdown(),
                "profit_factor": 0.0,
                "sharpe_estimate": 0.0
            }
            
        # Calculate stats
        wins = [t for t in recent_trades if t.pnl_usd > 0]
        losses = [t for t in recent_trades if t.pnl_usd < 0]
        
        total_trades = len(recent_trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        avg_win = sum(t.pnl_usd for t in wins) / len(wins) if wins else 0
        avg_loss = sum(abs(t.pnl_usd) for t in losses) / len(losses) if losses else 0
        
        total_pnl = sum(t.pnl_usd for t in recent_trades)
        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = sum(abs(t.pnl_usd) for t in losses)
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Estimate Sharpe
        returns = [t.pnl_pct for t in recent_trades]
        if len(returns) > 1:
            import numpy as np
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_estimate = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe_estimate = 0
            
        # Calculate max drawdown from equity curve
        max_dd = 0.0
        peak = 0.0
        for point in self.equity_curve:
            eq = point.get("equity", 0)
            if eq > peak:
                peak = eq
            if peak > 0:
                dd = (peak - eq) / peak
                max_dd = max(max_dd, dd)
                
        return {
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "total_pnl": total_pnl,
            "daily_avg_pnl": total_pnl / lookback_days,
            "current_drawdown": self.get_current_drawdown(),
            "max_drawdown": max_dd,
            "profit_factor": profit_factor,
            "sharpe_estimate": sharpe_estimate,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "initial_equity": self.initial_equity,
            "total_return_pct": (self.current_equity - self.initial_equity) / self.initial_equity if self.initial_equity > 0 else 0
        }

# Global instance
PERFORMANCE_TRACKER = PerformanceTracker()
