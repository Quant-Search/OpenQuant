"""Trading Session Optimizer.

Determines optimal trading times based on:
- Market sessions (London, NY, Tokyo overlap)
- Historical volatility patterns
- Spread patterns
- Day of week effects
"""
from datetime import datetime, time as dtime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import pytz

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

class MarketSession(Enum):
    """Major forex market sessions."""
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    SYDNEY = "sydney"

@dataclass
class TradingWindow:
    """A recommended trading window."""
    start_hour: int  # UTC hour
    end_hour: int    # UTC hour
    session_name: str
    score: float     # 0-100 quality score
    reason: str

# Session times in UTC
SESSION_TIMES = {
    MarketSession.TOKYO: (0, 9),    # 00:00 - 09:00 UTC
    MarketSession.LONDON: (7, 16),  # 07:00 - 16:00 UTC
    MarketSession.NEW_YORK: (12, 21),  # 12:00 - 21:00 UTC
    MarketSession.SYDNEY: (21, 6),  # 21:00 - 06:00 UTC (next day)
}

# Optimal overlaps (highest liquidity)
OPTIMAL_WINDOWS = [
    TradingWindow(7, 9, "London-Tokyo", 90, "High liquidity overlap"),
    TradingWindow(12, 16, "London-NY", 95, "Best liquidity, major moves"),
    TradingWindow(8, 11, "London Morning", 85, "Strong trends"),
    TradingWindow(13, 15, "NY Morning", 80, "High volume"),
]

# Low activity periods to avoid
AVOID_PERIODS = [
    (22, 23),  # Late NY, low volume
    (4, 6),    # Dead zone before London
]

# Day of week effects (0 = Monday, 4 = Friday)
DAY_QUALITY = {
    0: 0.85,  # Monday: Often reversals
    1: 1.0,   # Tuesday: Best trends
    2: 1.0,   # Wednesday: Best trends
    3: 0.95,  # Thursday: Good
    4: 0.75,  # Friday: Risk of weekend gaps
    5: 0.0,   # Saturday: Closed
    6: 0.3,   # Sunday: Very low volume
}

class SessionOptimizer:
    """
    Optimizes trading based on session timing.
    
    Features:
    - Identifies optimal trading windows
    - Avoids low-liquidity periods
    - Considers day-of-week effects
    - Tracks historical performance by session
    """
    
    def __init__(self, timezone: str = "UTC"):
        self.timezone = pytz.timezone(timezone)
        self.performance_by_session: Dict[str, List[float]] = {
            "tokyo": [],
            "london": [],
            "new_york": [],
            "overlap_london_ny": []
        }
        
    def get_current_session(self, dt: Optional[datetime] = None) -> Tuple[str, float]:
        """
        Get current trading session and quality score.
        
        Returns:
            (session_name, quality_score)
        """
        if dt is None:
            dt = datetime.now(pytz.UTC)
        elif dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        else:
            dt = dt.astimezone(pytz.UTC)
            
        hour = dt.hour
        weekday = dt.weekday()
        
        # Base day quality
        day_quality = DAY_QUALITY.get(weekday, 0.5)
        
        if day_quality == 0:
            return "closed", 0.0
            
        # Determine session
        session = "off_hours"
        session_quality = 0.3
        
        # Check overlaps first (highest quality)
        if 12 <= hour < 16:
            session = "overlap_london_ny"
            session_quality = 1.0
        elif 7 <= hour < 9:
            session = "overlap_tokyo_london"
            session_quality = 0.85
        # Then individual sessions
        elif 0 <= hour < 9:
            session = "tokyo"
            session_quality = 0.6
        elif 7 <= hour < 16:
            session = "london"
            session_quality = 0.8
        elif 12 <= hour < 21:
            session = "new_york"
            session_quality = 0.75
            
        # Check avoid periods
        for start, end in AVOID_PERIODS:
            if start <= hour < end:
                session_quality *= 0.5
                session = f"{session}_low_volume"
                break
                
        final_quality = session_quality * day_quality
        
        return session, final_quality
        
    def should_trade_now(
        self,
        min_quality: float = 0.5,
        dt: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Determine if now is a good time to trade.
        
        Args:
            min_quality: Minimum quality score to trade (0-1)
            dt: Optional datetime (defaults to now)
            
        Returns:
            (should_trade, reason)
        """
        session, quality = self.get_current_session(dt)
        
        if quality >= min_quality:
            return True, f"Good session: {session} (quality: {quality:.0%})"
        else:
            return False, f"Low quality session: {session} (quality: {quality:.0%})"
            
    def get_next_optimal_window(
        self,
        dt: Optional[datetime] = None
    ) -> TradingWindow:
        """
        Get the next optimal trading window.
        
        Returns:
            TradingWindow for next best opportunity
        """
        if dt is None:
            dt = datetime.now(pytz.UTC)
        elif dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
            
        current_hour = dt.hour
        
        # Find next window
        for window in sorted(OPTIMAL_WINDOWS, key=lambda w: w.score, reverse=True):
            if current_hour < window.start_hour:
                # Window is later today
                return window
            elif current_hour >= window.end_hour:
                # Window passed, will be tomorrow
                continue
            else:
                # We're in this window!
                return window
                
        # Return best window for tomorrow
        return max(OPTIMAL_WINDOWS, key=lambda w: w.score)
        
    def get_session_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by session."""
        stats = {}
        for session, returns in self.performance_by_session.items():
            if returns:
                stats[session] = {
                    "avg_return": sum(returns) / len(returns),
                    "win_rate": sum(1 for r in returns if r > 0) / len(returns),
                    "trade_count": len(returns)
                }
            else:
                stats[session] = {"avg_return": 0, "win_rate": 0, "trade_count": 0}
        return stats
        
    def record_trade_result(self, pnl: float, dt: Optional[datetime] = None):
        """Record trade result for session performance tracking."""
        session, _ = self.get_current_session(dt)
        
        # Normalize session name
        if "london" in session and "ny" in session:
            session = "overlap_london_ny"
        elif "london" in session:
            session = "london"
        elif "ny" in session or "york" in session:
            session = "new_york"
        elif "tokyo" in session:
            session = "tokyo"
            
        if session in self.performance_by_session:
            self.performance_by_session[session].append(pnl)
            # Keep last 100
            self.performance_by_session[session] = self.performance_by_session[session][-100:]
            
    def get_recommended_schedule(self) -> Dict[str, Any]:
        """
        Get recommended trading schedule.
        
        Returns dict with recommended hours and days.
        """
        return {
            "best_hours_utc": [(12, 16), (8, 11)],  # London-NY overlap, London morning
            "good_hours_utc": [(7, 9), (13, 15)],   # Tokyo-London overlap, NY morning
            "avoid_hours_utc": [(22, 6)],           # Low volume period
            "best_days": ["Tuesday", "Wednesday"],
            "avoid_days": ["Friday afternoon", "Sunday", "Monday morning"],
            "optimal_windows": [
                {
                    "name": w.session_name,
                    "hours": f"{w.start_hour:02d}:00 - {w.end_hour:02d}:00 UTC",
                    "score": w.score,
                    "reason": w.reason
                }
                for w in sorted(OPTIMAL_WINDOWS, key=lambda x: x.score, reverse=True)
            ]
        }
