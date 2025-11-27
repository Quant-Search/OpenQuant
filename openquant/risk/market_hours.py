"""Market Hours Checker - Validates if markets are open for trading.

Supports multiple market types:
- Forex: 24/5 (Sunday 17:00 EST to Friday 17:00 EST)
- Crypto: 24/7 (always open)
- US Stocks: Mon-Fri 9:30-16:00 EST (with pre/post market options)

Usage:
    from openquant.risk.market_hours import MarketHours, MarketType
    
    # Check if forex is open
    checker = MarketHours(MarketType.FOREX)
    if checker.is_open():
        # Execute trade
        pass
    else:
        print(f"Market closed. Opens at: {checker.next_open()}")
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Optional, Tuple
import pytz  # type: ignore


class MarketType(Enum):
    """Supported market types with their trading schedules."""
    FOREX = "forex"      # 24/5 Sunday 17:00 EST - Friday 17:00 EST
    CRYPTO = "crypto"    # 24/7 always open
    US_STOCKS = "stocks" # Mon-Fri 9:30-16:00 EST (regular hours)
    US_EXTENDED = "extended"  # Mon-Fri 4:00-20:00 EST (extended hours)


@dataclass
class TradingSession:
    """Defines a trading session with open/close times."""
    # Days: 0=Monday, 6=Sunday
    open_day: int       # Day market opens (0-6)
    open_time: time     # Time market opens (in EST)
    close_day: int      # Day market closes (0-6)
    close_time: time    # Time market closes (in EST)


# Define market schedules in EST timezone
MARKET_SCHEDULES = {
    # Forex: Opens Sunday 17:00 EST, Closes Friday 17:00 EST
    MarketType.FOREX: TradingSession(
        open_day=6,       # Sunday
        open_time=time(17, 0),
        close_day=4,      # Friday
        close_time=time(17, 0)
    ),
    # Crypto: Always open (24/7)
    MarketType.CRYPTO: None,  # None = always open
    # US Stocks: Mon-Fri 9:30-16:00 EST
    MarketType.US_STOCKS: TradingSession(
        open_day=0,       # Monday (applies Mon-Fri)
        open_time=time(9, 30),
        close_day=4,      # Friday
        close_time=time(16, 0)
    ),
    # US Extended: Mon-Fri 4:00-20:00 EST
    MarketType.US_EXTENDED: TradingSession(
        open_day=0,
        open_time=time(4, 0),
        close_day=4,
        close_time=time(20, 0)
    ),
}

# EST timezone for market hours
EST = pytz.timezone("US/Eastern")


class MarketHours:
    """
    Checks if a market is currently open for trading.
    
    Uses EST timezone for all calculations (standard for US markets).
    """
    
    def __init__(self, market_type: MarketType = MarketType.FOREX):
        """Initialize with market type."""
        self.market_type = market_type
        self.schedule = MARKET_SCHEDULES.get(market_type)
    
    def _now_est(self) -> datetime:
        """Get current time in EST timezone."""
        return datetime.now(EST)
    
    def is_open(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if market is currently open.
        
        Args:
            check_time: Optional datetime to check (defaults to now)
        
        Returns:
            True if market is open, False otherwise
        """
        # Crypto is always open
        if self.schedule is None:
            return True
        
        # Get time in EST
        if check_time is None:
            now = self._now_est()
        else:
            now = check_time.astimezone(EST)
        
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        current_time = now.time()
        
        # Check based on market type
        if self.market_type == MarketType.FOREX:
            # Forex: Closed Saturday and Sunday before 17:00 EST
            if weekday == 5:  # Saturday - always closed
                return False
            if weekday == 6:  # Sunday - open after 17:00 EST
                return current_time >= self.schedule.open_time
            if weekday == 4:  # Friday - open until 17:00 EST
                return current_time < self.schedule.close_time
            # Mon-Thu: always open
            return True
        
        elif self.market_type in (MarketType.US_STOCKS, MarketType.US_EXTENDED):
            # Stocks: Only Mon-Fri within hours
            if weekday > 4:  # Weekend
                return False
            # Check time window
            return self.schedule.open_time <= current_time < self.schedule.close_time
        
        return False
    
    def next_open(self, from_time: Optional[datetime] = None) -> datetime:
        """
        Get next market open time.
        
        Args:
            from_time: Starting point for calculation (defaults to now)
        
        Returns:
            Datetime when market next opens (in EST)
        """
        if self.schedule is None:
            # Crypto: always open, return now
            return self._now_est() if from_time is None else from_time
        
        now = self._now_est() if from_time is None else from_time.astimezone(EST)
        
        # If already open, return now
        if self.is_open(now):
            return now
        
        # Calculate next open based on market type
        if self.market_type == MarketType.FOREX:
            # Next Sunday 17:00 EST
            days_until_sunday = (6 - now.weekday()) % 7
            if days_until_sunday == 0 and now.time() >= self.schedule.open_time:
                days_until_sunday = 7
            next_open = now.replace(
                hour=17, minute=0, second=0, microsecond=0
            ) + timedelta(days=days_until_sunday)
            return next_open
        
        elif self.market_type in (MarketType.US_STOCKS, MarketType.US_EXTENDED):
            # Next weekday at open time
            days_ahead = 1 if now.weekday() < 4 else (7 - now.weekday())
            if now.weekday() < 5 and now.time() < self.schedule.open_time:
                days_ahead = 0
            next_open = now.replace(
                hour=self.schedule.open_time.hour,
                minute=self.schedule.open_time.minute,
                second=0, microsecond=0
            ) + timedelta(days=days_ahead)
            return next_open
        
        return now
    
    def time_until_open(self) -> Optional[timedelta]:
        """Get time until market opens (None if already open)."""
        if self.is_open():
            return None
        return self.next_open() - self._now_est()

