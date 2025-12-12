"""
Transaction Cost Modeling.

Provides realistic transaction cost components for backtesting:
- SpreadSchedule: Time-of-day spread adjustments for illiquid hours
- MarketImpactModel: Square-root market impact model for slippage
- TickRounder: Tick size constraints for realistic price execution

Mathematical basis:
- Spread adjustment: spread(t) = base_spread * multiplier(hour_of_day)
- Market impact: slippage = λ * sqrt(order_size / avg_volume) * volatility
- Tick rounding: price = round(price / tick_size) * tick_size
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SpreadSchedule:
    """
    Time-of-day spread adjustment for illiquid hours.
    
    Spreads tend to widen during less liquid market hours (e.g., overnight,
    Asian session for USD pairs, weekend open for crypto).
    
    Usage:
        schedule = SpreadSchedule(
            base_spread_bps=5.0,
            hour_multipliers={
                0: 2.0,   # Midnight: 2x wider
                1: 2.0,
                2: 1.8,
                3: 1.8,
                8: 1.2,   # London open
                9: 1.0,   # Normal hours
                13: 1.0,  # NY open
                22: 1.5,  # After hours
                23: 1.8,
            }
        )
        
        adjusted_spread = schedule.get_spread(timestamp)
    """
    base_spread_bps: float = 5.0
    hour_multipliers: Dict[int, float] = None
    
    def __post_init__(self):
        if self.hour_multipliers is None:
            self.hour_multipliers = {
                0: 1.8, 1: 1.8, 2: 1.8, 3: 1.8, 4: 1.5, 5: 1.5,
                6: 1.3, 7: 1.2, 8: 1.1, 9: 1.0, 10: 1.0, 11: 1.0,
                12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.1, 17: 1.2,
                18: 1.3, 19: 1.4, 20: 1.5, 21: 1.6, 22: 1.7, 23: 1.8,
            }
    
    def get_spread(self, timestamp: pd.Timestamp) -> float:
        """Get spread in basis points for a given timestamp."""
        hour = timestamp.hour
        multiplier = self.hour_multipliers.get(hour, 1.0)
        return self.base_spread_bps * multiplier
    
    def get_spread_series(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        """Get spread series for a datetime index."""
        hours = timestamps.hour
        multipliers = hours.map(lambda h: self.hour_multipliers.get(h, 1.0))
        return pd.Series(self.base_spread_bps * multipliers, index=timestamps)


@dataclass
class MarketImpactModel:
    """
    Square-root market impact model for slippage estimation.
    
    Based on research showing market impact scales with square root of order size
    relative to average daily volume (Almgren et al., Kyle model).
    
    Slippage = impact_coeff * sqrt(order_size / avg_volume) * volatility
    
    Where:
    - order_size: Notional value of the trade
    - avg_volume: Average daily volume in same currency
    - volatility: Recent price volatility
    - impact_coeff: Market-specific calibration parameter
    
    Usage:
        model = MarketImpactModel(
            impact_coeff=0.1,      # Calibrate per market
            volume_lookback=20,    # Days for avg volume
            volatility_lookback=20 # Days for volatility
        )
        
        slippage_bps = model.calculate_impact(
            order_size=10000.0,
            avg_volume=1000000.0,
            volatility=0.02
        )
    """
    impact_coeff: float = 0.1
    volume_lookback: int = 20
    volatility_lookback: int = 20
    min_impact_bps: float = 0.1
    max_impact_bps: float = 50.0
    
    def calculate_impact(
        self,
        order_size: float,
        avg_volume: float,
        volatility: float
    ) -> float:
        """
        Calculate market impact slippage in basis points.
        
        Args:
            order_size: Notional value of order
            avg_volume: Average daily volume
            volatility: Recent volatility (fraction)
            
        Returns:
            Slippage cost in basis points
        """
        if avg_volume <= 0 or order_size <= 0:
            return self.min_impact_bps
        
        participation_rate = order_size / avg_volume
        impact = self.impact_coeff * np.sqrt(participation_rate) * volatility * 10000.0
        
        return np.clip(impact, self.min_impact_bps, self.max_impact_bps)
    
    def calculate_impact_series(
        self,
        df: pd.DataFrame,
        order_sizes: pd.Series,
        volume_col: str = "Volume",
        close_col: str = "Close"
    ) -> pd.Series:
        """
        Calculate market impact series for a backtest.
        
        Args:
            df: OHLCV DataFrame
            order_sizes: Series of order sizes (notional values)
            volume_col: Name of volume column
            close_col: Name of close price column
            
        Returns:
            Series of impact costs in basis points
        """
        if volume_col not in df.columns or close_col not in df.columns:
            LOGGER.warning(f"Missing {volume_col} or {close_col}, using min impact")
            return pd.Series(self.min_impact_bps, index=df.index)
        
        avg_volume = df[volume_col].rolling(
            window=self.volume_lookback,
            min_periods=1
        ).mean()
        
        returns = df[close_col].pct_change(fill_method=None).fillna(0.0)
        volatility = returns.rolling(
            window=self.volatility_lookback,
            min_periods=1
        ).std().fillna(0.01)
        
        impact_series = pd.Series(index=df.index, dtype=float)
        
        for idx in df.index:
            order_size = order_sizes.get(idx, 0.0)
            if order_size == 0:
                impact_series[idx] = 0.0
            else:
                impact_series[idx] = self.calculate_impact(
                    order_size=abs(order_size),
                    avg_volume=avg_volume[idx],
                    volatility=volatility[idx]
                )
        
        return impact_series


@dataclass
class TickRounder:
    """
    Tick size constraints for realistic price execution.
    
    Real markets enforce minimum price increments (tick sizes). Orders
    execute at discrete price levels, not arbitrary decimals.
    
    Examples:
    - US stocks: $0.01 for prices >= $1.00
    - Forex: 0.00001 for most pairs (half-pip)
    - Bitcoin: $0.01 on most exchanges
    - Futures: Varies by contract
    
    Usage:
        rounder = TickRounder(tick_size=0.01)
        
        # Round entry/exit prices
        entry_price = rounder.round_price(100.234)  # -> 100.23
        
        # Apply to limit orders
        limit_price = rounder.round_up(100.234)     # -> 100.24 (aggressive)
        limit_price = rounder.round_down(100.234)   # -> 100.23 (passive)
    """
    tick_size: float = 0.01
    
    def __post_init__(self):
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {self.tick_size}")
    
    def round_price(self, price: float) -> float:
        """Round price to nearest tick."""
        return round(price / self.tick_size) * self.tick_size
    
    def round_up(self, price: float) -> float:
        """Round price up to next tick (aggressive limit buy, passive limit sell)."""
        return np.ceil(price / self.tick_size) * self.tick_size
    
    def round_down(self, price: float) -> float:
        """Round price down to previous tick (passive limit buy, aggressive limit sell)."""
        return np.floor(price / self.tick_size) * self.tick_size
    
    def round_series(self, prices: pd.Series) -> pd.Series:
        """Round a series of prices to nearest tick."""
        return (prices / self.tick_size).round() * self.tick_size
    
    def round_up_series(self, prices: pd.Series) -> pd.Series:
        """Round a series of prices up to next tick."""
        return np.ceil(prices / self.tick_size) * self.tick_size
    
    def round_down_series(self, prices: pd.Series) -> pd.Series:
        """Round a series of prices down to previous tick."""
        return np.floor(prices / self.tick_size) * self.tick_size
    
    def get_spread_ticks(self, bid: float, ask: float) -> int:
        """Calculate spread in number of ticks."""
        return int(round((ask - bid) / self.tick_size))


class TransactionCostModel:
    """
    Integrated transaction cost model combining all components.
    
    Combines spread schedule, market impact, and tick constraints for
    comprehensive transaction cost modeling in backtests.
    
    Usage:
        cost_model = TransactionCostModel(
            spread_schedule=SpreadSchedule(base_spread_bps=5.0),
            impact_model=MarketImpactModel(impact_coeff=0.1),
            tick_rounder=TickRounder(tick_size=0.01)
        )
        
        # Calculate total cost for a trade
        total_cost_bps = cost_model.calculate_total_cost(
            timestamp=pd.Timestamp('2024-01-01 09:30:00'),
            order_size=10000.0,
            avg_volume=1000000.0,
            volatility=0.02,
            base_fee_bps=1.0
        )
    """
    
    def __init__(
        self,
        spread_schedule: Optional[SpreadSchedule] = None,
        impact_model: Optional[MarketImpactModel] = None,
        tick_rounder: Optional[TickRounder] = None
    ):
        self.spread_schedule = spread_schedule or SpreadSchedule()
        self.impact_model = impact_model or MarketImpactModel()
        self.tick_rounder = tick_rounder or TickRounder()
    
    def calculate_total_cost(
        self,
        timestamp: pd.Timestamp,
        order_size: float,
        avg_volume: float,
        volatility: float,
        base_fee_bps: float = 0.0
    ) -> float:
        """
        Calculate total transaction cost in basis points.
        
        Args:
            timestamp: Trade timestamp
            order_size: Notional order size
            avg_volume: Average daily volume
            volatility: Recent volatility
            base_fee_bps: Base exchange/broker fee
            
        Returns:
            Total cost in basis points (fee + spread + impact)
        """
        spread_cost = self.spread_schedule.get_spread(timestamp)
        impact_cost = self.impact_model.calculate_impact(
            order_size=order_size,
            avg_volume=avg_volume,
            volatility=volatility
        )
        
        total_cost = base_fee_bps + spread_cost + impact_cost
        return total_cost
    
    def apply_tick_constraints(
        self,
        entry_price: float,
        exit_price: float,
        direction: str = "long"
    ) -> tuple[float, float]:
        """
        Apply tick rounding to entry and exit prices.
        
        Args:
            entry_price: Intended entry price
            exit_price: Intended exit price
            direction: 'long' or 'short'
            
        Returns:
            Tuple of (rounded_entry, rounded_exit)
        """
        if direction == "long":
            rounded_entry = self.tick_rounder.round_up(entry_price)
            rounded_exit = self.tick_rounder.round_down(exit_price)
        else:
            rounded_entry = self.tick_rounder.round_down(entry_price)
            rounded_exit = self.tick_rounder.round_up(exit_price)
        
        return rounded_entry, rounded_exit
