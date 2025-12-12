from __future__ import annotations
"""Trailing Stop Manager for monitoring and adjusting Stop Loss levels.

This module implements a system to automatically adjust Stop Loss (SL) levels
as the price moves in favor of the trade, protecting profits while allowing
the position to continue running.

Mathematical Basis:
- For LONG positions: New SL = Current Price × (1 - trailing_bps / 10000)
  The SL trails below the current price. We only update if New SL > Current SL.
  
- For SHORT positions: New SL = Current Price × (1 + trailing_bps / 10000)
  The SL trails above the current price. We only update if New SL < Current SL.

Example:
  - Long position at 1.0000, trailing_bps = 100 (1%)
  - Price rises to 1.1000
  - New SL = 1.1000 × (1 - 0.01) = 1.0890
  - If current SL is 1.0000, we update to 1.0890 (trailing upward)
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import time


@dataclass
class PositionInfo:
    """Information about an open position."""
    symbol: str  # Symbol name
    ticket: int  # Position ticket
    volume: float  # Position size (positive for long, negative for short)
    type: int  # 0 = Buy/Long, 1 = Sell/Short
    sl: float  # Current Stop Loss
    tp: float  # Current Take Profit
    open_price: float  # Entry price


class TrailingStopManager:
    """Manages trailing stop loss for open positions.
    
    This class monitors positions and updates their Stop Loss levels
    to lock in profits as the price moves favorably.
    """
    
    def __init__(
        self,
        trailing_bps: float = 50.0,
        activation_bps: float = 0.0,
        min_update_bps: float = 5.0,
        atr_multiplier: Optional[float] = None,
        acceleration_factor: float = 0.0
    ):
        """Initialize the Trailing Stop Manager.
        
        Args:
            trailing_bps: Distance (in basis points) to trail behind the price.
            activation_bps: Minimum profit (in basis points) before trailing starts.
            min_update_bps: Minimum change required (in bps) to trigger an SL update.
            atr_multiplier: If set, uses ATR * multiplier as trailing distance instead of bps.
            acceleration_factor: If > 0, reduces trailing distance as profit increases.
                                 New Distance = Base Distance * (1 - acceleration * profit_ratio)
        """
        self.trailing_bps = trailing_bps
        self.activation_bps = activation_bps
        self.min_update_bps = min_update_bps
        self.atr_multiplier = atr_multiplier
        self.acceleration_factor = acceleration_factor
        
    def calculate_new_sl(
        self,
        pos: PositionInfo,
        current_price: float,
        atr: Optional[float] = None
    ) -> Optional[float]:
        """Calculate new Stop Loss level for a position.
        
        Args:
            pos: Position information
            current_price: Current market price
            atr: Current Average True Range (required if atr_multiplier is set)
            
        Returns:
            New SL level, or None if no update needed
        """
        is_long = pos.type == 0
        
        # Step 1: Check activation threshold
        if is_long:
            profit_bps = ((current_price - pos.open_price) / pos.open_price) * 10000
        else:
            profit_bps = ((pos.open_price - current_price) / pos.open_price) * 10000
            
        if profit_bps < self.activation_bps:
            return None
            
        # Step 2: Determine trailing distance
        distance = 0.0
        
        if self.atr_multiplier and atr:
            # ATR-based trailing
            distance = atr * self.atr_multiplier
        else:
            # Percentage-based trailing
            distance = current_price * (self.trailing_bps / 10000.0)
            
        # Apply acceleration (reduce distance as profit increases)
        if self.acceleration_factor > 0:
            # Profit ratio (e.g. 0.01 for 1% profit)
            profit_ratio = profit_bps / 10000.0
            # Reduce distance: Distance * (1 - accel * profit)
            # e.g. accel=5.0, profit=2% -> factor = 1 - 0.1 = 0.9
            factor = max(0.2, 1.0 - (self.acceleration_factor * profit_ratio))
            distance *= factor
            
        # Step 3: Calculate new SL
        if is_long:
            new_sl = current_price - distance
            
            # Only update if new SL is higher (tightening)
            if pos.sl > 0 and new_sl <= pos.sl:
                return None
                
            # Check min update
            if pos.sl > 0:
                improvement_bps = ((new_sl - pos.sl) / pos.sl) * 10000
                if improvement_bps < self.min_update_bps:
                    return None
        else:
            new_sl = current_price + distance
            
            # Only update if new SL is lower (tightening)
            if pos.sl > 0 and new_sl >= pos.sl:
                return None
                
            # Check min update
            if pos.sl > 0:
                improvement_bps = ((pos.sl - new_sl) / pos.sl) * 10000
                if improvement_bps < self.min_update_bps:
                    return None
                    
        return new_sl
        
    def update_mt5_positions(self, mt5_module) -> Dict[str, bool]:
        """Update trailing stops for all MT5 positions.
        
        This method:
        1. Fetches all open positions from MT5
        2. For each position, calculates if SL should be updated
        3. Calls modify_position to update the SL
        
        Args:
            mt5_module: The MT5 module (lazy imported)
            
        Returns:
            Dict mapping symbol to update success status
        """
        from ..paper.mt5_bridge import modify_position
        
        results: Dict[str, bool] = {}
        
        # Get all open positions
        positions = mt5_module.positions_get()
        if not positions:
            return results
            
        # Process each position
        for pos_raw in positions:
            # Convert MT5 position to PositionInfo
            pos = PositionInfo(
                symbol=str(pos_raw.symbol),
                ticket=int(pos_raw.ticket),
                volume=float(pos_raw.volume),
                type=int(pos_raw.type),
                sl=float(pos_raw.sl),
                tp=float(pos_raw.tp),
                open_price=float(pos_raw.price_open)
            )
            
            # Get current price
            tick = mt5_module.symbol_info_tick(pos.symbol)
            if not tick:
                continue
                
            # Use Ask for Long (exit at Ask), Bid for Short (exit at Bid)
            current_price = float(getattr(tick, 'bid' if pos.type == 0 else 'ask', 0.0))
            if current_price <= 0:
                continue
                
            # Calculate new SL
            new_sl = self.calculate_new_sl(pos, current_price)
            if new_sl is None:
                continue
                
            # Update the position
            success = modify_position(pos.symbol, sl=new_sl, tp=None)
            results[pos.symbol] = success
            
        return results
