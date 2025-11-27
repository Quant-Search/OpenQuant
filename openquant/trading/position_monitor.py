"""Active position monitoring for live trading.

This module provides continuous surveillance of open positions,
adjusting TP/SL levels and detecting scaling opportunities.
"""
from __future__ import annotations
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
import time

from ..utils.logging import get_logger
from ..risk.trailing_stop import TrailingStopManager, PositionInfo

LOGGER = get_logger(__name__)


@dataclass
class PositionMetrics:
    """Real-time metrics for an open position."""
    symbol: str
    ticket: int
    unrealized_pnl_pct: float  # P&L as percentage of entry price
    unrealized_pnl_usd: float  # P&L in USD
    current_price: float
    entry_price: float
    volume: float
    side: str  # "LONG" or "SHORT"
    age_seconds: float  # Time since position opened
    current_sl: float
    current_tp: float


class PositionMonitor:
    """Continuously monitors and manages open positions."""
    
    def __init__(
        self,
        check_interval_seconds: int = 60,
        trailing_stop_manager: Optional[TrailingStopManager] = None,
        on_position_update: Optional[Callable[[PositionMetrics], None]] = None
    ):
        """Initialize the position monitor.
        
        Args:
            check_interval_seconds: How often to check positions (default: 60s)
            trailing_stop_manager: Optional trailing stop manager
            on_position_update: Optional callback when position metrics are updated
        """
        self.check_interval = check_interval_seconds
        self.trailing_manager = trailing_stop_manager or TrailingStopManager()
        self.on_position_update = on_position_update
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
    def start(self, mt5_broker=None):
        """Start the monitoring loop in a background thread.
        
        Args:
            mt5_broker: MT5Broker instance (required for monitoring)
        """
        if self._running:
            LOGGER.warning("Position monitor already running")
            return
            
        if mt5_broker is None:
            raise ValueError("MT5Broker instance required for position monitoring")
            
        self.mt5_broker = mt5_broker
        self._running = True
        self._stop_event.clear()
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        LOGGER.info(f"Position monitor started (interval: {self.check_interval}s)")
        
    def stop(self):
        """Stop the monitoring loop."""
        if not self._running:
            return
            
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=5)
            
        LOGGER.info("Position monitor stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._check_positions()
            except Exception as e:
                LOGGER.error(f"Error in position monitor: {e}", exc_info=True)
                
            # Sleep in small increments to be responsive to stop signal
            for _ in range(self.check_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
                
    def _check_positions(self):
        """Check all positions and update TP/SL as needed."""
        try:
            # Get current positions
            positions = self.mt5_broker.get_positions()
            
            if not positions:
                return
                
            LOGGER.debug(f"Monitoring {len(positions)} positions")
            
            # Check each position
            for symbol, qty in positions.items():
                if qty == 0:
                    continue
                    
                try:
                    metrics = self._calculate_metrics(symbol, qty)
                    
                    if metrics:
                        # Call user callback if provided
                        if self.on_position_update:
                            self.on_position_update(metrics)
                            
                        # Update trailing stops
                        self._update_trailing_stop(symbol)
                        
                except Exception as e:
                    LOGGER.error(f"Error checking position {symbol}: {e}")
                    
        except Exception as e:
            LOGGER.error(f"Error getting positions: {e}")
            
    def _calculate_metrics(self, symbol: str, qty: float) -> Optional[PositionMetrics]:
        """Calculate real-time metrics for a position."""
        try:
            # Get position details from MT5
            mt5 = self.mt5_broker.mt5
            positions = mt5.positions_get(symbol=symbol)
            
            if not positions:
                return None
                
            # Aggregate position data (in case of multiple tickets for same symbol)
            total_volume = 0.0
            weighted_entry = 0.0
            latest_ticket = 0
            sl = 0.0
            tp = 0.0
            open_time = None
            
            for pos in positions:
                vol = float(pos.volume)
                total_volume += vol
                weighted_entry += float(pos.price_open) * vol
                
                if int(pos.ticket) > latest_ticket:
                    latest_ticket = int(pos.ticket)
                    sl = float(pos.sl)
                    tp = float(pos.tp)
                    open_time = datetime.fromtimestamp(int(pos.time))
                    
            avg_entry = weighted_entry / total_volume if total_volume > 0 else 0.0
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return None
                
            # Use bid for longs (sell to close), ask for shorts (buy to close)
            current_price = float(tick.bid if qty > 0 else tick.ask)
            
            # Calculate P&L
            if qty > 0:  # LONG
                pnl_pct = ((current_price - avg_entry) / avg_entry) * 100
                side = "LONG"
            else:  # SHORT
                pnl_pct = ((avg_entry - current_price) / avg_entry) * 100
                side = "SHORT"
                
            # Get contract size for USD calculation
            info = mt5.symbol_info(symbol)
            contract_size = float(getattr(info, "trade_contract_size", 1.0) or 1.0)
            
            pnl_usd = pnl_pct / 100 * avg_entry * abs(total_volume) * contract_size
            
            # Calculate age
            age_seconds = (datetime.now() - open_time).total_seconds() if open_time else 0
            
            return PositionMetrics(
                symbol=symbol,
                ticket=latest_ticket,
                unrealized_pnl_pct=pnl_pct,
                unrealized_pnl_usd=pnl_usd,
                current_price=current_price,
                entry_price=avg_entry,
                volume=total_volume,
                side=side,
                age_seconds=age_seconds,
                current_sl=sl,
                current_tp=tp
            )
            
        except Exception as e:
            LOGGER.error(f"Error calculating metrics for {symbol}: {e}")
            return None
            
    def _update_trailing_stop(self, symbol: str):
        """Update trailing stop for a symbol if needed."""
        try:
            if self.trailing_manager:
                mt5 = self.mt5_broker.mt5
                results = self.trailing_manager.update_mt5_positions(mt5)
                
                if symbol in results and results[symbol]:
                    LOGGER.info(f"Updated trailing stop for {symbol}")
                    
        except Exception as e:
            LOGGER.error(f"Error updating trailing stop for {symbol}: {e}")
            
    def get_all_metrics(self) -> List[PositionMetrics]:
        """Get current metrics for all positions.
        
        Returns:
            List of PositionMetrics for all open positions
        """
        metrics = []
        
        try:
            positions = self.mt5_broker.get_positions()
            
            for symbol, qty in positions.items():
                if qty == 0:
                    continue
                    
                m = self._calculate_metrics(symbol, qty)
                if m:
                    metrics.append(m)
                    
        except Exception as e:
            LOGGER.error(f"Error getting all metrics: {e}")
            
        return metrics
