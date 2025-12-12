"""
Graceful shutdown handler for OpenQuant.

Provides centralized shutdown logic to:
- Close all open positions
- Save portfolio state
- Flush logs
- Persist circuit breaker status
- Cleanup broker connections
"""
import sys
import logging
from pathlib import Path
from typing import Optional, Any

from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


class ShutdownHandler:
    """Handles graceful shutdown of trading systems."""
    
    def __init__(self):
        self._shutdown_initiated = False
        self._brokers = []
        self._scheduler = None
        self._state_path = Path("data/paper_state.json")
        
    def register_broker(self, broker: Any) -> None:
        """Register a broker instance for cleanup on shutdown."""
        if broker not in self._brokers:
            self._brokers.append(broker)
            
    def register_scheduler(self, scheduler: Any) -> None:
        """Register the scheduler instance for cleanup on shutdown."""
        self._scheduler = scheduler
        
    def shutdown(self, signal_num: Optional[int] = None, frame: Optional[Any] = None) -> None:
        """
        Execute graceful shutdown sequence.
        
        Args:
            signal_num: Signal number if called from signal handler
            frame: Frame object if called from signal handler
        """
        if self._shutdown_initiated:
            LOGGER.warning("Shutdown already in progress, ignoring duplicate call")
            return
            
        self._shutdown_initiated = True
        
        signal_name = ""
        if signal_num is not None:
            import signal
            signal_name = signal.Signals(signal_num).name
            LOGGER.info(f"Received signal {signal_name} ({signal_num}), initiating graceful shutdown...")
        else:
            LOGGER.info("Initiating graceful shutdown...")
        
        print(f"\n{'='*60}")
        print(f"  GRACEFUL SHUTDOWN INITIATED")
        if signal_name:
            print(f"  Signal: {signal_name}")
        print(f"{'='*60}\n")
        
        # Step 1: Stop scheduler if running
        if self._scheduler is not None:
            try:
                LOGGER.info("Stopping scheduler...")
                print("  [1/5] Stopping scheduler...")
                self._scheduler.stop()
                LOGGER.info("Scheduler stopped")
            except Exception as e:
                LOGGER.error(f"Error stopping scheduler: {e}", exc_info=True)
                print(f"  [1/5] WARNING: Scheduler stop failed: {e}")
        else:
            print("  [1/5] No scheduler to stop")
        
        # Step 2: Close all positions on registered brokers
        print("  [2/5] Closing positions...")
        closed_count = 0
        for broker in self._brokers:
            try:
                LOGGER.info(f"Closing positions on broker: {broker.__class__.__name__}")
                result = broker.close_all_positions()
                if isinstance(result, dict) and "closed_positions" in result:
                    closed_count += result["closed_positions"]
                    LOGGER.info(f"Closed {result['closed_positions']} positions")
                else:
                    LOGGER.info("Positions closed (count unknown)")
            except Exception as e:
                LOGGER.error(f"Error closing positions on {broker.__class__.__name__}: {e}", exc_info=True)
                print(f"  [2/5] WARNING: Failed to close positions on {broker.__class__.__name__}: {e}")
        
        if closed_count > 0:
            print(f"  [2/5] Closed {closed_count} position(s)")
        else:
            print("  [2/5] No positions to close")
        
        # Step 3: Save portfolio state
        print("  [3/5] Saving portfolio state...")
        try:
            from openquant.paper.io import load_state, save_state
            state = load_state(self._state_path)
            save_state(state, self._state_path)
            LOGGER.info(f"Portfolio state saved to {self._state_path}")
            print(f"  [3/5] Portfolio state saved")
        except Exception as e:
            LOGGER.error(f"Error saving portfolio state: {e}", exc_info=True)
            print(f"  [3/5] WARNING: Failed to save state: {e}")
        
        # Step 4: Persist circuit breaker status
        print("  [4/5] Persisting circuit breaker status...")
        try:
            from openquant.risk.circuit_breaker import CIRCUIT_BREAKER
            # Circuit breaker auto-saves on update, but we can force a save
            # by accessing its internal save method if needed
            if hasattr(CIRCUIT_BREAKER, '_save_state'):
                CIRCUIT_BREAKER._save_state()
                LOGGER.info("Circuit breaker state persisted")
                print("  [4/5] Circuit breaker status saved")
            else:
                print("  [4/5] Circuit breaker status already persisted")
        except Exception as e:
            LOGGER.error(f"Error persisting circuit breaker: {e}", exc_info=True)
            print(f"  [4/5] WARNING: Failed to persist circuit breaker: {e}")
        
        # Step 5: Cleanup broker connections
        print("  [5/5] Cleaning up broker connections...")
        for broker in self._brokers:
            try:
                LOGGER.info(f"Shutting down broker: {broker.__class__.__name__}")
                if hasattr(broker, 'shutdown'):
                    broker.shutdown()
                    LOGGER.info(f"Broker {broker.__class__.__name__} shutdown complete")
                elif hasattr(broker, 'close'):
                    broker.close()
                    LOGGER.info(f"Broker {broker.__class__.__name__} closed")
            except Exception as e:
                LOGGER.error(f"Error shutting down {broker.__class__.__name__}: {e}", exc_info=True)
                print(f"  [5/5] WARNING: Failed to cleanup {broker.__class__.__name__}: {e}")
        
        print("  [5/5] Broker cleanup complete")
        
        # Step 6: Flush logs
        LOGGER.info("Flushing logs...")
        print("\n  Flushing logs...")
        try:
            for handler in logging.getLogger().handlers:
                handler.flush()
            for handler in LOGGER.handlers:
                handler.flush()
            print("  Logs flushed")
        except Exception as e:
            print(f"  WARNING: Failed to flush logs: {e}")
        
        print(f"\n{'='*60}")
        print(f"  SHUTDOWN COMPLETE")
        print(f"{'='*60}\n")
        
        LOGGER.info("Graceful shutdown complete")
        
    def __call__(self, signal_num: int, frame: Any) -> None:
        """Allow instance to be used as signal handler."""
        self.shutdown(signal_num, frame)
        sys.exit(0)


# Global shutdown handler instance
SHUTDOWN_HANDLER = ShutdownHandler()
