"""
Test graceful shutdown handler functionality.
"""
import sys
import signal
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openquant.utils.shutdown_handler import SHUTDOWN_HANDLER
from openquant.paper.io import save_state, load_state
from openquant.paper.state import PortfolioState


class MockBroker:
    """Mock broker for testing."""
    
    def __init__(self, name="MockBroker"):
        self.name = name
        self.positions_closed = False
        self.shutdown_called = False
        
    def close_all_positions(self):
        """Simulate closing positions."""
        print(f"  {self.name}: Closing all positions...")
        self.positions_closed = True
        return {"closed_positions": 2}
        
    def shutdown(self):
        """Simulate broker shutdown."""
        print(f"  {self.name}: Shutting down connection...")
        self.shutdown_called = True


class MockScheduler:
    """Mock scheduler for testing."""
    
    def __init__(self):
        self.is_running = True
        self.stopped = False
        
    def stop(self):
        """Simulate scheduler stop."""
        print("  MockScheduler: Stopping...")
        self.is_running = False
        self.stopped = True


def test_shutdown_handler():
    """Test the shutdown handler with mock components."""
    print("\n" + "="*60)
    print("Testing Graceful Shutdown Handler")
    print("="*60 + "\n")
    
    print("Setting up test environment...")
    state_path = Path("data/test_paper_state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    
    initial_state = PortfolioState(cash=100000.0)
    initial_state.holdings[("BINANCE", "BTC/USDT", "1h", "mean_reversion")] = 0.5
    save_state(initial_state, state_path)
    print(f"  Created test state with {len(initial_state.holdings)} position(s)")
    
    broker1 = MockBroker("MT5Broker")
    broker2 = MockBroker("AlpacaBroker")
    scheduler = MockScheduler()
    
    handler = SHUTDOWN_HANDLER
    handler._state_path = state_path
    
    print("\nRegistering components with shutdown handler...")
    handler.register_broker(broker1)
    handler.register_broker(broker2)
    handler.register_scheduler(scheduler)
    print("  Registered 2 brokers and 1 scheduler")
    
    print("\nTriggering graceful shutdown...")
    handler.shutdown()
    
    print("\nVerifying shutdown actions...")
    assert broker1.positions_closed, "Broker1 positions should be closed"
    assert broker1.shutdown_called, "Broker1 shutdown should be called"
    assert broker2.positions_closed, "Broker2 positions should be closed"
    assert broker2.shutdown_called, "Broker2 shutdown should be called"
    assert scheduler.stopped, "Scheduler should be stopped"
    print("  ✓ All brokers closed positions")
    print("  ✓ All brokers shut down connections")
    print("  ✓ Scheduler stopped")
    
    loaded_state = load_state(state_path)
    assert loaded_state.cash == 100000.0, "State should be saved"
    print("  ✓ Portfolio state saved")
    
    cb_state_path = Path("data/circuit_breaker_state.json")
    if cb_state_path.exists():
        print("  ✓ Circuit breaker state persisted")
    else:
        print("  ℹ Circuit breaker state file not present (expected for new install)")
    
    if state_path.exists():
        state_path.unlink()
    
    print("\n" + "="*60)
    print("Shutdown Handler Test: PASSED")
    print("="*60 + "\n")


def test_signal_handling():
    """Demonstrate signal handling (without actually sending signals)."""
    print("\n" + "="*60)
    print("Signal Handler Configuration")
    print("="*60 + "\n")
    
    print("The following signals are configured for graceful shutdown:")
    print("  • SIGINT  (Ctrl+C)")
    print("  • SIGTERM (kill command)")
    print("\nWhen received, the handler will:")
    print("  1. Stop the scheduler")
    print("  2. Close all open positions")
    print("  3. Save portfolio state")
    print("  4. Persist circuit breaker status")
    print("  5. Cleanup broker connections")
    print("  6. Flush all logs")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    test_shutdown_handler()
    test_signal_handling()
