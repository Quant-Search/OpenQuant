"""
Test Alpaca Stock Discovery and Optimization.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from openquant.research.universe_runner import run_universe

def test_alpaca_research():
    print("\n--- Testing Alpaca Stock Research ---")
    
    # Run research on Alpaca exchange (Stocks)
    # Using top_n=3 for speed
    report = run_universe(
        exchange="alpaca", 
        top_n=3, 
        timeframes=["1d"], 
        strategies=["liquidity"],
        optuna_trials=10
    )
    
    print(f"\n✅ Research Complete. Report: {report}")

if __name__ == "__main__":
    test_alpaca_research()
