#!/usr/bin/env python3
"""Run OpenQuant Robot Simulation (Linux Friendly).

This script runs the research pipeline (backtest simulation) which is compatible with Linux.
It does NOT attempt to connect to MetaTrader 5.

Steps:
1. Runs universe research (backtest)
2. Generates reports
3. Launches dashboard (optional)
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 Starting OpenQuant Robot Simulation (Linux Mode)...")
    
    # Ensure we are in the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    
    # 1. Run Universe Research
    print("\n[1/3] Running Universe Research (Backtest)...")
    try:
        from openquant.research.universe_runner import run_universe
        
        # Run with a small subset for speed
        report_path = run_universe(
            exchange="binance",
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframes=["1h"],
            optimize=True,
            optuna_trials=5,  # Reduced trials for speed
            run_wfo=False     # Skip WFO for speed
        )
        print(f"✅ Research completed. Report: {report_path}")
    except Exception as e:
        print(f"❌ Research failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    # 2. Show Results
    print("\n[2/3] Checking Results...")
    results_db = project_root / "data" / "results.duckdb"
    if results_db.exists():
        print(f"✅ Results saved to {results_db}")
    else:
        print("⚠️ No results database found.")
        
    # 3. Dashboard Info
    print("\n[3/3] Dashboard")
    print("To view the results, run:")
    print("  streamlit run openquant/gui/dashboard.py")
    
    print("\n🎉 Simulation Test Complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
