#!/usr/bin/env python3
"""OpenQuant Robot - CLI Runner.
Runs the robot in headless mode using the shared Scheduler.
"""
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openquant.gui.scheduler import SCHEDULER
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)

DASHBOARD_PROCESS = None

def signal_handler(sig, frame):
    print("\n🛑 Stopping Robot (Ctrl+C)...")
    SCHEDULER.stop()
    if DASHBOARD_PROCESS:
        DASHBOARD_PROCESS.terminate()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="OpenQuant Robot CLI")
    parser.add_argument("--interval", type=int, default=60, help="Run interval in minutes")
    parser.add_argument("--top-n", type=int, default=100, help="Number of symbols to trade")
    parser.add_argument("--mt5", action="store_true", help="Enable MT5 integration")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable auto-launch of dashboard")
    args = parser.parse_args()

    print("🚀 Starting OpenQuant Robot (CLI Mode)")
    print(f"   Interval: {args.interval}m")
    print(f"   Top N:    {args.top_n}")
    print(f"   MT5:      {'Enabled' if args.mt5 else 'Disabled'}")
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Launch Dashboard
    if not args.no_dashboard:
        print("📊 Launching Dashboard...")
        global DASHBOARD_PROCESS
        
        # Log file for dashboard
        log_file = open(project_root / "dashboard.log", "w")
        
        # Remove headless=true so it might open, but we also force it below
        cmd = [sys.executable, "-m", "streamlit", "run", "openquant/gui/dashboard.py", "--server.headless", "false", "--server.port", "8501"]
        try:
            DASHBOARD_PROCESS = subprocess.Popen(cmd, cwd=project_root, stdout=log_file, stderr=log_file)
            print(f"   Dashboard process started (PID: {DASHBOARD_PROCESS.pid})")
            print(f"   Logs: {project_root / 'dashboard.log'}")
            
            # Give it a moment to start, then open browser
            time.sleep(2)
            import webbrowser
            webbrowser.open("http://localhost:8501")
            print(f"   Opened http://localhost:8501")
            
        except Exception as e:
            print(f"   Failed to launch dashboard: {e}")

    # Configure
    config = {
        "top_n": args.top_n,
        "use_mt5": args.mt5
    }
    
    # Start Scheduler
    SCHEDULER.start(interval_minutes=args.interval, config=config)
    
    # Monitor Loop
    try:
        while True:
            status = SCHEDULER.status_message
            err = SCHEDULER.error_message
            
            # Clear line and print status
            sys.stdout.write(f"\r[Status] {status} " + (f"| ⚠️ {err}" if err else ""))
            sys.stdout.flush()
            
            time.sleep(1)
            
            if not SCHEDULER.is_running:
                print("\nScheduler stopped unexpectedly.")
                break
                
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
