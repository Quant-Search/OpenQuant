#!/usr/bin/env python3
"""
OpenQuant - Quick Start Script
Runs the robot CLI which automatically launches the dashboard.
"""
import os
import sys
import signal
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from openquant.utils.shutdown_handler import SHUTDOWN_HANDLER

def main():
    project_root = Path(__file__).parent
    script_path = project_root / "scripts" / "run_robot_cli.py"
    
    signal.signal(signal.SIGINT, SHUTDOWN_HANDLER)
    signal.signal(signal.SIGTERM, SHUTDOWN_HANDLER)
    
    print("Initializing OpenQuant...")
    
    args = [sys.executable, str(script_path)] + sys.argv[1:]
    
    try:
        subprocess.run(args, cwd=project_root, check=True)
    except KeyboardInterrupt:
        SHUTDOWN_HANDLER.shutdown()
    except Exception as e:
        print(f"Error: {e}")
        SHUTDOWN_HANDLER.shutdown()

if __name__ == "__main__":
    main()
