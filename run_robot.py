"""Helper script to run the robot with environment variables."""
import os
import subprocess
from pathlib import Path

# Set environment variables
os.environ["OQ_MT5_TERMINAL"] = "C:/Program Files/MetaTrader/terminal64.exe"
os.environ["OQ_MT5_SERVER"] = "MetaQuotes-Demo"
os.environ["OQ_MT5_LOGIN"] = "10008295042"
os.environ["OQ_MT5_PASSWORD"] = "2aR@VaBb"

# Run the main script
script_path = Path(__file__).parent / "scripts" / "mt5_run_once.py"
python_exe = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"

subprocess.run([str(python_exe), str(script_path)], check=True)