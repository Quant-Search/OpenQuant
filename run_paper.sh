#!/bin/bash
# OpenQuant Paper Trading Launcher (Linux)

# Ensure we are in the project root
cd "$(dirname "$0")"

# Check if .venv exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Warning: .venv not found. Running with system python."
fi

# Set PYTHONPATH to current directory
export PYTHONPATH=.

echo "🚀 Starting OpenQuant Paper Trading..."
echo "Mode: Paper Trading (Safe Mode)"
echo "-----------------------------------"

# Example: Run allocation script with default settings
# You can modify this to run your specific strategy or allocation logic
python3 scripts/paper_apply_allocation.py --allocation-file reports/allocation_latest.json --daily-loss-limit 0.05

echo "-----------------------------------"
echo "Done."
