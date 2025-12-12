# AGENTS.md

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac; Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Configure credentials
```

## Commands
- **Test**: `pytest tests/`
- **Dev Server**: `python3 scripts/run_dashboard.py` (Streamlit on port 8501)
- **Paper Trading**: `./run_paper.sh` (Linux) or `run_robot.bat` (Windows)
- **Research**: `python3 scripts/run_robot_cli.py --symbols BTC/USD --strategy stat_arb`
- **Backup**: `python3 scripts/backup_databases.py backup` (Create database backup)
- **Restore**: `python3 scripts/backup_databases.py restore <backup_name>` (Restore from backup)

## Tech Stack
- **Python 3.10+** with pandas, numpy, statsmodels, scipy
- **Brokers**: Paper trading, Alpaca (Linux), MetaTrader 5 (Windows)
- **UI**: Streamlit dashboard
- **Testing**: pytest

## Architecture
- `openquant/quant/`: Statistical primitives (ADF, KPSS, Hurst, Kalman, GARCH)
- `openquant/strategies/`: Strategy implementations and mixer ensemble
- `openquant/broker/`: Broker adapters (Alpaca, MT5, abstract interface)
- `openquant/paper/`: Paper trading simulator
- `openquant/research/`: Research engine
- `openquant/risk/`: Risk management (stop-loss, guardrails, circuit breakers)
- `openquant/storage/`: Database storage and backup (results, portfolio, audit, TCA)

## Code Style
- Type hints required (e.g., `def func(x: float) -> Dict[str, Any]`)
- Docstrings for public APIs
- ABC pattern for interfaces
- Signal convention: `-1` short, `0` flat, `+1` long
