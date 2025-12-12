<<<<<<< HEAD
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
=======
1→# AGENTS.md
2→
3→## Setup
4→```bash
5→python3 -m venv .venv
6→source .venv/bin/activate  # Linux/Mac; Windows: .venv\Scripts\activate
7→pip install -r requirements.txt
8→cp .env.example .env  # Configure credentials
9→```
10→
11→## Commands
12→- **Test**: `pytest tests/`
13→- **Dev Server**: `python3 scripts/run_dashboard.py` (Streamlit on port 8501)
14→- **Paper Trading**: `./run_paper.sh` (Linux) or `run_robot.bat` (Windows)
15→- **Research**: `python3 scripts/run_robot_cli.py --symbols BTC/USD --strategy stat_arb`
16→
17→## Tech Stack
18→- **Python 3.10+** with pandas, numpy, statsmodels, scipy
19→- **Brokers**: Paper trading, Alpaca (Linux), MetaTrader 5 (Windows)
20→- **UI**: Streamlit dashboard
21→- **Testing**: pytest
22→
23→## Architecture
24→- `openquant/quant/`: Statistical primitives (ADF, KPSS, Hurst, Kalman, GARCH)
25→- `openquant/strategies/`: Strategy implementations and mixer ensemble
26→- `openquant/broker/`: Broker adapters (Alpaca, MT5, abstract interface)
27→- `openquant/paper/`: Paper trading simulator
28→- `openquant/research/`: Research engine
29→- `openquant/risk/`: Risk management (stop-loss, guardrails, circuit breakers)
30→
31→## Code Style
32→- Type hints required (e.g., `def func(x: float) -> Dict[str, Any]`)
33→- Docstrings for public APIs
34→- ABC pattern for interfaces
35→- Signal convention: `-1` short, `0` flat, `+1` long
36→
>>>>>>> 9bae6a0 (Optimize hot paths in backtest engine using NumPy vectorization, Numba JIT compilation, and Dask parallelization)
