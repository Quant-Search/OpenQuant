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