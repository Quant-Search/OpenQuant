# OpenQuant

An open-source lab for quantitative trading: research, backtesting, and eventually paper/live execution (with strict risk controls). Modular, transparent, reproducible.

## Quickstart (Windows / PowerShell)

1) Create virtual env and install deps

```
python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -U pip; .\.venv\Scripts\python.exe -m pip install -U pandas numpy yfinance ccxt pyyaml python-dotenv pytest
```

2) Configure environment (optional, for live/paper later)

- Copy `.env.example` to `.env` and fill keys (kept local, gitignored)

3) Run tests

```
.\.venv\Scripts\python.exe -m pytest -q
```

4) Run a minimal research example

```
.\.venv\Scripts\python.exe scripts/run_research.py configs/aapl_sma.yaml
```

Outputs a Markdown report in `reports/` and (optionally) an equity CSV for the top result.

## Structure

- `openquant/data`: data loaders (yfinance, ccxt)
- `openquant/features`: feature engineering (TA)
- `openquant/strategies`: strategy interfaces and implementations
- `openquant/backtest`: vectorized backtester + metrics
- `openquant/research`: orchestration for research runs
- `openquant/reporting`: reporting utilities (Markdown)
- `configs/`: YAML experiment configs
- `tests/`: unit tests
- `diagrams/`: architecture diagrams (Mermaid)

## Security & Policies

- See SECURITY.md and POLICIES.md. Never commit secrets; use `.env`.
- This repository is for research/education. Use at your own risk; markets carry risk.
