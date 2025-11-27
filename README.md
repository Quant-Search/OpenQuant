# OpenQuant

## Abstract

OpenQuant is an open-source quantitative trading laboratory designed for rigorous financial research, backtesting, and execution. It prioritizes mathematical correctness, reproducibility, and modular architecture. The system integrates advanced time-series analysis, statistical arbitrage models, and genetic optimization algorithms to identify and exploit market inefficiencies.

## Key Features

*   **Multi-Broker Support**:
    *   **Paper Trading**: Robust simulation with order management, slippage/fee modeling, and portfolio tracking.
    *   **Alpaca**: Native integration for Linux/Cloud environments (Live & Paper).
    *   **MetaTrader 5**: Full broker integration for Forex/CFD trading on Windows via `MT5Broker` class.
*   **Quantitative Core**:
    *   **Stationarity**: ADF, KPSS tests.
    *   **Regime Classification**: Hurst Exponent, Trend/Mean-Reversion scoring.
    *   **Filtering**: Kalman Filters, Hodrick-Prescott.
    *   **Volatility**: GARCH, Garman-Klass.
*   **Advanced Strategies**:
    *   **Ensembling**: Combine multiple strategies via Voting or Weighted Average (`StrategyMixer`).
    *   **Genetic Optimization**: Evolve strategy parameters using evolutionary algorithms.
*   **Operational Robustness**:
    *   **Risk Management**: Stop-Loss, Take-Profit, Daily Loss Limits.
    *   **Scheduling**: Trading windows and day restrictions.
    *   **Alerting**: WhatsApp/Webhook notifications for critical events.

## Architecture

*   **`openquant.quant`**: Mathematical primitives and statistical tests.
*   **`openquant.research`**: Hypothesis testing engine for large universes.
*   **`openquant.strategies`**: Strategy implementations and the `StrategyMixer` ensemble engine.
*   **`openquant.paper`**: Paper trading simulator and state management.
*   **`openquant.broker`**: Broker adapters (Alpaca, MT5).

## Installation

### Prerequisites
*   Python 3.10+
*   Linux (Recommended) or Windows

### Setup

1.  **Environment Initialization**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Configuration**:
    Copy the example configuration and set your credentials (Alpaca/MT5).
    ```bash
    cp .env.example .env
    ```

## Usage

### Paper Trading (Linux/Safe Mode)
Run the paper trading simulation with the latest allocation:
```bash
./run_paper.sh
```

### Research & Backtesting
Run a research cycle on a specific asset:
```bash
python3 scripts/run_robot_cli.py --symbols BTC/USD --strategy stat_arb
```

### Strategy Ensembling
Combine strategies using the `StrategyMixer` in your research config:
```python
# Example config snippet
strategy="mixer",
params={
    "sub_strategies": ["kalman", "hurst"],
    "weights": [0.6, 0.4]
}
```

### Notifications
Test WhatsApp notifications:
```bash
python3 scripts/test_whatsapp.py --url "YOUR_WEBHOOK_URL"
```

## Deployment
For cloud deployment, refer to `CLOUD_DEPLOY.md`.

## Security & Policies
Refer to `SECURITY.md` and `POLICIES.md`.
