# OpenQuant

## Abstract

OpenQuant is an open-source quantitative trading laboratory designed for rigorous financial research, backtesting, and execution. It prioritizes mathematical correctness, reproducibility, and modular architecture over retail trading heuristics. The system integrates advanced time-series analysis, statistical arbitrage models, and genetic optimization algorithms to identify and exploit market inefficiencies.

## Architecture

The system is composed of several distinct modules, each responsible for a specific domain of the quantitative pipeline:

*   **`openquant.quant`**: The mathematical core. Implements statistical tests (ADF, KPSS), filters (Kalman, Hodrick-Prescott), and volatility estimators (GARCH, Garman-Klass).
*   **`openquant.research`**: Orchestrates large-scale hypothesis testing across multiple assets and timeframes.
*   **`openquant.backtest`**: A vectorized, event-driven backtesting engine designed for high-performance simulation of trading strategies.
*   **`openquant.optimization`**: A genetic algorithm framework for evolving strategy parameters and logic based on objective fitness functions (e.g., Sharpe Ratio, Sortino Ratio).
*   **`openquant.gui`**: A Streamlit-based control center for real-time monitoring, visualization, and manual intervention.

## Mathematical Specifications

The system implements the following quantitative models:

1.  **Stationarity Analysis**: Augmented Dickey-Fuller (ADF) and KPSS tests to determine the integration order of time series.
2.  **Regime Classification**: Hurst Exponent estimation to classify market regimes as persistent (trending), anti-persistent (mean-reverting), or geometric Brownian motion.
3.  **Dynamic Filtering**: Kalman Filters for recursive estimation of hedge ratios in pairs trading and dynamic beta calculation.
4.  **Trend Separation**: Hodrick-Prescott filter for decomposing time series into trend and cyclical components.
5.  **Volatility Modeling**: GARCH(1,1) and Garman-Klass estimators for precise volatility forecasting and risk normalization.
6.  **Market Microstructure**: VPIN (Volume-Synchronized Probability of Informed Trading) and order flow imbalance metrics.

## Installation and Configuration

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
    Copy the example configuration file and populate it with necessary credentials (Alpaca, MetaTrader 5).
    ```bash
    cp .env.example .env
    ```

## Usage

### Research and Backtesting

To execute a research cycle on a specific asset using the command-line interface:

```bash
python3 scripts/run_robot_cli.py --symbols BTC/USD --strategy stat_arb
```

### Genetic Optimization

To initiate the evolutionary optimization process for strategy parameters:

```bash
python3 scripts/run_genetic_optimization.py --symbol EURUSD --pop-size 50 --generations 20
```

### Dashboard

To launch the graphical user interface for monitoring and analysis:

```bash
python3 scripts/run_dashboard.py
```

## Deployment

For deployment on cloud infrastructure (e.g., Oracle Cloud, AWS), refer to `CLOUD_DEPLOY.md`. The system is optimized for headless execution in Linux environments.

## Security and Policies

Refer to `SECURITY.md` for vulnerability reporting and `POLICIES.md` for contribution guidelines. This software is provided for educational and research purposes. Users are responsible for all financial risks associated with its use.
