# OpenQuant Roadmap

## Design Philosophy
**Pure Quantitative Trading**: This robot is based exclusively on mathematical models, statistics, and probability theory.
- NO retail/technical analysis indicators (SMA, EMA, RSI, MACD, Bollinger, etc.)
- Focus on: Kalman filtering, cointegration, GARCH, Hurst exponent, stat-arb, market microstructure
- All strategies must have mathematical/statistical foundation

## Current Status (2025-11-27)
- **99 tests passing**, 2 skipped (AppImage on Windows)
- Retail indicator code removed (ta_features.py, pandas_ta wrapper)
- All 7 previously failing tests fixed
- Risk management enhanced: Kill switch, circuit breakers, per-asset limits
- Audit trail database for regulatory compliance
- Periodic retrain scheduler for automated WFO

## Phase 1 — Core Engine (Completed)
- [x] Data loaders: yfinance (equities), ccxt (crypto), MT5 (forex/cfd)
- [x] Backtesting: vectorized, fees/slippage, position sizing
- [x] Evaluation: Sharpe/DSR, MaxDD, CVaR, WFO mean test Sharpe
- [x] Overfitting controls: walk-forward (WFO), purged CV, deflated Sharpe (DSR)
- [x] Results DB: DuckDB persistence with best_configs view
- [x] Reporting: Markdown report, Streamlit dashboard (auto-open) with filters
- [x] Optimization: Optuna (always on), seeded from DB; adaptive grid narrowing
- [x] Risk: Guardrails (max DD, CVaR, daily loss cap) and concentration limits
- [x] Concurrency: bounded parallel fetch and evaluation; rate limiting
- [x] Validation: data/signals/backtest validators (fail-safe)

## Phase 3 — Paper Trading (Current Focus)
- [x] PortfolioState and basic order simulator (rebalance by target weights)
- [x] JSON I/O for paper state (save/load)
- [x] Script: scripts/paper_apply_allocation.py to apply latest allocation to paper state
- [x] Persist portfolio ledger to DuckDB (positions, trades, equity) and dashboard view (basic)
- [x] Execution model: next-bar fills, partial fills (basic fees/slippage implemented)
- [x] MT5 bridge for paper trading (init/login, symbol select, market orders)
- [x] Implement stop-loss orders
- [x] Implement take-profit orders
- [x] MT5 mirroring (basic): apply allocation weights to MT5 net positions via Python bridge
- [x] MT5 FX research mode: direct OHLCV + universe discovery from MT5 terminal (EURUSD, GBPUSD, ...)
- [x] MT5 chart EA: auto-attach viewer EA to chart and visualize strategy signals (via CSV export)
- [x] Alerts on paper execution anomalies (slippage too high, rejected orders)

### Next Steps
- [x] Add daily loss limits
- [x] Consider adding a trading schedule/time window
- [x] Add email/notification alerts for significant events (WhatsApp via Webhook)
- [x] Implement proper logging of all actions
- [x] Add a kill switch for emergency stops (integrated into all execution paths)
- [x] Circuit breakers (daily loss, drawdown, volatility limits)
- [x] Per-asset risk limits (max notional, max % portfolio, max positions)
- [x] Audit trail database (DuckDB persistence of all trading decisions)
- [x] Periodic retrain scheduler (automated WFO retraining)

## Phase 2 — Advanced Strategies (Completed)
- [x] Quantitative Core:
  - [x] Stationarity: ADF, KPSS tests
  - [x] Filtering: Kalman Filter (dynamic beta), Hodrick-Prescott (trend separation)
  - [x] Cointegration: Engle-Granger test, Half-life estimation
  - [x] Volatility: GARCH(1,1), Garman-Klass
  - [x] Microstructure: VPIN, Order flow imbalance
- [x] Genetic Optimization:
  - [x] Evolution engine (`openquant.optimization`)
  - [x] CLI runner (`run_genetic_optimization.py`)
- [x] Pairs Trading / Stat-Arb strategies (`openquant.strategies.quant.stat_arb`)
- [ ] Strategy selection & ensembling under constraints (exposure, DD, CVaR)
- [ ] Regression/Classification models (SKLearn integration)

## Phase 3 — Paper Trading & Operations (Current Focus)
- [x] PortfolioState and basic order simulator (rebalance by target weights)
- [x] JSON I/O for paper state (save/load)
- [x] Script: scripts/paper_apply_allocation.py to apply latest allocation to paper state
- [x] Persist portfolio ledger to DuckDB (positions, trades, equity) and dashboard view (basic)
- [x] Execution model: next-bar fills, partial fills (basic fees/slippage implemented)
- [x] MT5 bridge for paper trading (init/login, symbol select, market orders)
- [x] Alpaca Broker integration (Linux-ready)
- [x] Implement stop-loss orders
- [x] Implement take-profit orders
- [x] MT5 mirroring (basic): apply allocation weights to MT5 net positions via Python bridge
- [x] MT5 FX research mode: direct OHLCV + universe discovery from MT5 terminal (EURUSD, GBPUSD, ...)
- [x] MT5 chart EA: auto-attach viewer EA to chart and visualize strategy signals (via CSV export)
- [x] Alerts on paper execution anomalies (slippage too high, rejected orders)

### Operational Robustness (Next Steps)
- [x] Add daily loss limits <!-- id: 5 -->
- [x] Consider adding a trading schedule/time window <!-- id: 6 -->
- [x] Add email/notification alerts for significant events (WhatsApp via Webhook)
- [x] Implement proper logging of all actions <!-- id: 7 -->
- [x] Add a kill switch for emergency stops (integrated into all execution paths) <!-- id: 8 -->
- [x] Circuit breakers (daily loss, drawdown, volatility limits) <!-- id: 9 -->
- [x] Per-asset risk limits (max notional, max % portfolio, max positions) <!-- id: 10 -->
- [x] Audit trail database (DuckDB persistence of all trading decisions) <!-- id: 11 -->
- [x] Periodic retrain scheduler (automated WFO retraining) <!-- id: 12 -->

## Phase 4 — Live Execution (Guarded)
- [ ] Kill switches, circuit breakers, per-asset risk limits
- [ ] Audit logs, persistence, periodic retrain loop (WFO)
- [ ] Real-money broker adapters (Alpaca for Linux, MT5 for Windows)

## Phase 5 — Advanced Research (Next Frontier)
- [x] Strategy Selection & Ensembling (Meta-Strategy)
- [x] Regression/Classification models (SKLearn integration)
- [ ] RL baseline for a single asset (SB3)
- [ ] Reality Check/SPA, more robust CVaR estimation
