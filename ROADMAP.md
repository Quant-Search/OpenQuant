# OpenQuant Roadmap

## Phase 1 — Offline Research Engine (MVP)
- Data loaders: yfinance (equities), ccxt (crypto)
- Feature engineering: core TA features (SMA/EMA/RSI/MACD/Bollinger)
- Strategies: rule-based (SMA/EMA/RSI/MACD/Bollinger; momentum & mean reversion)
- Backtesting: vectorized, fees/slippage, position sizing
- Evaluation: Sharpe/DSR, MaxDD, CVaR, WFO mean test Sharpe
- Overfitting controls: walk-forward (WFO), purged CV (initial), deflated Sharpe (DSR)
- Results DB: DuckDB persistence with best_configs view
- Reporting: Markdown report, Streamlit dashboard (auto-open) with filters
- Optimization: Optuna (always on), seeded from DB; adaptive grid narrowing & trial budget; regime-aware tuning (trend vs mean-reversion)
- Risk: Guardrails (max DD, CVaR, daily loss cap) and concentration limits (per symbol and per (symbol,strategy))
- Concurrency: bounded parallel fetch and evaluation; rate limiting
- Validation: data/signals/backtest validators (fail-safe)
- Tests: unit + integration (no mocks for core paths)


### Status (current)
- Completed: Always-on optimization + WFO, DB persistence, dashboard auto-launch, validators, rate limiter, RSI/MACD/Bollinger, seeding + adaptive grid/trials.
- Phase 3 (Paper trading) — in progress:
  - [x] PortfolioState and basic order simulator (rebalance by target weights)
  - [x] JSON I/O for paper state (save/load)
  - [x] Script: scripts/paper_apply_allocation.py to apply latest allocation to paper state
  - [x] Persist portfolio ledger to DuckDB (positions, trades, equity) and dashboard view (basic)
  - [ ] Execution model: next-bar fills, partial fills (basic fees/slippage implemented)
  - [x] MT5 bridge for paper trading (init/login, symbol select, market orders)
  - [x] MT5 mirroring (basic): apply allocation weights to MT5 net positions via Python bridge
  - [x] MT5 FX research mode: direct OHLCV + universe discovery from MT5 terminal (EURUSD, GBPUSD, ...)
  - [ ] MT5 chart EA: auto-attach viewer EA to chart and visualize strategy signals

  - [ ] Alerts on paper execution anomalies (slippage too high, rejected orders)

- In progress: Scaling research coverage and stability heuristics.
- Completed: Scheduler + guardrails (kill switches, daily loss cap, max DD) and concentration limits; regime tagging baseline.
- Completed: Alerts/notifications (run summaries + optional webhook), exposure caps helper and allocation snapshot per run.
- Next: MT5 paper trading, exposure/portfolio DB table and dashboard view.

## Phase 2 — Broader Strategy Space & Selection
- Add pairs/stat-arb, regression/classification models (SKLearn)
- Scheduler/Guardrails: repeat runs (--repeat-mins), risk kill-switches (max DD, daily loss cap, CVaR)

- Hyperparameter search (Optuna)
- Strategy selection & ensembling under constraints (exposure, DD, CVaR)

## Phase 3 — Paper Trading
- Broker/exchange connectors (Alpaca, Binance via CCXT)
- Paper trading router, stateful portfolio, and risk limits
- Monitoring dashboards & alerts

## Phase 4 — Live Execution (Guarded)
- Kill switches, circuit breakers, per-asset risk limits
- Audit logs, persistence, periodic retrain loop (WFO)

## Phase 5 — Advanced Methods
- RL baseline for a single asset (SB3)
- Reality Check/SPA, more robust CVaR estimation
- MetaTrader 5 bridge (if MT5 terminal available)

