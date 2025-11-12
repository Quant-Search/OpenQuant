## [Unreleased]
- MT5 FX mode: universe runner now supports `exchange="mt5"` with direct OHLCV from MetaTrader5 and FX symbol discovery (majors, metals)
- Data source: added openquant/data/mt5_source.py with `fetch_ohlcv()` and `discover_fx_symbols()`; cached under source key `mt5`
- Dashboard: added "Use MT5 FX mode" toggle and MT5 settings now default from env (OQ_MT5_TERMINAL, OQ_MT5_SERVER, OQ_MT5_LOGIN, OQ_MT5_PASSWORD)
- Fix: mt5_bridge.init now passes `path=terminal_path` to `mt5.initialize()` so a non-running terminal can be launched
- E2E: one-click GUI button runs research → allocation → paper ledger → MT5 apply for supported FX symbols
- Security: removed hardcoded MT5 credentials fallback from `scripts/mt5_run_once.py`; now strictly requires env vars; added SECURITY.md and .env.example

- Guardrails: Added apply_guardrails (max DD, CVaR, daily loss cap) and integrated into run_universe with optional thresholds
- Concentration limits: Post-processing cap to demote excess OK rows per symbol and per (symbol,strategy); CLI flags --max-per-symbol and --max-per-strategy-per-symbol
- Regime tagging: Added compute_regime_features (trend score, volatility) and used to tune search band and trials per strategy family
- Scheduler: scripts/run_universe_research.py supports --repeat-mins and guardrail flags; dashboard launches once per session

- Defaults: Concentration limits now enabled by default (max_per_symbol=20, max_per_strategy_per_symbol=5) in runner and CLI
- Alerts: Added run summaries and optional webhook alerts (env OPENQUANT_ALERT_WEBHOOK or --alert-webhook)
- Exposure: Added greedy portfolio allocation helper with caps; allocation snapshot is saved per run (allocation_YYYYMMDD_HHMMSS.json)
- Fix: Guardrails check order in runner (compute DD/CVaR before applying guardrails)


- Paper trading: Added PortfolioState and simple order simulator; tests added; diagrams/paper_trading_flow.mmd

- Portfolio ledger: DuckDB tables (portfolio_trades, portfolio_positions, portfolio_equity) and dashboard section
- Execution realism: simulator supports fee_bps and slippage_bps; paper_apply_allocation CLI adds --fee-bps/--slippage-bps and --portfolio-db
- MT5: optional bridge stub (no dependency required yet); safe availability checks
- Dashboard: Added Robot Control section to run research and apply allocation to Paper + MT5 from the GUI



# Changelog

## Unreleased
- Fix: runner stored max drawdown with wrong sign in metrics; now positive fraction
- Fix: reporting/markdown.py code fence typo causing runtime error when writing reports
- Fix: scripts now ensure repository root on sys.path for reliable `openquant` imports
- Test: add tests/test_cvar.py for CVaR tail-loss behavior
- Test: add tests/test_research_offline.py for end-to-end offline research run
- Tooling: add scripts/run_research_offline.py to generate a report without network (synthetic data)

- Fix: Sharpe ratio division-by-zero handling — floor stdev to epsilon instead of returning 0.0
- Fix: pandas deprecations — use `pct_change(fill_method=None)`; replace frequency strings `'H' -> 'h'`
- Fix: yfinance 4h resample — use lowercase `'4h'` consistently
- Fix: research runner passes `start`/`end` through to DataLoader
- Fix: trade counting — use `.to_numpy().sum()` to avoid pandas deprecation
- Verify: full test suite passing locally (7 passed)
- Enhancement: Runner continues reporting when no strategies satisfy constraints; report includes an `ok` column per row
- Chore: Add ignore patterns for local run logs (`out_*.txt`, `pytest_*.txt`) and remove transient log files

- Feature: Position sizing (weight) added to backtest engine; fees and returns scaled by allocated weight
- Config: Added backtest.initial_capital, position_weight, max_leverage, max_weight (single-asset caps enforced in runner)
- Test: Added position sizing tests to tests/test_backtest_engine.py

- Feature: Binance universe discovery (USDT pairs) ranked by 24h volume via ccxt; new script `scripts/run_universe_research.py`
- Feature: Multi-asset/timeframe research runner over discovered symbols and [1d,4h,1h]
- Feature: Data caching to Parquet with CSV fallback (no extra deps required); cache under `data_cache/`
- Reporting: Added universe summary report writer with ranking across symbol/timeframe/params
- Infra: `.gitignore` updated to ignore `data_cache/`
- Tests: Switched to real network integration script for Binance universe (background run)

- Eval: Added evaluation modules (deflated Sharpe with SciPy-free approximation, WFO minimal, Purged K-Fold splits)
- Strategies: Added EMA crossover strategy and strategy registry; runner now supports multiple strategies per run
- Reporting: Summary now includes strategy name and Deflated Sharpe Ratio (dsr)
- Perf: Parallelized per-symbol strategy grid via ThreadPool (configurable max_workers); network fetch remains serialized to respect rate limits
- Tests: Added unit tests for EMA feature/strategy
- Chore: Removed mocked tests (ccxt dummy and loader) per real-network policy
- Dep: Installed pyarrow to enable efficient Parquet caching

- Features: Added RSI, MACD, Bollinger indicators and rule-based strategies; registry updated
- Runner: per-strategy parameter grids; fetch parallelism across symbols/timeframes (bounded ThreadPool)
- Rate limiting: process-wide token-bucket for ccxt requests
- Persistence: DuckDB results store (data/results.duckdb) with upsert from runner
- Validation: data/signals/backtest validators; metrics.ok flag per row
- GUI: Streamlit dashboard (openquant/gui/dashboard.py) to browse/filter results
- Deps: Installed duckdb and optuna (Bayesian optimization to be integrated next)
- Diagrams: diagrams/validation_pipeline.mmd
- Optimization: Optuna Bayesian search integrated (run_universe optimize=True, optuna_trials=N)
- Evaluators: WFO integrated; wfo_mts persisted in results (added column)
- Default behavior: Optimization (Optuna) and WFO now enabled by default in run_universe
- Results DB: best_configs view auto-refreshed on insert (latest + highest WFO/DSR/Sharpe per key)
- Results DB API: get_best_params() and get_best_config() helpers for seeding optimization
- Optimization: Seeding from best_configs with adaptive grid-narrowing (±10–30% by stability score) and adaptive Optuna trial budget (8–40 trials)
- Script: scripts/run_universe_research.py auto-launches Streamlit dashboard after run
- Risk: Concentration limits available post-run (per symbol, per (symbol,strategy))





## 0.0.1 — Initial scaffolding
- Repo structure, policies, security, roadmap
- Minimal research pipeline: features, strategy, backtest, metrics
- Basic reporting and tests with synthetic data

