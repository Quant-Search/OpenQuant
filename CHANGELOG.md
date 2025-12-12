## [Unreleased]

### Execution Quality Monitoring (NEW)
- **ExecutionQualityMonitor**: Comprehensive order execution tracking in `openquant/analysis/execution_quality.py`
  - Fill rate monitoring (percentage of orders successfully filled)
  - Rejection rate tracking (orders rejected or cancelled)
  - Slippage distribution analysis with percentiles and histograms
  - Benchmark comparison against historical TCA data
  - Automated alerts for execution quality degradation
  - Database persistence of metrics snapshots and alerts
- **Key Metrics**: Fill rate, rejection rate, partial fill rate, average/median slippage, slippage volatility, fill time, trading fees
- **Alert Types**: fill_rate_degradation, rejection_rate_high, slippage_high, slippage_volatility_high
- **Configurable Thresholds**: Customize for high/medium/low frequency trading styles
- **Scripts**: 
  - `scripts/monitor_execution_quality.py` - Standalone monitoring tool
  - `scripts/execution_quality_integration_example.py` - Integration examples
- **Documentation**: See `EXECUTION_QUALITY.md` for detailed usage guide

### Strategy Comparison Report (NEW)
- **Comprehensive Backtesting Comparison**: New `openquant/reporting/strategy_comparison.py` module
  - Side-by-side metrics tables (Sharpe, Sortino, Max DD, Win Rate, Profit Factor, CVaR, Calmar, etc.)
  - Equity curve overlays for visual performance comparison
  - Drawdown curve overlays to analyze risk patterns
  - Statistical tests: Paired t-test and Diebold-Mariano test for strategy selection
  - Return correlation matrix for diversification analysis
  - Automated strategy ranking with composite scoring (customizable weights)
  - Export capabilities: CSV files, text reports, matplotlib/seaborn visualizations
- **Example Script**: `scripts/strategy_comparison_example.py` demonstrates full workflow
- **Comprehensive Tests**: `tests/test_strategy_comparison.py` with 20+ test cases
- **Documentation**: `openquant/reporting/README_STRATEGY_COMPARISON.md` with usage guide

### Single GUI Dashboard (NEW)
- **Consolidated Control Center**: All robot controls in one Streamlit dashboard
- **Risk Monitor Page**: Real-time view of kill switch, circuit breaker, market hours status
- **Emergency Controls**: One-click kill switch activation/deactivation
- **Audit Trail Viewer**: Recent trading events displayed in dashboard
- **Status Indicators**: Sidebar shows robot state (Running/Stopped/Risk Alert)

### Execution Improvements (NEW)
- **Volatility-Adjusted Position Sizing**: Inverse volatility weighting in `openquant/risk/exposure.py`
  - Uses max drawdown as risk proxy when returns unavailable
  - Clamps volatility factors to 0.5x-2.0x of base weight
- **Market Hours Check**: New `openquant/risk/market_hours.py` module
  - Forex: Sun 17:00 - Fri 17:00 EST (24/5)
  - Crypto: 24/7
  - US Stocks: 9:30-16:00 EST (Mon-Fri)
  - Extended Hours: 4:00-20:00 EST
- **Connection Retry Logic**: Enhanced `openquant/utils/retry.py`
  - Exponential backoff with jitter
  - Pre-configured handlers: MT5_RETRY, ALPACA_RETRY, CCXT_RETRY
- **Live Circuit Breaker Updates**: Circuit breaker updated after every execution cycle

### Risk Management Enhancements
- **Kill Switch Integration**: Added kill switch checks to all execution paths (simulator, MT5 bridge, Alpaca broker)
- **Circuit Breakers**: New `openquant/risk/circuit_breaker.py` module with:
  - Daily loss limit (default 2%)
  - Drawdown limit (default 10%)
  - Volatility limit (default 5%)
  - State persistence to JSON
  - Automatic daily reset
- **Per-Asset Risk Limits**: New `openquant/risk/asset_limits.py` module with:
  - Max notional per asset
  - Max percentage of portfolio per asset
  - Max positions per asset
  - Max leverage per asset
  - JSON configuration support
- **Audit Trail Database**: New `openquant/storage/audit_trail.py` module with:
  - Persistent DuckDB logging of all trading decisions
  - Event types: SIGNAL, ORDER_DECISION, ORDER_EXECUTION, ORDER_REJECTED, KILL_SWITCH, CIRCUIT_BREAKER, LIMIT_VIOLATION, SYSTEM_START, SYSTEM_STOP, ERROR, WARNING
  - Query API with filters by event type, symbol, strategy, time range
- **Periodic Retrain Scheduler**: New `openquant/research/retrain_scheduler.py` module with:
  - Configurable frequency (daily, weekly, biweekly, monthly)
  - State persistence for tracking last retrain time
  - Daemon mode for background operation
  - Integration with universe_runner for full WFO pipeline

### Cleanup: Pure Quantitative Focus
- **Removed retail/technical analysis code**: Deleted SMA, EMA, RSI, MACD, Bollinger indicators and strategies
- **Deleted files**: `openquant/features/ta_features.py`, `openquant/strategies/wrappers/pandas_ta.py`, `openquant/research/runner.py`
- **Deleted tests**: `tests/test_ema_strategy.py`, `tests/test_sma_strategy.py`, `tests/test_rsi_macd_bollinger.py`, `tests/test_research_offline.py`
- **Updated registry**: Removed pandas_ta wrapper imports from `openquant/strategies/registry.py`
- **Philosophy**: Robot now focuses purely on mathematical models, statistics, probability (Kalman, cointegration, GARCH, Hurst, stat-arb)

### Bug Fixes
- **test_tca**: Fixed module-level duckdb mock in `tests/test_schedule.py` that was polluting subsequent tests
- **test_tca**: Made row unpacking more robust in `openquant/analysis/tca.py` (explicit indexing instead of tuple unpacking)
- **test_allocation_respects_caps**: Fixed correlation check in `openquant/risk/exposure.py` to exclude same symbol
- **test_optuna_best_params**: Fixed categorical sampling for small grids in `openquant/optimization/optuna_search.py`
- **test_results_db_best_view**: Enabled `_refresh_best_view()` call in `openquant/storage/results_db.py`
- **test_view_charting**: Fixed ccxt mock in `tests/test_dashboard_components.py` using `sys.modules` patch
- **test_appimage_execution**: Added platform check for Windows in `tests/test_appimage_run.py`

### Test Results
- 99 tests passing, 2 skipped (AppImage tests on Windows)
- All 7 previously failing tests now fixed
- 7 new tests for new modules (audit trail, retrain scheduler, circuit breaker)

---

- MT5 alerts: Added execution anomaly detection (rejected orders, high slippage) to mt5_bridge with alerting via openquant.utils.alerts
- Risk Management: Added SL/TP support to MT5 bridge (apply_allocation_to_mt5 now accepts sl/tp in allocation entries)
- Risk Management: Added modify_position() to mt5_bridge for updating SL/TP on existing positions
- Trailing Stop: Implemented TrailingStopManager in openquant/risk/trailing_stop.py with configurable trailing distance/activation
- Runner: Added scripts/run_trailing_stop.py for continuous monitoring and SL updates
- Tests: Added tests/test_mt5_bridge_alerts.py and tests/test_mt5_bridge_risk.py and tests/test_trailing_stop.py (13 tests total, all passing)

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

- **Operational Safety**: Daily Loss Limits and Trading Schedule (Time/Day restrictions).
- **Notifications**: WhatsApp notification support via Webhook.
- **Scripts**: `run_paper.sh` for easy Linux execution.
- **ML Strategy**: Scikit-learn integration with Walk-Forward Optimization and statistical feature engineering.
- **Structured Logging**: JSON-formatted logs with daily rotation, sensitive data redaction, and `TradeLogger` context manager.

### Changedds --fee-bps/--slippage-bps and --portfolio-db
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

