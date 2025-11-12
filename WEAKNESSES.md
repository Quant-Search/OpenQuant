Data fetching and timezone/market calendar assumptions

Data loader returns OHLCV with assumed UTC indexing, but yfinance can return timezone-aware or naive indexes; this can introduce mismatches (e.g., shifting signals vs returns).
No explicit harmonization for pre/post-market or non-trading days; using pct_change on daily closes can be fine, but intraday and exchanges with different trading hours need careful alignment.

Execution / realistic slippage & fees


Fees are modeled as basis points per position change and subtracted from returns; there is no slippage model for market impact or bid/ask spread.
The model assumes execution at close / next bar without modeling fill probabilities or partial fills — may overstate performance.

Position sizing, portfolio-level constraints, and leverage

Current backtest supports single-asset weight and basic leverage cap. For multi-asset research you’ll need portfolio construction, rebalancing logic, and correlation-aware sizing.
No cash management (e.g., borrowing costs for leverage, margin/interest).


Risk measures & statistical robustness

Sharpe and CVaR are computed plainly; small-sample issues or return skewness can mislead ranking. Sharpe uses sample std and floors sd to eps which can inflate Sharpe when sd is nearly zero.
CVaR uses losses = -returns and then quantile; it's OK for a quick check but you should verify direction and interpretation across datasets.

Edge cases in backtest implementation

The backtester uses pos = sig.shift(1) to simulate entering next bar; but when signals lead/lag or when there are missing timestamps, that can misalign. Also fee calculation uses pos_change * w * fee_bps — fees scale with weight but not with notional price or slippage.
Trades counting: trades = pos_change — fine for simple counting, but masks partial rebalances and multiple entry/exit events.

I didn’t find a CI setup or test-runner in the repository root. Tests exist in tests but some test runner invocation failed to find tests when targeted. CI to run pytest on pushes would catch regressions.
Lacking type hints in many modules (there are some) — adding typing and static checks (mypy) would prevent API mismatches.

Reproducibility & environment

No lockfile or pinned dependency versions (only README install instructions and the requirements.txt I added). This can lead to environment drift and reproducibility problems.

Security & secrets

.env usage is suggested — good — but scripts that talk to brokers (MT5, CCXT) must have strict guardrails to avoid accidental live execution using research configs.


Add CI with tests (GitHub Actions)
Run pytest, flake8/ruff, and mypy on push and PRs. This prevents regressions.
Pin dependencies / add lockfile
Add requirements.txt (done) and consider pip-tools/poetry/pipenv to pin versions and produce reproducible installs.
Add a CONTRIBUTING.md and a run_research.py usage note in README (or example command line) — README already has Quickstart, but add exact commands for Windows cmd and PowerShell.
Improve test coverage for small units:
Backtest engine edge cases (no price changes, constant price, NaNs).
Data loader fallback behavior when yfinance/ccxt not installed.
Strategy validation (fast<slow, negative inputs).
Add a simple integration smoke test that runs the example config end-to-end in CI but with a short timeframe or cached small dataset to avoid long network calls.

Near-term (medium effort)
6. More realistic cost and execution model

Add optional spread and slippage models, e.g., spread per trade, slippage as pct of price, and ability to simulate limit vs market fills.
Portfolio & sizing improvements
Add multi-asset portfolio module (weights, rebalancing schedule), cash accounting, and margin/interest costs.
Improve risk calculations
Add bootstrapped confidence intervals for Sharpe/CVaR; add annualization options per asset class.
Timezone and calendar handling
Harmonize timezones across data sources and ensure index alignment when resampling/merging data.

Longer-term (higher effort / high value)
10. Add a streaming/paper-trading simulation harness (with strict dry-run switch)
- Make live adapters explicit and require an "enable-live" flag for any code that touches brokers, with safety checks (e.g., no real keys in config).
11. Performance: vectorize and profile hotspots
- If you plan to run large universe research, consider incremental improvements (numba, cython, or optimized numpy) for metrics/backtest inner loops.
12. Add stronger governance around model selection
- Avoid selecting the best strategy purely by in-sample Sharpe. Add cross-validation, time-based walk-forward, or out-of-sample testing.

Edge cases to watch

Empty data returns (runner already raises). Unit tests should verify loader behavior.
Single-sample returns: Sharpe can be unstable — guard your metrics.
Mixed timeframes or missing bars — ensure reindexing/resampling logic is explicit.
Floating rounding or off-by-one in fee application: fees are per change in position but your interpretation of weight vs notional matters when adding leverage.

Execution Realism
Add bid/ask spread modeling
Implement realistic slippage
Add volume-based liquidity checks
More granular fee structure (maker/taker)
Risk Management
Add position exit rules (stop-loss, trailing stops)
Implement portfolio-level risk limits
Add correlation-aware position sizing
Real-time drawdown monitoring
Operational Robustness
Add proper error handling for data gaps
Implement heartbeat monitoring
Add connection failure recovery
Build logging/alerting system

The strategy themselves have to be very and strictky and have to had the best standards possibles and mathematical computations or executions realistically possible.

MT5 Bridge Safety:
Lazy imports to avoid MT5 dependency
Credential validation
Position netting and volume limits
Market order safety checks
Demo account defaults
Paper Trading Safety:
Separate paper state from live trading
Simulated fees and slippage
Transaction logging in DuckDB
Exposure and risk limits

Add position size limits in mt5_bridge.py
Implement stop-loss orders
Add daily loss limits
Consider adding a trading schedule/time window
Add email/notification alerts for significant events
Implement proper logging of all actions
Add a kill switch for emergency stops