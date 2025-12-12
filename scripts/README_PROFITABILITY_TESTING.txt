================================================================================
PROFITABILITY TESTING FRAMEWORK
================================================================================

OVERVIEW
--------
Comprehensive end-to-end profitability testing framework that validates trading
strategies before production deployment using walk-forward optimization, 
multiple risk-adjusted metrics, and Monte Carlo robustness testing.

KEY FILES
---------
1. test_profitability.py              - Main testing framework (27KB)
2. example_profitability_test.py      - Usage examples (7KB)
3. check_profitability_test_env.py    - Environment checker (8KB)
4. README_PROFITABILITY_TEST.md       - Full documentation
5. PROFITABILITY_TESTING_QUICKSTART.md - Quick start guide

FEATURES
--------
✓ Walk-Forward Optimization (3+ years of data)
✓ Risk-Adjusted Metrics (Sharpe, Sortino, Calmar, Omega)
✓ Monte Carlo Bootstrap (500+ runs)
✓ Target Validation (>50% return, <25% drawdown)
✓ Confidence Scoring (0-100 scale)
✓ Go/No-Go Recommendations

QUICK START
-----------
1. Check environment:
   python scripts/check_profitability_test_env.py

2. Run basic test:
   python scripts/test_profitability.py --strategy stat_arb --symbol BTC/USDT

3. Review results:
   Check console output and reports/profitability_*.json

USAGE EXAMPLES
--------------
# Basic test
python scripts/test_profitability.py --strategy stat_arb --symbol BTC/USDT

# Custom targets
python scripts/test_profitability.py \
  --strategy kalman \
  --symbol ETH/USDT \
  --return-target 0.60 \
  --max-drawdown 0.20

# High confidence test
python scripts/test_profitability.py \
  --strategy hurst \
  --symbol BTC/USDT \
  --monte-carlo-runs 1000 \
  --min-years 3.5

# Run examples
python scripts/example_profitability_test.py --example 1

METRICS CALCULATED
------------------
• Total Return & Annualized Return
• Sharpe Ratio (risk-adjusted return)
• Sortino Ratio (downside risk-adjusted)
• Calmar Ratio (return / max drawdown)
• Omega Ratio (probability-weighted gains/losses)
• Maximum Drawdown
• Win Rate
• Profit Factor
• Number of Trades
• Average Trade Return

CONFIDENCE SCORING
------------------
Score = Sharpe(30) + MCStability(20) + Targets(30) + WinRate(10) + PF(10)

70-100: GO - Strong recommendation
60-69:  GO - Moderate recommendation  
50-59:  CONDITIONAL GO - Paper trade first
40-49:  NO GO - Needs optimization
0-39:   NO GO - Fundamental issues

RECOMMENDATIONS
---------------
Based on confidence score and target achievement:
• GO - Deploy to production
• CONDITIONAL GO - Extended paper trading
• NO GO - Requires optimization/redesign

COMMAND-LINE OPTIONS
--------------------
--strategy            Strategy name (default: stat_arb)
--symbol              Trading symbol (default: BTC/USDT)
--source              Data source (default: ccxt:binance)
--timeframe           Timeframe (default: 1h)
--return-target       Min return % (default: 0.50 = 50%)
--max-drawdown        Max drawdown % (default: 0.25 = 25%)
--monte-carlo-runs    Bootstrap runs (default: 500)
--min-years           Min data years (default: 3.0)
--output              Output JSON path

AVAILABLE STRATEGIES
--------------------
• kalman     - Kalman Filter Mean Reversion
• hurst      - Hurst Exponent Regime Detection
• stat_arb   - Statistical Arbitrage
• liquidity  - Liquidity Provision
• ml         - Machine Learning
• mixer      - Ensemble Strategy

OUTPUT FILES
------------
• Console: Formatted text report
• JSON: reports/profitability_<strategy>_<timestamp>.json

EXIT CODES
----------
0: Confidence ≥50 (deployment candidate)
1: Confidence <50 (needs work)

REQUIREMENTS
------------
• Python 3.10+
• pandas, numpy, scipy
• ccxt (for crypto data)
• yfinance (for stock data)
• OpenQuant modules

DOCUMENTATION
-------------
• Full Guide: README_PROFITABILITY_TEST.md
• Quick Start: PROFITABILITY_TESTING_QUICKSTART.md
• Implementation: PROFITABILITY_TESTING_IMPLEMENTATION.md

SUPPORT
-------
1. Run environment check first
2. Review documentation
3. Check example scripts
4. Verify data source connectivity

================================================================================
