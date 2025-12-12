# Profitability Testing Framework - Quick Start Guide

## 🚀 Quick Start (5 minutes)

### 1. Check Environment
```bash
python scripts/check_profitability_test_env.py
```

### 2. Run Basic Test
```bash
python scripts/test_profitability.py --strategy stat_arb --symbol BTC/USDT
```

### 3. Review Results
Check console output and `reports/profitability_*.json`

---

## 📋 Common Use Cases

### Test Strategy Before Production
```bash
python scripts/test_profitability.py \
  --strategy kalman \
  --symbol BTC/USDT \
  --return-target 0.50 \
  --max-drawdown 0.25 \
  --monte-carlo-runs 500
```

**Interpretation:**
- Confidence Score ≥70 → Deploy to production
- Confidence Score 60-69 → Paper trade first
- Confidence Score 50-59 → Extended testing needed
- Confidence Score <50 → Needs optimization

### Compare Multiple Strategies
```bash
# Test each strategy
for strategy in kalman hurst stat_arb; do
  python scripts/test_profitability.py \
    --strategy $strategy \
    --symbol BTC/USDT \
    --output reports/test_$strategy.json
done
```

### High-Confidence Production Validation
```bash
python scripts/test_profitability.py \
  --strategy stat_arb \
  --symbol BTC/USDT \
  --monte-carlo-runs 1000 \
  --min-years 3.5 \
  --return-target 0.60 \
  --max-drawdown 0.20
```

### Quick Development Test
```bash
python scripts/test_profitability.py \
  --strategy kalman \
  --symbol ETH/USDT \
  --timeframe 4h \
  --monte-carlo-runs 100 \
  --min-years 2.0
```

---

## 📊 Understanding the Output

### Console Report Sections

1. **IN-SAMPLE PERFORMANCE**
   - Training period metrics
   - Shows what strategy *could* achieve
   - Compare to out-of-sample to detect overfitting

2. **OUT-OF-SAMPLE PERFORMANCE**
   - Testing period metrics (most important!)
   - Real predictive performance
   - Used for go/no-go decision

3. **MONTE CARLO ROBUSTNESS TEST**
   - Bootstrap confidence intervals
   - Tests stability under different scenarios
   - Lower variance = more robust strategy

4. **TARGET VALIDATION**
   - Pass/fail on return and drawdown targets
   - Both must pass for production deployment

5. **FINAL ASSESSMENT**
   - Confidence score (0-100)
   - Actionable recommendation

### Key Metrics Explained

| Metric | Good Value | Description |
|--------|------------|-------------|
| **Sharpe Ratio** | >1.5 | Risk-adjusted return. Higher is better. |
| **Sortino Ratio** | >2.0 | Downside risk-adjusted return. |
| **Calmar Ratio** | >1.0 | Return / Max Drawdown. |
| **Omega Ratio** | >1.5 | Probability of gains vs losses. |
| **Max Drawdown** | <25% | Largest peak-to-trough decline. |
| **Win Rate** | >50% | Percentage of winning trades. |
| **Profit Factor** | >1.5 | Ratio of gross profit to gross loss. |

### Confidence Score Components

```
Total Score (0-100):
├─ Sharpe Ratio (0-30 points)
├─ Monte Carlo Stability (0-20 points)
├─ Target Achievement (0-30 points)
├─ Win Rate (0-10 points)
└─ Profit Factor (0-10 points)
```

---

## 🔍 Troubleshooting

### "No data returned for symbol"
**Solution:** Check internet connection, try different symbol format
```bash
# Try different formats
python scripts/test_profitability.py --symbol BTC/USDT  # Crypto
python scripts/test_profitability.py --symbol BTCUSDT  # Alternative
```

### "Strategy not found in registry"
**Solution:** List available strategies
```python
from openquant.strategies.registry import REGISTRY
print(list(REGISTRY.keys()))
```

### "Insufficient data"
**Solution:** Reduce minimum years or use larger timeframe
```bash
python scripts/test_profitability.py \
  --min-years 2.0 \
  --timeframe 4h
```

### Slow execution
**Solution:** Reduce Monte Carlo runs for development
```bash
python scripts/test_profitability.py \
  --monte-carlo-runs 100
```

---

## 📁 File Outputs

### JSON Report Structure
```json
{
  "strategy_name": "stat_arb",
  "confidence_score": 73.5,
  "recommendation": "GO - Strong recommendation",
  "out_of_sample_metrics": {
    "total_return": 0.567,
    "sharpe_ratio": 1.52,
    "max_drawdown": -0.186
  },
  "monte_carlo_results": {
    "mean_sharpe": 1.52,
    "percentile_5_sharpe": 1.23,
    "percentile_95_sharpe": 1.81
  }
}
```

### Output Locations
- **Reports:** `reports/profitability_<strategy>_<timestamp>.json`
- **Logs:** Console output (redirect with `> output.log`)

---

## 🎯 Production Deployment Checklist

Before deploying any strategy to production:

- [ ] Confidence score ≥60
- [ ] Out-of-sample Sharpe ratio >1.0
- [ ] Max drawdown <25%
- [ ] Total return >50%
- [ ] Monte Carlo 5th percentile Sharpe >0.5
- [ ] Win rate >45%
- [ ] Profit factor >1.2
- [ ] Tested on ≥3 years of data
- [ ] Reviewed individual window results
- [ ] Tested on multiple symbols (if applicable)
- [ ] Paper traded for 1-2 weeks minimum

---

## 💡 Best Practices

### 1. Always Start with Environment Check
```bash
python scripts/check_profitability_test_env.py
```

### 2. Test Multiple Timeframes
```bash
for tf in 1h 4h 1d; do
  python scripts/test_profitability.py \
    --strategy stat_arb \
    --timeframe $tf \
    --output reports/test_${tf}.json
done
```

### 3. Use Examples for Learning
```bash
python scripts/example_profitability_test.py --example 5
```

### 4. Review Window-by-Window Performance
Check `window_results` in JSON output to ensure consistency across different market regimes.

### 5. Compare Against Benchmark
Compare strategy Sharpe ratio against buy-and-hold:
- Buy-and-hold Sharpe ~0.5-1.0 for crypto
- Strategy should be >1.5 for meaningful improvement

### 6. Progressive Validation
1. Quick test (100 MC runs, 2 years)
2. Full test (500 MC runs, 3 years)
3. High-confidence test (1000 MC runs, 3.5 years)
4. Paper trading (1-2 weeks)
5. Live with small allocation (1-5%)
6. Scale up gradually

---

## 🔧 Advanced Usage

### Custom Strategy Testing
```python
from test_profitability import ProfitabilityTester

tester = ProfitabilityTester(
    strategy_name="my_strategy",
    symbols=["BTC/USDT"],
    timeframe="1h",
    return_target=0.50,
    max_drawdown_constraint=0.25,
    monte_carlo_runs=500
)

report = tester.run_profitability_test()
tester.save_report(report, output_path="reports/my_test.json")
```

### Accessing Detailed Results
```python
# Out-of-sample metrics
oos = report.out_of_sample_metrics
print(f"Sharpe: {oos.sharpe_ratio}")
print(f"Return: {oos.total_return}")

# Monte Carlo results
mc = report.monte_carlo_results
print(f"Sharpe CI: [{mc.percentile_5_sharpe}, {mc.percentile_95_sharpe}]")

# Window-by-window results
for window in report.details['window_results']:
    print(f"Window {window['window']}: "
          f"Train={window['train_return']:.1%}, "
          f"Test={window['test_return']:.1%}")
```

---

## 📚 Further Reading

- Full documentation: `scripts/README_PROFITABILITY_TEST.md`
- Example scripts: `scripts/example_profitability_test.py`
- Environment checker: `scripts/check_profitability_test_env.py`
- Strategy registry: `openquant/strategies/registry.py`

---

## 🆘 Support

If you encounter issues:

1. Run environment check first
2. Review error messages carefully
3. Check data source connectivity
4. Verify strategy exists in registry
5. Review examples for correct usage

Common error patterns:
- ImportError → Missing dependency (check requirements.txt)
- KeyError → Strategy not found (check registry)
- ValueError → Invalid parameters (check argument types)
- ConnectionError → Data source unavailable (check internet/API)
