# OpenQuant Research Log

## Purpose
Document every strategy attempt, what worked, what failed, and lessons learned until we find a profitable strategy.

---

## Philosophy: Pure Quantitative Approach

**This project is based ONLY on:**
- Mathematics (linear algebra, calculus, optimization)
- Statistics (hypothesis testing, regression, time series)
- Probability theory (stochastic processes, Bayesian inference)
- Quantitative finance (market microstructure, risk models)

**We do NOT use:**
- Technical analysis indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Chart patterns (head & shoulders, triangles, etc.)
- Retail trading "signals"
- Gut feeling or intuition

**Every strategy must answer:**
1. What is the mathematical model?
2. What statistical edge does it exploit?
3. What is the probability distribution of returns?
4. Can we prove it mathematically or statistically?

---

## Mathematical Foundations

### Current: Kalman Filter

**State-Space Model:**
```
State equation:     x(t+1) = x(t) + w(t),    w(t) ~ N(0, Q)
Observation:        z(t) = x(t) + v(t),      v(t) ~ N(0, R)
```

**Kalman Recursion:**
```
Predict:  x_pred = x_prev
          P_pred = P_prev + Q

Update:   K = P_pred / (P_pred + R)        # Kalman Gain
          x = x_pred + K * (z - x_pred)    # State Update
          P = (1 - K) * P_pred             # Covariance Update
```

**Trading Logic:**
```
deviation = observed_price - kalman_estimate
z_score = deviation / std(deviation, 50)

if z_score < -threshold: LONG   (price below fair value)
if z_score > +threshold: SHORT  (price above fair value)
```

**Statistical Assumption:**
- Prices are mean-reverting around a latent "true" price
- Deviations follow approximately normal distribution
- Mean reversion speed is related to half-life

### Future Models to Explore

| Model | Mathematical Basis | Use Case |
|-------|-------------------|----------|
| Ornstein-Uhlenbeck | dX = θ(μ - X)dt + σdW | Mean reversion with known half-life |
| GARCH(1,1) | σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1) | Volatility clustering |
| Cointegration | Y(t) = β·X(t) + ε(t), ε ~ I(0) | Pairs trading / Stat-arb |
| Hurst Exponent | E[R/S] ~ c·n^H | Trend vs mean-reversion regime |
| Hidden Markov | P(S(t)|S(t-1)) transition matrix | Regime detection |
| Kelly Criterion | f* = (p·b - q) / b | Optimal position sizing |

---

## Workflow to Follow

### Daily Research Cycle

```
1. HYPOTHESIS
   - What do I believe will work?
   - Why should this make money?
   - Write it down BEFORE testing

2. DESIGN
   - Define parameters
   - Define entry/exit rules
   - Define risk management

3. BACKTEST
   - Run: python mvp_robot.py --mode backtest --symbols EURUSD
   - Record ALL metrics in the results table
   - Be honest - no cherry-picking

4. ANALYZE
   - Did it meet targets? (Sharpe > 1, DD < 20%)
   - Why did it work or fail?
   - What patterns do you see?

5. ITERATE
   - Change ONE thing at a time
   - Document the change
   - Go back to step 3

6. DECIDE
   - If profitable: Move to paper trading
   - If not: Archive with lessons, try new idea
```

### Rules to Follow

1. **One change at a time** - Never change multiple parameters at once
2. **Write before you test** - Document hypothesis first
3. **No cherry-picking** - Record ALL results, even bad ones
4. **Statistical significance** - Need 100+ trades minimum
5. **Out-of-sample testing** - Don't optimize on data you'll trade
6. **Weekly review** - Every Sunday, review the week

### Path to Live Trading

```
[Idea] -> [Backtest] -> [Paper Trade 2 weeks] -> [Small Live] -> [Scale Up]
            |                    |                    |
            v                    v                    v
        Sharpe > 1?         Still works?         Still works?
        DD < 20%?           No curve fit?        Real slippage OK?
        100+ trades?        
```

### When to Abandon a Strategy

- Backtest Sharpe < 0.5 after 3 iterations
- Cannot explain WHY it should work
- Only works on one symbol/timeframe
- Requires unrealistic assumptions (no slippage, instant fills)

---

## Current Status
- **Active Strategy:** Kalman Filter Mean Reversion
- **Status:** Testing
- **Last Updated:** 2024-12-22

---

## Strategy Attempts

### 1. Kalman Filter Mean Reversion
**Date Started:** 2024-12-22  
**Status:** In Progress

#### Mathematical Model
```
State-Space Representation:
  x(t+1) = x(t) + w(t)     # True price follows random walk
  z(t) = x(t) + v(t)       # Observed price = true + noise

Where:
  x(t) = latent true price
  z(t) = observed market price
  w(t) ~ N(0, Q)           # Process noise
  v(t) ~ N(0, R)           # Measurement noise

Edge Exploited:
  - Market overreacts to noise
  - Price reverts to Kalman estimate (fair value)
  - Z-score of deviation is approximately N(0,1)
```

#### Statistical Tests Needed
- [ ] Test if residuals are stationary (ADF test)
- [ ] Test if z-scores are normally distributed (Shapiro-Wilk)
- [ ] Calculate half-life of mean reversion
- [ ] Test significance of returns (t-test vs zero)

#### Hypothesis
The market price oscillates around a "true" price. Using a Kalman Filter, we can estimate this true price and trade the deviation (mean reversion).

#### Parameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Process Noise (Q) | 1e-5 | How much true price varies |
| Measurement Noise (R) | 1e-3 | Observation noise |
| Signal Threshold | 1.5 | Z-score for entry |
| Stop Loss | 2x ATR | |
| Take Profit | 3x ATR | |

#### Backtest Results
| Symbol | Period | Return | Sharpe | Max DD | Trades | Notes |
|--------|--------|--------|--------|--------|--------|-------|
| EURUSD | 1 month | +0.03% | 0.03 | -0.88% | 54 | Initial test |
| | | | | | | |

#### Observations
- [ ] Too many trades? (54 in 1 month)
- [ ] Threshold too sensitive?
- [ ] Need longer test period

#### Next Steps
- [ ] Test with higher threshold (2.0, 2.5)
- [ ] Test on 4h timeframe instead of 1h
- [ ] Test on multiple symbols
- [ ] Run longer backtest (3-6 months)

#### Changes Made
| Date | Change | Result |
|------|--------|--------|
| 2024-12-22 | Initial implementation | Baseline established |
| | | |

---

## Ideas to Try

### Strategy Ideas
- [ ] Combine Kalman with trend filter (only trade with trend)
- [ ] Add volatility filter (skip high volatility periods)
- [ ] Pairs trading (EURUSD vs GBPUSD correlation)
- [ ] Time-of-day filter (avoid Asian session for EUR pairs)

### Parameter Ideas
- [ ] Adaptive threshold based on volatility
- [ ] Dynamic stop loss (trailing)
- [ ] Scale position size with signal strength

---

## Failed Strategies (Archive)

### Template
```
### [Strategy Name]
**Date:** YYYY-MM-DD
**Result:** Failed / Abandoned
**Reason:** [Why it didn't work]
**Lesson:** [What we learned]
```

---

## Key Metrics to Track

### Performance Metrics
| Metric | Formula | Target |
|--------|---------|--------|
| Sharpe Ratio | (μ - rf) / σ | > 1.0 |
| Sortino Ratio | (μ - rf) / σ_downside | > 1.5 |
| Max Drawdown | max(peak - trough) / peak | < 20% |
| Profit Factor | Σ(wins) / Σ(losses) | > 1.5 |
| Win Rate | n_wins / n_total | > 50% |

### Statistical Significance
| Test | Purpose | Requirement |
|------|---------|-------------|
| t-test on returns | Is mean return ≠ 0? | p-value < 0.05 |
| Number of trades | Statistical power | n > 100 |
| Out-of-sample test | Avoid overfitting | OOS Sharpe > 0.5 * IS Sharpe |
| Monte Carlo | Robustness | 95% CI doesn't include 0 |

### Risk Metrics
| Metric | Formula | Target |
|--------|---------|--------|
| VaR (95%) | Quantile(returns, 0.05) | < 2% daily |
| CVaR / ES | E[loss | loss > VaR] | < 3% daily |
| Volatility | σ(returns) * √252 | < 20% annual |

---

## Weekly Review Template

```markdown
## Week of [DATE]

### What I Tested
- 

### Results
- 

### What Worked
- 

### What Didn't Work
- 

### Lessons Learned
- 

### Next Week Plan
- 
```

---

## Resources & References

- Kalman Filter: https://en.wikipedia.org/wiki/Kalman_filter
- Mean Reversion: [Add papers/links you find useful]
- [Add more as you go]

---

## Notes

[Free-form notes, ideas, observations]


