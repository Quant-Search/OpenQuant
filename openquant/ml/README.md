# Automated Model Retraining Pipeline

Automated model retraining system that monitors live strategy performance using deflated Sharpe ratio (DSR), triggers retraining when performance degrades, and deploys new models only when significant improvement is validated.

## Features

- **Performance Monitoring**: Continuously tracks strategy performance using DSR
- **Automatic Triggering**: Initiates retraining when DSR drops below threshold
- **Walk-Forward Optimization**: Retains models using rolling train/test splits
- **Holdout Validation**: Validates new models on unseen data before deployment
- **Conservative Deployment**: Only deploys models with >10% improvement (configurable)
- **Rate Limiting**: Prevents over-retraining with configurable limits
- **Persistence**: Saves models and retraining history to disk
- **Thread-Safe**: Supports concurrent access for live trading systems

## Quick Start

```python
from openquant.ml.retrain_pipeline import RetrainingPipeline, RetrainingConfig
from openquant.strategies.quant.stat_arb import StatArb

# Configure pipeline
config = RetrainingConfig(
    dsr_threshold=1.0,              # Trigger when DSR < 1.0
    improvement_threshold=0.10,     # Deploy if improvement > 10%
    min_samples_retrain=500,
    lookback_window=1000,
    holdout_fraction=0.15,
    monitoring_interval_hours=6,
    max_retrain_per_day=4
)

# Initialize pipeline
pipeline = RetrainingPipeline(config=config)

# In your live trading loop:
should_retrain, reason = pipeline.update_and_check(
    strategy_id="my_strategy",
    equity_curve=equity_curve,
    returns=returns,
    num_trials=10
)

if should_retrain:
    deployed, new_params = pipeline.run_retrain_cycle(
        strategy_id="my_strategy",
        df=historical_data,
        strategy_factory=lambda **params: StatArb(**params),
        current_params=current_params,
        param_grid=param_grid,
        fee_bps=2.0,
        weight=1.0
    )
    
    if deployed:
        # Update your live strategy with new_params
        strategy.update_params(new_params)
```

## Architecture

### Components

1. **PerformanceMonitor**: Tracks live strategy performance and calculates DSR
2. **ModelRetrainer**: Executes walk-forward optimization to find best parameters
3. **ModelValidator**: Validates new models against holdout data
4. **RetrainingPipeline**: Main orchestrator that coordinates all components

### Workflow

```
Live Performance → Monitor DSR → Trigger Check
                                      ↓
                                DSR < Threshold?
                                      ↓ YES
                         Walk-Forward Optimization
                                      ↓
                          Find Best Parameters
                                      ↓
                         Holdout Validation
                                      ↓
                        Improvement > 10%?
                                      ↓ YES
                           Deploy New Model
                                      ↓
                          Save to Disk
```

## Configuration

### RetrainingConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dsr_threshold` | 1.0 | DSR value below which retraining is triggered |
| `improvement_threshold` | 0.10 | Minimum improvement (10%) required for deployment |
| `min_samples_retrain` | 500 | Minimum samples needed to trigger retraining |
| `lookback_window` | 1000 | Number of recent samples used for retraining |
| `holdout_fraction` | 0.15 | Fraction of data held out for validation (15%) |
| `monitoring_interval_hours` | 6 | Minimum hours between retraining attempts |
| `max_retrain_per_day` | 4 | Maximum retraining attempts per day |
| `trials_per_strategy` | 50 | Number of trials for DSR calculation |
| `wfo_n_splits` | 4 | Number of walk-forward splits |
| `wfo_train_frac` | 0.7 | Train/test ratio per WFO split |
| `enable_cpcv` | False | Use Combinatorially Purged CV |
| `model_save_dir` | `data/models` | Directory for saved models |
| `metrics_save_dir` | `data/retrain_metrics` | Directory for metrics |

## API Reference

### RetrainingPipeline

Main class for automated retraining.

#### Methods

**`update_and_check(strategy_id, equity_curve, returns, num_trials=1)`**
- Updates performance metrics and checks if retraining is needed
- Returns: `(should_retrain: bool, reason: str)`

**`run_retrain_cycle(strategy_id, df, strategy_factory, current_params, param_grid, fee_bps=2.0, weight=1.0)`**
- Runs complete retraining cycle: retrain → validate → deploy
- Returns: `(deployed: bool, new_params: Dict[str, Any])`

**`load_latest_params(strategy_id)`**
- Loads the most recently saved parameters for a strategy
- Returns: `Optional[Dict[str, Any]]`

**`get_stats(strategy_id)`**
- Returns statistics about retraining history
- Returns: `Dict` with `total_retrains`, `deployments`, `avg_improvement`, etc.

**`get_retrain_history(strategy_id)`**
- Returns full history of retraining events
- Returns: `List[RetrainingEvent]`

## Deflated Sharpe Ratio (DSR)

The pipeline uses DSR to account for multiple testing and sample size:

```python
DSR = (SR - E_max) / σ_SR
```

Where:
- `SR` = Observed Sharpe ratio
- `E_max` = Expected maximum Sharpe under null hypothesis (multiple testing)
- `σ_SR` = Standard error of Sharpe estimate

DSR < 1.0 indicates that the strategy's performance is not significantly better than what could be expected by chance given the number of trials.

## Walk-Forward Optimization (WFO)

The retraining process uses WFO to avoid overfitting:

1. Split data into rolling windows
2. For each window:
   - Train on 70% (find best params)
   - Test on 30% (evaluate performance)
3. Aggregate test results across all windows
4. Vote for most common parameter combination

## Holdout Validation

Before deployment, new models are validated:

1. Reserve 15% of data as holdout set (never seen during WFO)
2. Evaluate old model on holdout
3. Evaluate new model on holdout
4. Calculate improvement: `(new_sharpe - old_sharpe) / |old_sharpe|`
5. Deploy only if improvement > threshold (default 10%)

## Persistence

### Model Files
- Location: `data/models/{strategy_id}_params.json`
- Format: JSON with params, timestamp, and version

```json
{
  "strategy_id": "stat_arb_btc",
  "params": {
    "lookback": 50,
    "entry_threshold": 2.0,
    "exit_threshold": 0.5
  },
  "timestamp": "2024-01-15T10:30:00",
  "version": 3
}
```

### Retraining Events
- Location: `data/retrain_metrics/{strategy_id}_events.jsonl`
- Format: JSON Lines (one event per line)

```json
{"timestamp": "2024-01-15T10:30:00", "trigger_reason": "DSR=0.85", "old_sharpe": 1.2, "new_sharpe": 1.5, "improvement": 0.25, "deployed": true, ...}
```

## Example: Live Trading Integration

```python
import time
from openquant.ml.retrain_pipeline import RetrainingPipeline, RetrainingConfig

# Initialize pipeline
pipeline = RetrainingPipeline(RetrainingConfig(
    dsr_threshold=1.0,
    improvement_threshold=0.10,
    monitoring_interval_hours=6
))

# Load latest params or use defaults
params = pipeline.load_latest_params("my_strategy") or default_params
strategy = create_strategy(**params)

# Live trading loop
while True:
    # Collect recent performance data
    equity_curve, returns = get_recent_performance()
    
    # Check if retraining needed
    should_retrain, reason = pipeline.update_and_check(
        strategy_id="my_strategy",
        equity_curve=equity_curve,
        returns=returns,
        num_trials=10
    )
    
    if should_retrain:
        print(f"Retraining triggered: {reason}")
        
        # Get historical data
        df = fetch_historical_data()
        
        # Run retraining
        deployed, new_params = pipeline.run_retrain_cycle(
            strategy_id="my_strategy",
            df=df,
            strategy_factory=create_strategy,
            current_params=params,
            param_grid=param_grid
        )
        
        if deployed:
            print(f"New model deployed: {new_params}")
            params = new_params
            strategy = create_strategy(**params)
    
    # Continue trading...
    time.sleep(3600)  # Check every hour
```

## Best Practices

1. **Choose DSR Threshold Carefully**: 
   - DSR > 1.5: High confidence in edge
   - DSR > 1.0: Statistically significant
   - DSR < 1.0: Performance not better than chance

2. **Set Appropriate Improvement Threshold**:
   - Too low: Deploy noisy improvements
   - Too high: Miss genuine improvements
   - Default 10% is a good starting point

3. **Monitor Retraining Frequency**:
   - Use `max_retrain_per_day` to prevent over-retraining
   - Review `get_stats()` regularly

4. **Validate Parameter Grid**:
   - Ensure grid covers reasonable parameter space
   - Too large: Overfitting risk
   - Too small: May miss optimal parameters

5. **Keep Sufficient Data**:
   - `lookback_window` should be at least 1000 samples
   - More data = more reliable retraining

## Limitations

- Requires sufficient historical data for reliable WFO
- Assumes stationarity within lookback window
- May not adapt quickly to regime changes
- Computational cost scales with parameter grid size

## See Also

- [Deflated Sharpe Ratio](../evaluation/deflated_sharpe.py)
- [Walk-Forward Optimization](../evaluation/wfo.py)
- [ML Strategy](../strategies/ml_strategy.py)
- [Example Script](../../scripts/example_retrain_pipeline.py)
