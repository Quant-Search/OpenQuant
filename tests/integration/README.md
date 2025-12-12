# Integration Tests

Comprehensive integration tests covering end-to-end workflows in OpenQuant.

## Test Structure

### Core Workflow Tests

1. **test_data_to_backtest.py**
   - Data fetching → Signal generation → Backtesting
   - Tests multiple strategies and cost models
   - Validates stop-loss and take-profit functionality

2. **test_backtest_to_optimization.py**
   - Backtesting → Hyperparameter optimization with Optuna
   - Tests optimization across different strategy types
   - Validates that optimization improves performance

3. **test_optimization_to_paper_trading.py**
   - Optimization → Paper trading deployment
   - Tests multi-symbol portfolio management
   - Validates stop-loss order execution

4. **test_paper_trading_to_risk_checks.py**
   - Paper trading → Risk management validation
   - Tests kill switch and circuit breaker functionality
   - Validates guardrails enforcement

5. **test_full_end_to_end.py**
   - Complete pipeline integration tests
   - Multi-strategy portfolio workflows
   - Live update simulation
   - Portfolio rebalancing
   - Stress testing under high volatility

### Specialized Tests

6. **test_mt5_integration.py**
   - MetaTrader 5 broker integration (fully mocked)
   - Order execution and position management
   - Symbol mapping and signal export
   - Risk control integration

7. **test_research_workflow.py**
   - Research engine workflows
   - Walk-forward optimization
   - Concentration limits
   - Regime detection
   - Results persistence

8. **test_edge_cases_and_errors.py**
   - Edge case handling
   - Error recovery
   - Data quality issues
   - Extreme market conditions

## Fixtures

All tests use fixtures defined in `conftest.py`:

- `sample_ohlcv_df`: Synthetic OHLCV data for testing
- `mock_mt5_module`: Complete MT5 module mock
- `mock_data_loader`: Mocked data loader
- `temp_state_files`: Temporary files for risk management
- `clean_risk_state`: Clean state for risk tests
- `small_param_grid`: Small parameter grid for fast tests

## Running Tests

Run all integration tests:
```bash
pytest tests/integration/
```

Run specific test file:
```bash
pytest tests/integration/test_full_end_to_end.py
```

Run with verbose output:
```bash
pytest tests/integration/ -v
```

Run specific test:
```bash
pytest tests/integration/test_full_end_to_end.py::test_full_pipeline_single_strategy
```

## Test Coverage

The integration tests cover:

✅ **Data Workflows**
- Data fetching and caching
- OHLCV validation
- Multiple data sources

✅ **Strategy Workflows**
- Signal generation
- Multiple strategy types (Kalman, Hurst, StatArb, Liquidity)
- Strategy composition and mixing

✅ **Backtesting**
- Vectorized backtesting
- Transaction costs (fees, slippage, spread)
- Stop-loss and take-profit
- Leverage and swap costs
- Performance metrics calculation

✅ **Optimization**
- Optuna hyperparameter search
- Walk-forward optimization
- Parameter grid narrowing
- Deflated Sharpe ratio

✅ **Paper Trading**
- Portfolio state management
- Order execution
- Position sizing
- Stop-loss/take-profit checks
- Daily loss limits

✅ **Risk Management**
- Kill switch activation
- Circuit breaker (daily loss, drawdown, volatility)
- Guardrails (max drawdown, CVaR, daily loss)
- Concentration limits
- Exposure allocation

✅ **MT5 Integration**
- Broker connection (mocked)
- Order placement
- Position management
- Symbol mapping
- Signal export for EAs

✅ **Edge Cases**
- Empty data handling
- Insufficient data
- NaN values
- Extreme volatility
- Rapid signal changes
- Invalid inputs

## Performance Considerations

Integration tests are designed to be fast:
- Use small parameter grids (2-3 values per parameter)
- Limited Optuna trials (3-5 trials)
- Synthetic data (200 bars)
- Parallel execution where possible

Typical runtime: 30-60 seconds for full test suite

## Mocking Strategy

Tests use comprehensive mocking for:
- **MetaTrader5**: Complete module mock with all required methods
- **Data sources**: Synthetic data generation instead of API calls
- **File I/O**: Temporary directories for state files
- **Risk controls**: Isolated state for each test

This ensures:
- No external dependencies
- Fast execution
- Reproducible results
- Safe for CI/CD

## Key Integration Scenarios

### 1. Research → Production Pipeline
```
Data Fetch → Optimize → Validate → Backtest → Deploy
```

### 2. Live Trading Loop
```
Market Update → Signal Generation → Risk Check → Order Execution → Monitor
```

### 3. Risk Management Chain
```
Position Update → Exposure Check → Circuit Breaker → Kill Switch → Alert
```

### 4. Multi-Strategy Portfolio
```
Strategy A ↘
Strategy B → Allocation → Risk Filter → Paper/Live Trading
Strategy C ↗
```

## Notes

- All tests are independent and can run in any order
- Tests use deterministic random seeds for reproducibility
- Temporary files are cleaned up automatically
- Risk management state is isolated per test
- No real trading or API calls are made

## Future Enhancements

Potential additions:
- Real broker integration tests (optional, behind feature flag)
- Load testing with large portfolios
- Time-series data validation
- Cross-strategy correlation tests
- Advanced risk scenario testing
