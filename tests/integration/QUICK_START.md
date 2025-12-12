# Integration Tests Quick Start

## Run All Tests

```bash
pytest tests/integration/
```

## Run Specific Workflow

```bash
# Data to Backtest
pytest tests/integration/test_data_to_backtest.py

# Backtest to Optimization
pytest tests/integration/test_backtest_to_optimization.py

# Optimization to Paper Trading
pytest tests/integration/test_optimization_to_paper_trading.py

# Paper Trading to Risk Checks
pytest tests/integration/test_paper_trading_to_risk_checks.py

# Full End-to-End
pytest tests/integration/test_full_end_to_end.py

# MT5 Integration
pytest tests/integration/test_mt5_integration.py

# Research Workflow
pytest tests/integration/test_research_workflow.py

# Edge Cases
pytest tests/integration/test_edge_cases_and_errors.py
```

## Run Specific Test

```bash
pytest tests/integration/test_full_end_to_end.py::test_full_pipeline_single_strategy
```

## Verbose Output

```bash
pytest tests/integration/ -v
```

## With Coverage

```bash
pytest tests/integration/ --cov=openquant --cov-report=html
```

## Key Test Scenarios

### Complete Pipeline Test
```python
# tests/integration/test_full_end_to_end.py::test_full_pipeline_single_strategy
# Tests: Data → Optimize → Backtest → Paper Trade → Risk Check
```

### Multi-Strategy Portfolio
```python
# tests/integration/test_full_end_to_end.py::test_multi_strategy_pipeline
# Tests: Multiple strategies running in parallel
```

### Live Trading Simulation
```python
# tests/integration/test_full_end_to_end.py::test_live_update_simulation
# Tests: Rolling window updates like live trading
```

### MT5 Integration
```python
# tests/integration/test_mt5_integration.py::test_mt5_integration_with_paper_trading
# Tests: Backtest → Signals → MT5 allocation
```

### Risk Management
```python
# tests/integration/test_paper_trading_to_risk_checks.py::test_integrated_risk_workflow
# Tests: Trading → Loss → Circuit Breaker → Halt
```

## Common Fixtures

```python
@pytest.fixture
def sample_ohlcv_df():
    """200-bar synthetic OHLCV data"""
    
@pytest.fixture
def mock_mt5_module():
    """Complete MT5 module mock"""
    
@pytest.fixture
def clean_risk_state():
    """Clean risk management state"""
    
@pytest.fixture
def small_param_grid():
    """Small parameter grid for fast tests"""
```

## Debugging Tips

### Show Print Statements
```bash
pytest tests/integration/ -s
```

### Stop on First Failure
```bash
pytest tests/integration/ -x
```

### Run Last Failed Tests
```bash
pytest tests/integration/ --lf
```

### Show Locals on Failure
```bash
pytest tests/integration/ -l
```

### Full Traceback
```bash
pytest tests/integration/ --tb=long
```

## Expected Output

```
tests/integration/test_data_to_backtest.py ....                    [  8%]
tests/integration/test_backtest_to_optimization.py ...             [ 14%]
tests/integration/test_optimization_to_paper_trading.py ...        [ 20%]
tests/integration/test_paper_trading_to_risk_checks.py .....       [ 30%]
tests/integration/test_full_end_to_end.py ......                   [ 42%]
tests/integration/test_mt5_integration.py ..........               [ 62%]
tests/integration/test_research_workflow.py ........               [ 78%]
tests/integration/test_edge_cases_and_errors.py ...........        [100%]

======================== 50 passed in 45.23s =========================
```

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the project root
cd /path/to/OpenQuant

# Install dependencies
pip install -r requirements.txt
```

### Fixture Not Found
```bash
# Ensure conftest.py is present
ls tests/integration/conftest.py
```

### Slow Tests
```bash
# Run only fast tests (mark slow tests)
pytest tests/integration/ -m "not slow"
```

### Permission Errors
```bash
# Check temp directory permissions
pytest tests/integration/ --basetemp=/tmp/pytest
```

## Performance

- **Average test duration**: 0.5-2s per test
- **Total suite runtime**: 30-60s
- **Parallel execution**: Can reduce to ~15s

## CI/CD Example

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/integration/ -v --cov=openquant
```

## Quick Reference

| Test File | Purpose | Key Tests |
|-----------|---------|-----------|
| test_data_to_backtest.py | Data → Backtest | 4 tests |
| test_backtest_to_optimization.py | Backtest → Optimize | 3 tests |
| test_optimization_to_paper_trading.py | Optimize → Paper | 3 tests |
| test_paper_trading_to_risk_checks.py | Paper → Risk | 5 tests |
| test_full_end_to_end.py | Complete Pipeline | 6 tests |
| test_mt5_integration.py | MT5 Broker | 10 tests |
| test_research_workflow.py | Research Engine | 8 tests |
| test_edge_cases_and_errors.py | Error Handling | 11 tests |

Total: **50+ integration tests**

## Support

For issues or questions:
1. Check test output for specific errors
2. Review fixture definitions in conftest.py
3. Ensure all dependencies are installed
4. Verify Python version (3.10+)
5. Check AGENTS.md for project-specific setup
