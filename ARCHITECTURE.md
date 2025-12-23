# OpenQuant Architecture

## Modular Design (SOLID Principles)

The codebase follows SOLID principles and extreme modularization:

### Structure

```
OpenQuant/
  main.py              # Entry point (thin wrapper)
  robot/               # Main package
    __init__.py        # Package exports
    config.py          # Configuration (SRP)
    strategy.py        # Strategy interface + Kalman (Open/Closed)
    data_fetcher.py    # Data fetching (SRP)
    risk_manager.py    # Risk calculations (SRP)
    trader.py          # Trade execution (SRP)
    robot.py           # Orchestrator (SRP)
```

### SOLID Principles Applied

#### Single Responsibility Principle (SRP)
- **config.py**: Only handles configuration
- **strategy.py**: Only handles signal generation
- **data_fetcher.py**: Only handles data fetching
- **risk_manager.py**: Only handles risk calculations
- **trader.py**: Only handles trade execution
- **robot.py**: Only orchestrates components

#### Open/Closed Principle
- **BaseStrategy** interface allows adding new strategies without modifying existing code
- Example: Add `HurstStrategy(BaseStrategy)` without touching `KalmanStrategy`

#### Liskov Substitution Principle
- Any strategy implementing `BaseStrategy` can replace `KalmanStrategy`
- Strategies are interchangeable

#### Interface Segregation Principle
- Clean, focused interfaces (no fat interfaces)
- Each module exposes only what's needed

#### Dependency Inversion Principle
- `Robot` depends on `BaseStrategy` abstraction, not concrete `KalmanStrategy`
- Components depend on interfaces, not implementations

### Benefits

1. **Testability**: Each module can be tested independently
2. **Maintainability**: Changes isolated to one module
3. **Extensibility**: Easy to add new strategies, data sources, risk models
4. **Readability**: Clear separation of concerns
5. **Reusability**: Modules can be reused in other projects

### Adding a New Strategy

```python
# robot/strategy.py
class HurstStrategy(BaseStrategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Implementation
        pass

# main.py or robot/robot.py
robot = Robot(mode="paper", strategy=HurstStrategy())
```

No need to modify existing code!

### Module Responsibilities

| Module | Responsibility | Lines | Complexity |
|--------|---------------|-------|------------|
| config.py | Configuration | ~50 | Low |
| strategy.py | Signal generation | ~150 | Medium |
| data_fetcher.py | Data fetching | ~150 | Medium |
| risk_manager.py | Risk calculations | ~100 | Low |
| trader.py | Trade execution | ~200 | Medium |
| robot.py | Orchestration | ~200 | Medium |
| main.py | Entry point | ~40 | Low |

**Total: ~890 lines** (vs 942 in monolithic file, but much more maintainable)


