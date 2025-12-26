"""
OpenQuant Robot - Modular Trading System

Following SOLID principles:
- Single Responsibility: Each module has one clear purpose
- Open/Closed: Easy to extend without modifying existing code
- Liskov Substitution: Strategies are interchangeable
- Interface Segregation: Clean, focused interfaces
- Dependency Inversion: Depend on abstractions, not concretions

Note: We use lazy imports via __all__ to avoid circular import issues.
Import directly from submodules: from robot.config import Config
"""

# Lazy imports - don't import at package level to avoid circular imports
__all__ = [
    "Config",
    "KalmanStrategy",
    "BaseStrategy",
    "DataFetcher",
    "RiskManager",
    "Trader",
    "Robot",
    "Backtester",
    "ParameterOptimizer",
]


def __getattr__(name):
    """Lazy loading to avoid circular imports."""
    if name == "Config":
        from .config import Config
        return Config
    elif name == "KalmanStrategy":
        from .strategy import KalmanStrategy
        return KalmanStrategy
    elif name == "BaseStrategy":
        from .strategy import BaseStrategy
        return BaseStrategy
    elif name == "DataFetcher":
        from .data_fetcher import DataFetcher
        return DataFetcher
    elif name == "RiskManager":
        from .risk_manager import RiskManager
        return RiskManager
    elif name == "Trader":
        from .trader import Trader
        return Trader
    elif name == "Robot":
        from .robot import Robot
        return Robot
    elif name == "Backtester":
        from .backtester import Backtester
        return Backtester
    elif name == "ParameterOptimizer":
        from .optimizer import ParameterOptimizer
        return ParameterOptimizer
    raise AttributeError(f"module 'robot' has no attribute {name!r}")


