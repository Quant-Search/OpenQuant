"""
OpenQuant Robot - Modular Trading System

Following SOLID principles:
- Single Responsibility: Each module has one clear purpose
- Open/Closed: Easy to extend without modifying existing code
- Liskov Substitution: Strategies are interchangeable
- Interface Segregation: Clean, focused interfaces
- Dependency Inversion: Depend on abstractions, not concretions
"""

from .config import Config
from .strategy import KalmanStrategy, BaseStrategy
from .data_fetcher import DataFetcher
from .risk_manager import RiskManager
from .trader import Trader
from .robot import Robot

__all__ = [
    "Config",
    "KalmanStrategy",
    "BaseStrategy",
    "DataFetcher",
    "RiskManager",
    "Trader",
    "Robot",
]


