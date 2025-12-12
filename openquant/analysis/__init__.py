"""Analysis modules for OpenQuant.

Includes TCA monitoring, execution quality tracking, sentiment analysis,
and performance attribution.
"""
from .attribution import (
    PerformanceAttributor,
    AttributionResult,
    TradeAttribution,
    quick_attribution
)
from .tca import TCAMonitor
from .execution_quality import (
    ExecutionQualityMonitor,
    ExecutionQualityMetrics,
    ExecutionAlert
)
from .sentiment import SentimentAnalyzer

__all__ = [
    "PerformanceAttributor",
    "AttributionResult", 
    "TradeAttribution",
    "quick_attribution",
    "TCAMonitor",
    "ExecutionQualityMonitor",
    "ExecutionQualityMetrics",
    "ExecutionAlert",
    "SentimentAnalyzer"
]
