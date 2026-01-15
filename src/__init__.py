"""
Precipitation Pattern Analysis Package
"""

__version__ = "1.0.0"

from .data_collection import KMAHourlyDataCollector
from .event_extraction import PrecipitationEventManager
from .huff_analysis import HuffCurveAnalyzer
from .beta_analysis import BetaDistributionAnalyzer
from .comparison_analysis import HuffBetaComparison
from .visualization import (
    HuffVisualizer,
    BetaVisualizer,
    ComparisonVisualizer,
    TrendVisualizer
)

__all__ = [
    'KMAHourlyDataCollector',
    'PrecipitationEventManager',
    'HuffCurveAnalyzer',
    'BetaDistributionAnalyzer',
    'HuffBetaComparison',
    'HuffVisualizer',
    'BetaVisualizer',
    'ComparisonVisualizer',
    'TrendVisualizer'
]