# src/__init__.py - Source Package Initialization

"""
Cross-Border Energy Interconnections Dashboard
Management Information System for ISA

This package contains all the core modules for the dashboard:
- cache_manager: Caching system (SQLite & JSON)
- data_collector: Data generation and collection
- web_crawler: News scraping and sentiment analysis
- data_processor: Advanced data processing and analytics
- visualizations: Reusable visualization components
"""

from .cache_manager import CacheManager, JsonCacheManager
from .data_collector import DataCollector
from .web_crawler import WebCrawler
from .data_processor import DataProcessor, EnergyMetricsCalculator, TradeAnalytics
from .visualizations import EnergyVisualizations, GridVisualizationUtils

__version__ = "1.0.0"
__author__ = "M.Tech Renewable Energy Student"
__maintainer__ = "International Solar Alliance"

__all__ = [
    "CacheManager",
    "JsonCacheManager",
    "DataCollector",
    "WebCrawler",
    "DataProcessor",
    "EnergyMetricsCalculator",
    "TradeAnalytics",
    "EnergyVisualizations",
    "GridVisualizationUtils",
]
