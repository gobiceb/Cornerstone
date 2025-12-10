# config.py - Configuration & Constants

import os
from datetime import datetime, timedelta

# Application Settings
APP_NAME = "Cross-Border Energy Interconnections Dashboard"
APP_VERSION = "1.0.0"
DEBUG_MODE = os.getenv("DEBUG_MODE", "False") == "True"

# ISA Member Countries (International Solar Alliance)
ISA_MEMBER_COUNTRIES = {
    "India": "IN",
    "France": "FR",
    "Germany": "DE",
    "Brazil": "BR",
    "UAE": "AE",
    "China": "CN",
    "Japan": "JP",
    "Mexico": "MX",
    "South Korea": "KR",
    "Morocco": "MA",
    "Chile": "CL",
    "Bangladesh": "BD",
    "Denmark": "DK",
    "Egypt": "EG",
    "Kenya": "KE",
    "Mali": "ML",
    "Australia": "AU",
    "Canada": "CA",
    "Tanzania": "TZ",
    "Nigeria": "NG",
}

# Data Sources & APIs
DATA_SOURCES = {
    "IEA": "https://www.iea.org/data-and-statistics/data-tools/electricity-information-explorer",
    "World Bank": "https://data.worldbank.org/indicator/EG.ELC.EXPC.KH",
    "IRENA": "https://www.irena.org/",
    "ENTSO-E": "https://transparency.entsoe.eu/",
}

# News Sources for Web Crawler
NEWS_SOURCES = [
    "https://www.rechargenews.com/",
    "https://www.energymonitor.ai/",
    "https://www.power-technology.com/",
    "https://www.renewableenergyworld.com/",
    "https://www.cleantechnica.com/",
    "https://electrek.co/",
    "https://www.pv-magazine.com/",
]

# Technical Metrics Configuration
TECHNICAL_METRICS = [
    "Grid Capacity (MW)",
    "Renewable Penetration (%)",
    "Transmission Losses (%)",
    "Average Load Factor (%)",
    "Reserve Margin (%)",
    "System Stability Index",
]

# Economic Indicators Configuration
ECONOMIC_INDICATORS = [
    "Average Electricity Price (USD/MWh)",
    "Trade Volume (TWh)",
    "Total Trade Value (USD Million)",
    "Cost Savings from Trade (%)",
    "Grid Investment (USD Million)",
    "Export Revenue (USD Million)",
]

# Bilateral Trade Partners (Example pairs for demonstration)
BILATERAL_TRADE_PAIRS = [
    ("India", "Bangladesh"),
    ("France", "Germany"),
    ("Brazil", "Uruguay"),
    ("UAE", "Oman"),
    ("Mexico", "USA"),
    ("Denmark", "Sweden"),
    ("China", "Vietnam"),
    ("South Korea", "Japan"),
    ("Portugal", "Spain"),
    ("Australia", "New Zealand"),
]

# Cache Settings
CACHE_DIR = "data"
CACHE_FILE = os.path.join(CACHE_DIR, "cached_data.json")
CACHE_EXPIRY_HOURS = 24
CACHE_VERSION = "1.0"

# Web Crawler Settings
CRAWLER_UPDATE_INTERVAL_MINUTES = 60
CRAWLER_TIMEOUT_SECONDS = 10
MAX_NEWS_ARTICLES = 50
NEWS_RETENTION_DAYS = 30

# Visualization Settings
COLORSCALE = "Viridis"
MAP_CENTER = [20, 0]  # Global center for maps
MAP_ZOOM = 3

# Date Range Settings
DEFAULT_START_DATE = datetime.now() - timedelta(days=365)
DEFAULT_END_DATE = datetime.now()

# Performance Settings
BATCH_SIZE = 1000
MAX_WORKERS_THREADING = 5
REQUEST_TIMEOUT = 15

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "logs/dashboard.log"

# File paths for data
DATA_FILES = {
    "trade_data": os.path.join(CACHE_DIR, "trade_data.csv"),
    "technical_metrics": os.path.join(CACHE_DIR, "technical_metrics.csv"),
    "economic_indicators": os.path.join(CACHE_DIR, "economic_indicators.csv"),
    "news_data": os.path.join(CACHE_DIR, "news_data.json"),
}

# Create necessary directories
for directory in [CACHE_DIR, "logs", "exports"]:
    os.makedirs(directory, exist_ok=True)
