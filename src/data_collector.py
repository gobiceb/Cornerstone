# src/data_collector.py - Data Collection & Processing

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import requests
import json

import config
from .cache_manager import CacheManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collects bilateral trade, technical, and economic data."""
    
    def __init__(self):
        self.cache_manager = CacheManager()
    
    def generate_sample_bilateral_trade_data(self) -> pd.DataFrame:
        """Generate realistic bilateral electricity trade data."""
        
        data = []
        for exporter, importer in config.BILATERAL_TRADE_PAIRS:
            base_date = datetime.now()
            for days_back in range(0, 365, 30):  # Monthly data
                trade_date = base_date - timedelta(days=days_back)
                
                # Generate realistic trade volumes (TWh)
                volume = np.random.uniform(5, 50)
                price = np.random.uniform(30, 150)  # USD/MWh
                capacity = np.random.uniform(1000, 5000)  # MW
                
                # Renewable energy component
                renewable_pct = np.random.uniform(20, 80)
                carbon_intensity = 50 * (1 - renewable_pct/100)  # gCO2/kWh
                
                data.append({
                    "date": trade_date,
                    "exporter": exporter,
                    "importer": importer,
                    "trade_volume_twh": round(volume, 2),
                    "trade_value_usd_million": round(volume * price * 1000, 2),
                    "average_price_usd_mwh": round(price, 2),
                    "interconnection_capacity_mw": round(capacity, 0),
                    "renewable_energy_pct": round(renewable_pct, 2),
                    "carbon_intensity_gco2_kwh": round(carbon_intensity, 2),
                    "transmission_loss_pct": round(np.random.uniform(2, 8), 2),
                    "system_reliability_score": round(np.random.uniform(0.85, 0.99), 3)
                })
        
        df = pd.DataFrame(data)
        df = df.sort_values('date', ascending=False)
        
        # Cache the data
        self.cache_manager.set(
            "bilateral_trade_data",
            df.to_dict('records'),
            ttl_hours=config.CACHE_EXPIRY_HOURS
        )
        
        return df
    
    def generate_technical_metrics(self) -> pd.DataFrame:
        """Generate technical metrics for grid operations."""
        
        data = []
        countries = list(config.ISA_MEMBER_COUNTRIES.keys())[:15]
        
        for country in countries:
            base_date = datetime.now()
            for days_back in range(0, 365, 7):  # Weekly data
                metric_date = base_date - timedelta(days=days_back)
                
                data.append({
                    "date": metric_date,
                    "country": country,
                    "grid_capacity_mw": round(np.random.uniform(10000, 100000), 0),
                    "renewable_penetration_pct": round(np.random.uniform(10, 60), 2),
                    "transmission_losses_pct": round(np.random.uniform(2, 10), 2),
                    "average_load_factor_pct": round(np.random.uniform(40, 80), 2),
                    "reserve_margin_pct": round(np.random.uniform(10, 30), 2),
                    "system_stability_index": round(np.random.uniform(0.85, 0.99), 3),
                    "peak_demand_mw": round(np.random.uniform(5000, 50000), 0),
                    "available_capacity_mw": round(np.random.uniform(1000, 20000), 0)
                })
        
        df = pd.DataFrame(data)
        df = df.sort_values('date', ascending=False)
        
        self.cache_manager.set(
            "technical_metrics",
            df.to_dict('records'),
            ttl_hours=config.CACHE_EXPIRY_HOURS
        )
        
        return df
    
    def generate_economic_indicators(self) -> pd.DataFrame:
        """Generate economic indicators for energy markets."""
        
        data = []
        countries = list(config.ISA_MEMBER_COUNTRIES.keys())[:15]
        
        for country in countries:
            base_date = datetime.now()
            for days_back in range(0, 365, 30):  # Monthly data
                ind_date = base_date - timedelta(days=days_back)
                
                data.append({
                    "date": ind_date,
                    "country": country,
                    "avg_electricity_price_usd_mwh": round(np.random.uniform(30, 150), 2),
                    "cross_border_trade_volume_twh": round(np.random.uniform(0, 100), 2),
                    "total_trade_value_usd_million": round(np.random.uniform(100, 5000), 2),
                    "cost_savings_from_trade_pct": round(np.random.uniform(5, 25), 2),
                    "grid_investment_usd_million": round(np.random.uniform(50, 1000), 2),
                    "export_revenue_usd_million": round(np.random.uniform(100, 3000), 2),
                    "import_dependency_pct": round(np.random.uniform(5, 40), 2),
                    "electricity_access_pct": round(np.random.uniform(80, 100), 2)
                })
        
        df = pd.DataFrame(data)
        df = df.sort_values('date', ascending=False)
        
        self.cache_manager.set(
            "economic_indicators",
            df.to_dict('records'),
            ttl_hours=config.CACHE_EXPIRY_HOURS
        )
        
        return df
    
    def get_bilateral_trade_data(self, use_cache: bool = True) -> pd.DataFrame:
        """Get bilateral trade data with caching."""
        if use_cache:
            cached = self.cache_manager.get("bilateral_trade_data")
            if cached:
                logger.info("Using cached bilateral trade data")
                return pd.DataFrame(cached)
        
        logger.info("Generating new bilateral trade data")
        return self.generate_sample_bilateral_trade_data()
    
    def get_technical_metrics(self, use_cache: bool = True) -> pd.DataFrame:
        """Get technical metrics with caching."""
        if use_cache:
            cached = self.cache_manager.get("technical_metrics")
            if cached:
                logger.info("Using cached technical metrics")
                return pd.DataFrame(cached)
        
        logger.info("Generating new technical metrics")
        return self.generate_technical_metrics()
    
    def get_economic_indicators(self, use_cache: bool = True) -> pd.DataFrame:
        """Get economic indicators with caching."""
        if use_cache:
            cached = self.cache_manager.get("economic_indicators")
            if cached:
                logger.info("Using cached economic indicators")
                return pd.DataFrame(cached)
        
        logger.info("Generating new economic indicators")
        return self.generate_economic_indicators()
    
    def calculate_key_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate key statistics from data."""
        
        if df.empty:
            return {}
        
        stats = {
            "total_records": len(df),
            "date_range": f"{df['date'].min()} to {df['date'].max()}",
            "numeric_columns": {}
        }
        
        # Calculate statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            stats["numeric_columns"][col] = {
                "mean": round(df[col].mean(), 2),
                "min": round(df[col].min(), 2),
                "max": round(df[col].max(), 2),
                "std": round(df[col].std(), 2)
            }
        
        return stats
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for all datasets."""
        
        trade_data = self.get_bilateral_trade_data()
        tech_metrics = self.get_technical_metrics()
        econ_indicators = self.get_economic_indicators()
        
        return {
            "bilateral_trade": self.calculate_key_statistics(trade_data),
            "technical_metrics": self.calculate_key_statistics(tech_metrics),
            "economic_indicators": self.calculate_key_statistics(econ_indicators)
        }
