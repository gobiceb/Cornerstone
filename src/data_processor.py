# src/data_processor.py - Data Processing & Transformation

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and transforms energy data for analysis."""
    
    @staticmethod
    def aggregate_by_period(df: pd.DataFrame, date_col: str, 
                           value_col: str, period: str = 'M') -> pd.DataFrame:
        """
        Aggregate data by time period.
        
        Args:
            df: Input dataframe
            date_col: Date column name
            value_col: Value column to aggregate
            period: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Y' (yearly)
            
        Returns:
            Aggregated dataframe
        """
        df[date_col] = pd.to_datetime(df[date_col])
        return df.set_index(date_col).resample(period)[value_col].sum().reset_index()
    
    @staticmethod
    def calculate_moving_average(df: pd.DataFrame, column: str, 
                                window: int = 7) -> pd.Series:
        """Calculate moving average."""
        return df[column].rolling(window=window).mean()
    
    @staticmethod
    def normalize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize specified columns to 0-1 range."""
        df_normalized = df.copy()
        for col in columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        return df_normalized
    
    @staticmethod
    def calculate_yoy_growth(df: pd.DataFrame, value_col: str, 
                            date_col: str) -> pd.DataFrame:
        """Calculate year-over-year growth."""
        df[date_col] = pd.to_datetime(df[date_col])
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        
        pivot_df = df.pivot_table(
            values=value_col,
            index='month',
            columns='year',
            aggfunc='sum'
        )
        
        growth = pivot_df.pct_change(axis=1) * 100
        return growth
    
    @staticmethod
    def identify_outliers(df: pd.DataFrame, column: str, 
                         threshold: float = 2.0) -> pd.DataFrame:
        """Identify outliers using z-score method."""
        from scipy import stats
        
        df_clean = df.copy()
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = z_scores > threshold
        
        df_clean['is_outlier'] = False
        df_clean.loc[df_clean.index[outliers], 'is_outlier'] = True
        
        return df_clean
    
    @staticmethod
    def fill_missing_values(df: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """Fill missing values using specified method."""
        if method == 'forward':
            return df.fillna(method='ffill')
        elif method == 'backward':
            return df.fillna(method='bfill')
        elif method == 'interpolate':
            return df.interpolate()
        elif method == 'mean':
            return df.fillna(df.mean())
        else:
            return df
    
    @staticmethod
    def calculate_correlation(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix."""
        return df[numeric_cols].corr()
    
    @staticmethod
    def create_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create time-based features from date column."""
        df[date_col] = pd.to_datetime(df[date_col])
        
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        
        return df
    
    @staticmethod
    def resample_data(df: pd.DataFrame, date_col: str, 
                     freq: str = 'D', fill_method: str = 'forward') -> pd.DataFrame:
        """Resample time series data to different frequency."""
        df[date_col] = pd.to_datetime(df[date_col])
        df_resampled = df.set_index(date_col).resample(freq).last()
        
        if fill_method == 'forward':
            df_resampled = df_resampled.fillna(method='ffill')
        elif fill_method == 'interpolate':
            df_resampled = df_resampled.interpolate()
        
        return df_resampled.reset_index()
    
    @staticmethod
    def group_and_aggregate(df: pd.DataFrame, group_col: str, 
                           agg_dict: Dict) -> pd.DataFrame:
        """Group by column and apply multiple aggregations."""
        return df.groupby(group_col).agg(agg_dict).reset_index()
    
    @staticmethod
    def calculate_percentiles(df: pd.DataFrame, column: str, 
                             percentiles: List[float] = [25, 50, 75]) -> Dict:
        """Calculate percentiles for a column."""
        result = {}
        for p in percentiles:
            result[f"p{int(p)}"] = df[column].quantile(p/100)
        return result


class EnergyMetricsCalculator:
    """Calculates energy-specific metrics and KPIs."""
    
    @staticmethod
    def calculate_capacity_factor(actual_output: pd.Series, 
                                  rated_capacity: pd.Series) -> pd.Series:
        """Calculate capacity factor (%)."""
        return (actual_output / rated_capacity) * 100
    
    @staticmethod
    def calculate_renewable_integration_score(renewable_pct: float, 
                                             grid_stability: float) -> float:
        """Calculate renewable integration score (0-100)."""
        return renewable_pct * 0.6 + (grid_stability * 100) * 0.4
    
    @staticmethod
    def calculate_carbon_intensity(fossil_mix_pct: float) -> float:
        """
        Estimate carbon intensity based on fuel mix.
        Returns gCO2/kWh
        """
        # Average carbon intensities
        gas_intensity = 490
        coal_intensity = 820
        oil_intensity = 650
        
        avg_fossil = (gas_intensity + coal_intensity + oil_intensity) / 3
        return avg_fossil * (fossil_mix_pct / 100)
    
    @staticmethod
    def calculate_cost_benefit(trade_volume_gwh: float, 
                               avg_price_reduction_pct: float) -> float:
        """
        Calculate cost benefit of cross-border trade.
        Returns USD savings
        """
        avg_electricity_price = 100  # USD/MWh
        baseline_cost = trade_volume_gwh * 1000 * avg_electricity_price
        savings = baseline_cost * (avg_price_reduction_pct / 100)
        return savings
    
    @staticmethod
    def calculate_grid_reliability_score(system_stability: float,
                                        reserve_margin_pct: float,
                                        transmission_losses_pct: float) -> float:
        """Calculate overall grid reliability score (0-100)."""
        stability_score = system_stability * 100 * 0.4
        reserve_score = min(reserve_margin_pct, 30) * (100/30) * 0.35
        efficiency_score = (100 - transmission_losses_pct) * 0.25
        
        return stability_score + reserve_score + efficiency_score
    
    @staticmethod
    def project_renewable_growth(current_capacity_mw: float,
                                annual_growth_rate_pct: float,
                                years: int) -> List[float]:
        """Project renewable capacity growth."""
        projections = []
        capacity = current_capacity_mw
        
        for _ in range(years):
            capacity = capacity * (1 + annual_growth_rate_pct/100)
            projections.append(capacity)
        
        return projections


class TradeAnalytics:
    """Analyzes bilateral trade patterns."""
    
    @staticmethod
    def calculate_trade_balance(exports_gwh: float, 
                               imports_gwh: float) -> float:
        """Calculate trade balance (negative = deficit)."""
        return exports_gwh - imports_gwh
    
    @staticmethod
    def calculate_trade_intensity(bilateral_trade_gwh: float,
                                 total_consumption_gwh: float) -> float:
        """Calculate trade intensity (%)."""
        return (bilateral_trade_gwh / total_consumption_gwh) * 100 if total_consumption_gwh > 0 else 0
    
    @staticmethod
    def analyze_trade_complementarity(country1_profile: Dict,
                                     country2_profile: Dict) -> float:
        """
        Analyze trade complementarity between countries.
        Higher score = more complementary resources
        """
        score = 0
        
        # Generation timing complementarity
        if country1_profile.get('peak_hour', 12) != country2_profile.get('peak_hour', 12):
            score += 25
        
        # Renewable resource complementarity
        resource1 = country1_profile.get('primary_renewable', 'solar')
        resource2 = country2_profile.get('primary_renewable', 'wind')
        if resource1 != resource2:
            score += 25
        
        # Seasonal complementarity
        if country1_profile.get('peak_season') != country2_profile.get('peak_season'):
            score += 25
        
        # Geographic proximity (assume closer = better)
        if country1_profile.get('existing_interconnection', False):
            score += 25
        
        return min(score, 100)
    
    @staticmethod
    def calculate_trade_diversification_index(trade_partners: List[Tuple[str, float]]) -> float:
        """
        Calculate Herfindahl index for trade diversification.
        Higher value = more concentrated, Lower = more diversified
        """
        total_trade = sum(volume for _, volume in trade_partners)
        hhi = sum((volume / total_trade) ** 2 for _, volume in trade_partners)
        
        # Normalize to 0-100 scale
        n = len(trade_partners)
        return ((hhi - 1/n) / (1 - 1/n)) * 100 if n > 1 else 100
