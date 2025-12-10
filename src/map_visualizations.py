# src/map_visualizations.py - Geographic Map Visualizations for Energy Interconnections

import folium
from folium import plugins
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st


class InterconnectionMapper:
    """Creates interactive maps of energy interconnections and cross-border flows."""
    
    # Country coordinates (latitude, longitude)
    COUNTRY_COORDINATES = {
        "India": [20.5937, 78.9629],
        "France": [46.2276, 2.2137],
        "Germany": [51.1657, 10.4515],
        "Brazil": [-14.2350, -51.9253],
        "UAE": [23.4241, 53.8478],
        "China": [35.8617, 104.1954],
        "Japan": [36.2048, 138.2529],
        "Mexico": [23.6345, -102.5528],
        "South Korea": [35.9078, 127.7669],
        "Morocco": [31.7917, -7.0926],
        "Chile": [-35.6751, -71.5430],
        "Bangladesh": [23.6850, 90.3563],
        "Denmark": [56.2639, 9.5018],
        "Egypt": [26.8206, 30.8025],
        "Kenya": [-0.0236, 37.9062],
        "Mali": [17.5707, -3.9962],
        "Australia": [-25.2744, 133.7751],
        "Canada": [56.1304, -106.3468],
        "Tanzania": [-6.3690, 34.8888],
        "Nigeria": [9.0820, 8.6753],
        "Nepal": [28.3949, 84.1240],
        "Bhutan": [27.5142, 90.4336],
        "Pakistan": [30.3753, 69.3451],
        "Thailand": [15.8700, 100.9925],
        "Laos": [19.8523, 102.4955],
        "Malaysia": [4.2105, 101.6964],
        "Singapore": [1.3521, 103.8198],
        "Vietnam": [14.0583, 108.2772],
        "Indonesia": [-0.7893, 113.9213],
        "Philippines": [12.8797, 121.7740],
    }
    
    def __init__(self, zoom_level: int = 4, center_lat: float = 20, center_lon: float = 0):
        """Initialize mapper with default zoom and center."""
        self.zoom_level = zoom_level
        self.center_lat = center_lat
        self.center_lon = center_lon
    
    def create_base_map(self) -> folium.Map:
        """Create base folium map."""
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom_level,
            tiles="OpenStreetMap"
        )
        return m
    
    def add_country_markers(self, m: folium.Map, countries: List[str],
                           popup_template: str = "{country}") -> folium.Map:
        """Add markers for countries on the map."""
        for country in countries:
            if country in self.COUNTRY_COORDINATES:
                lat, lon = self.COUNTRY_COORDINATES[country]
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    popup=popup_template.format(country=country),
                    color='#1f77b4',
                    fill=True,
                    fillColor='#1f77b4',
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
        
        return m
    
    def add_interconnections(self, m: folium.Map, 
                            connections: List[Tuple[str, str, float]]) -> folium.Map:
        """
        Add interconnection lines between countries.
        
        Args:
            m: Folium map object
            connections: List of tuples (country1, country2, capacity_mw)
        """
        for country1, country2, capacity in connections:
            if country1 in self.COUNTRY_COORDINATES and country2 in self.COUNTRY_COORDINATES:
                lat1, lon1 = self.COUNTRY_COORDINATES[country1]
                lat2, lon2 = self.COUNTRY_COORDINATES[country2]
                
                # Line color and weight based on capacity
                weight = min(10, max(2, capacity / 500))  # Scale weight by capacity
                color = self._get_color_by_capacity(capacity)
                
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color=color,
                    weight=weight,
                    opacity=0.7,
                    popup=f"{country1} ↔ {country2}<br>Capacity: {capacity:.0f} MW"
                ).add_to(m)
        
        return m
    
    def add_trade_flows(self, m: folium.Map,
                       trade_data: pd.DataFrame) -> folium.Map:
        """
        Add animated trade flow arrows between countries.
        
        Args:
            m: Folium map object
            trade_data: DataFrame with columns: exporter, importer, trade_volume_twh
        """
        for _, row in trade_data.iterrows():
            exporter = row['exporter']
            importer = row['importer']
            volume = row['trade_volume_twh']
            
            if exporter in self.COUNTRY_COORDINATES and importer in self.COUNTRY_COORDINATES:
                lat1, lon1 = self.COUNTRY_COORDINATES[exporter]
                lat2, lon2 = self.COUNTRY_COORDINATES[importer]
                
                # Arrow color based on volume
                color = self._get_color_by_volume(volume)
                weight = min(8, max(1, volume / 10))
                
                # Main flow line
                folium.PolyLine(
                    locations=[[lat1, lon1], [lat2, lon2]],
                    color=color,
                    weight=weight,
                    opacity=0.6,
                    popup=f"{exporter} → {importer}<br>Volume: {volume:.2f} TWh"
                ).add_to(m)
        
        return m
    
    def add_renewable_penetration_heatmap(self, m: folium.Map,
                                         penetration_data: pd.DataFrame) -> folium.Map:
        """
        Add heat layer showing renewable energy penetration by country.
        
        Args:
            m: Folium map object
            penetration_data: DataFrame with columns: country, renewable_penetration_pct
        """
        # Get latest data for each country
        latest_data = penetration_data.sort_values('date').drop_duplicates('country', keep='last')
        
        heat_points = []
        for _, row in latest_data.iterrows():
            country = row['country']
            penetration = row['renewable_penetration_pct']
            
            if country in self.COUNTRY_COORDINATES:
                lat, lon = self.COUNTRY_COORDINATES[country]
                # Heat intensity based on penetration (0-1)
                heat_points.append([lat, lon, penetration / 100])
        
        # Add heat layer
        plugins.HeatMap(heat_points, radius=50, blur=15, max_zoom=1).add_to(m)
        
        return m
    
    def add_grid_capacity_circles(self, m: folium.Map,
                                 capacity_data: pd.DataFrame) -> folium.Map:
        """
        Add proportional circles showing grid capacity by country.
        
        Args:
            m: Folium map object
            capacity_data: DataFrame with columns: country, grid_capacity_mw
        """
        # Get latest capacity for each country
        latest_data = capacity_data.sort_values('date').drop_duplicates('country', keep='last')
        
        # Find min/max for scaling
        min_capacity = latest_data['grid_capacity_mw'].min()
        max_capacity = latest_data['grid_capacity_mw'].max()
        
        for _, row in latest_data.iterrows():
            country = row['country']
            capacity = row['grid_capacity_mw']
            
            if country in self.COUNTRY_COORDINATES:
                lat, lon = self.COUNTRY_COORDINATES[country]
                
                # Scale radius (5-30 pixels based on capacity)
                radius = 5 + (capacity - min_capacity) / (max_capacity - min_capacity) * 25
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    popup=f"{country}<br>Grid Capacity: {capacity:.0f} MW",
                    color='#d62728',
                    fill=True,
                    fillColor='#ff7f0e',
                    fillOpacity=0.6,
                    weight=2
                ).add_to(m)
        
        return m
    
    def create_interconnection_network_map(self, countries: List[str],
                                          connections: List[Tuple[str, str, float]],
                                          title: str = "Energy Interconnections Network") -> folium.Map:
        """
        Create a complete network map showing countries and interconnections.
        
        Args:
            countries: List of countries to show
            connections: List of (country1, country2, capacity_mw) tuples
            title: Map title
        
        Returns:
            Folium map object
        """
        m = self.create_base_map()
        m = self.add_country_markers(m, countries)
        m = self.add_interconnections(m, connections)
        
        # Add title using HTML
        title_html = f'''
                     <div style="position: fixed; 
                     top: 10px; left: 50px; width: 300px; height: 60px; 
                     background-color: white; border:2px solid grey; z-index:9999; 
                     font-size:16px; padding: 10px">
                     <b>{title}</b>
                     </div>
                     '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def create_trade_flow_map(self, trade_data: pd.DataFrame,
                             title: str = "Cross-Border Electricity Trade") -> folium.Map:
        """
        Create map showing bilateral trade flows.
        
        Args:
            trade_data: DataFrame with exporter, importer, trade_volume_twh
            title: Map title
        
        Returns:
            Folium map object
        """
        m = self.create_base_map()
        
        # Get unique countries
        countries = list(set(trade_data['exporter'].tolist() + trade_data['importer'].tolist()))
        m = self.add_country_markers(m, countries)
        m = self.add_trade_flows(m, trade_data)
        
        # Add title
        title_html = f'''
                     <div style="position: fixed; 
                     top: 10px; left: 50px; width: 300px; height: 60px; 
                     background-color: white; border:2px solid grey; z-index:9999; 
                     font-size:16px; padding: 10px">
                     <b>{title}</b>
                     </div>
                     '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def create_renewable_penetration_map(self, penetration_data: pd.DataFrame,
                                        title: str = "Renewable Energy Penetration") -> folium.Map:
        """
        Create heat map showing renewable penetration across countries.
        
        Args:
            penetration_data: DataFrame with country, renewable_penetration_pct
            title: Map title
        
        Returns:
            Folium map object
        """
        m = self.create_base_map()
        m = self.add_renewable_penetration_heatmap(m, penetration_data)
        
        # Get countries and add markers
        countries = penetration_data['country'].unique().tolist()
        m = self.add_country_markers(m, countries)
        
        # Add title and legend
        title_html = f'''
                     <div style="position: fixed; 
                     top: 10px; left: 50px; width: 400px; height: 100px; 
                     background-color: white; border:2px solid grey; z-index:9999; 
                     font-size:14px; padding: 10px">
                     <b>{title}</b><br>
                     <span style="color:red;">●</span> Low penetration | 
                     <span style="color:yellow;">●</span> Medium | 
                     <span style="color:green;">●</span> High penetration
                     </div>
                     '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    def create_grid_capacity_map(self, capacity_data: pd.DataFrame,
                                title: str = "Grid Capacity by Country") -> folium.Map:
        """
        Create map with proportional circles showing grid capacity.
        
        Args:
            capacity_data: DataFrame with country, grid_capacity_mw
            title: Map title
        
        Returns:
            Folium map object
        """
        m = self.create_base_map()
        m = self.add_grid_capacity_circles(m, capacity_data)
        
        # Add title and legend
        title_html = f'''
                     <div style="position: fixed; 
                     top: 10px; left: 50px; width: 300px; height: 80px; 
                     background-color: white; border:2px solid grey; z-index:9999; 
                     font-size:14px; padding: 10px">
                     <b>{title}</b><br>
                     <span style="color:orange;">●</span> Larger circles = Higher capacity<br>
                     Size proportional to MW
                     </div>
                     '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        return m
    
    @staticmethod
    def _get_color_by_capacity(capacity: float) -> str:
        """Get color based on capacity value."""
        if capacity < 500:
            return '#1f77b4'  # Blue (low)
        elif capacity < 2000:
            return '#ff7f0e'  # Orange (medium)
        elif capacity < 5000:
            return '#d62728'  # Red (high)
        else:
            return '#9467bd'  # Purple (very high)
    
    @staticmethod
    def _get_color_by_volume(volume: float) -> str:
        """Get color based on trade volume."""
        if volume < 10:
            return '#2ca02c'  # Green (low)
        elif volume < 25:
            return '#ff7f0e'  # Orange (medium)
        elif volume < 50:
            return '#d62728'  # Red (high)
        else:
            return '#8b008b'  # Dark magenta (very high)


class RegionalMapBuilder:
    """Builds region-specific interconnection maps."""
    
    @staticmethod
    def create_south_asia_map(trade_data: pd.DataFrame = None) -> folium.Map:
        """
        Create South Asia cross-border energy map.
        
        Args:
            trade_data: Optional trade data for flow visualization
        
        Returns:
            Folium map object
        """
        mapper = InterconnectionMapper(
            zoom_level=5,
            center_lat=23,
            center_lon=82
        )
        
        # South Asian countries
        countries = ["India", "Nepal", "Bhutan", "Bangladesh", "Pakistan"]
        
        # Key interconnections in South Asia
        connections = [
            ("India", "Nepal", 600),
            ("India", "Bhutan", 300),
            ("India", "Bangladesh", 500),
            ("India", "Pakistan", 200),
        ]
        
        m = mapper.create_interconnection_network_map(countries, connections, 
                                                      title="South Asia Cross-Border Energy Network")
        
        if trade_data is not None:
            m = mapper.add_trade_flows(m, trade_data)
        
        return m
    
    @staticmethod
    def create_asean_map(trade_data: pd.DataFrame = None) -> folium.Map:
        """
        Create ASEAN cross-border energy map.
        
        Args:
            trade_data: Optional trade data for flow visualization
        
        Returns:
            Folium map object
        """
        mapper = InterconnectionMapper(
            zoom_level=5,
            center_lat=10,
            center_lon=105
        )
        
        # ASEAN countries
        countries = ["Thailand", "Laos", "Vietnam", "Malaysia", "Singapore", 
                    "Indonesia", "Philippines"]
        
        # Key interconnections in ASEAN
        connections = [
            ("Thailand", "Laos", 1500),
            ("Thailand", "Malaysia", 800),
            ("Malaysia", "Singapore", 600),
            ("Laos", "Vietnam", 500),
            ("Vietnam", "Thailand", 400),
        ]
        
        m = mapper.create_interconnection_network_map(countries, connections,
                                                      title="ASEAN Cross-Border Energy Network")
        
        if trade_data is not None:
            m = mapper.add_trade_flows(m, trade_data)
        
        return m
    
    @staticmethod
    def create_global_map(countries: List[str], connections: List[Tuple[str, str, float]]) -> folium.Map:
        """
        Create global energy interconnections map.
        
        Args:
            countries: List of countries to display
            connections: List of interconnections
        
        Returns:
            Folium map object
        """
        mapper = InterconnectionMapper(zoom_level=3, center_lat=20, center_lon=0)
        return mapper.create_interconnection_network_map(countries, connections,
                                                         title="Global Energy Interconnections Network")
