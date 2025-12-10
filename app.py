# app.py - Main Streamlit Dashboard Application

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import sys
import os
from src.map_visualizations import InterconnectionMapper, RegionalMapBuilder
import streamlit_folium as stf


# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_collector import DataCollector
from src.web_crawler import WebCrawler
from src.cache_manager import CacheManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# PAGE CONFIGURATION
# ============================
st.set_page_config(
    page_title="Cross-Border Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# INITIALIZE SESSION STATE
# ============================
if 'data_collector' not in st.session_state:
    st.session_state.data_collector = DataCollector()

if 'web_crawler' not in st.session_state:
    st.session_state.web_crawler = WebCrawler()

if 'cache_manager' not in st.session_state:
    st.session_state.cache_manager = CacheManager()

# ============================
# SIDEBAR NAVIGATION
# ============================
st.sidebar.title("🌍 Navigation")
page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Home",
        "Bilateral Trade Analysis",
        "Technical Metrics",
        "Economic Indicators",
        "Energy Interconnections Map",
        "News & Updates",
        "Data Export",
        "Cache Status"
    ]
)

# ============================
# HOME PAGE
# ============================
if page == "Home":
    st.title("⚡ Global Cross-Border Energy Interconnections Dashboard")
    st.markdown("### Management Information System | International Solar Alliance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Countries Tracked", len(config.ISA_MEMBER_COUNTRIES))
    
    with col2:
        st.metric("Active Trade Corridors", len(config.BILATERAL_TRADE_PAIRS))
    
    with col3:
        st.metric("News Sources", len(config.NEWS_SOURCES))
    
    st.markdown("---")
    
    st.subheader("📊 Dashboard Overview")
    st.write("""
    This comprehensive Management Information System monitors and analyzes:
    
    1. **Bilateral Electricity Trade** - Country-to-country energy flows
    2. **Technical Metrics** - Grid capacity, renewable penetration, system reliability
    3. **Economic Indicators** - Trade values, pricing, investment trends
    4. **Real-Time News** - Global renewable energy and grid integration updates
    5. **Local Data Caching** - Fast loading with offline capability
    
    #### Key Features:
    - ✅ Interactive visualizations and maps
    - ✅ Real-time web crawler for news updates
    - ✅ Advanced filtering and analysis tools
    - ✅ Data export capabilities (CSV, Excel, PDF)
    - ✅ Sentiment analysis of energy news
    - ✅ Performance-optimized with local caching
    """)
    
    st.markdown("---")
    st.subheader("🎯 ISA Member Countries")
    
    # Display ISA countries
    countries_list = list(config.ISA_MEMBER_COUNTRIES.keys())
    col_width = 3
    cols = st.columns(col_width)
    
    for idx, country in enumerate(countries_list):
        with cols[idx % col_width]:
            st.write(f"🌐 {country}")

# ============================
# BILATERAL TRADE ANALYSIS
# ============================
elif page == "Bilateral Trade Analysis":
    st.title("📈 Bilateral Electricity Trade Visualization")
    
    # Load data with caching
    @st.cache_data(ttl=3600)
    def load_trade_data():
        return st.session_state.data_collector.get_bilateral_trade_data()
    
    trade_df = load_trade_data()
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    selected_exporters = st.sidebar.multiselect(
        "Select Exporters",
        sorted(trade_df['exporter'].unique()),
        default=sorted(trade_df['exporter'].unique())[:3]
    )
    
    if selected_exporters:
        trade_filtered = trade_df[trade_df['exporter'].isin(selected_exporters)]
    else:
        trade_filtered = trade_df
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=365), datetime.now())
    )
    
    trade_filtered['date'] = pd.to_datetime(trade_filtered['date'])
    
    if len(date_range) == 2:
        trade_filtered = trade_filtered[
            (trade_filtered['date'] >= pd.Timestamp(date_range[0])) &
            (trade_filtered['date'] <= pd.Timestamp(date_range[1]))
        ]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Trade Volume (TWh)",
            f"{trade_filtered['trade_volume_twh'].sum():.2f}"
        )
    
    with col2:
        st.metric(
            "Total Trade Value (USD Billion)",
            f"{trade_filtered['trade_value_usd_million'].sum() / 1000:.2f}"
        )
    
    with col3:
        st.metric(
            "Avg Renewable %",
            f"{trade_filtered['renewable_energy_pct'].mean():.1f}%"
        )
    
    with col4:
        st.metric(
            "Avg Transmission Loss",
            f"{trade_filtered['transmission_loss_pct'].mean():.2f}%"
        )
    
    st.markdown("---")
    
    # Trade volume trend
    st.subheader("Trade Volume Trend")
    trade_by_date = trade_filtered.groupby('date')['trade_volume_twh'].sum().reset_index()
    
    fig_trend = px.line(
        trade_by_date,
        x='date',
        y='trade_volume_twh',
        markers=True,
        labels={'trade_volume_twh': 'Volume (TWh)', 'date': 'Date'},
        title="Monthly Trade Volume Trend"
    )
    fig_trend.update_layout(hovermode='x unified')
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Trade corridor comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Trade Corridors by Volume")
        corridors = trade_filtered.copy()
        corridors['corridor'] = corridors['exporter'] + ' → ' + corridors['importer']
        corridor_volumes = corridors.groupby('corridor')['trade_volume_twh'].sum().nlargest(10)
        
        fig_corridors = px.bar(
            x=corridor_volumes.values,
            y=corridor_volumes.index,
            orientation='h',
            labels={'x': 'Trade Volume (TWh)', 'y': 'Corridor'}
        )
        st.plotly_chart(fig_corridors, use_container_width=True)
    
    with col2:
        st.subheader("Price vs Renewable Energy %")
        fig_scatter = px.scatter(
            trade_filtered,
            x='renewable_energy_pct',
            y='average_price_usd_mwh',
            size='trade_volume_twh',
            hover_data=['exporter', 'importer'],
            labels={
                'renewable_energy_pct': 'Renewable Energy %',
                'average_price_usd_mwh': 'Price (USD/MWh)'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Detailed data table
    st.subheader("📊 Detailed Trade Data")
    display_cols = [
        'date', 'exporter', 'importer', 'trade_volume_twh',
        'average_price_usd_mwh', 'renewable_energy_pct',
        'transmission_loss_pct'
    ]
    st.dataframe(
        trade_filtered[display_cols].sort_values('date', ascending=False),
        use_container_width=True
    )

# ============================
# TECHNICAL METRICS
# ============================
elif page == "Technical Metrics":
    st.title("🔧 Technical Metrics & Grid Performance")
    
    @st.cache_data(ttl=3600)
    def load_technical_data():
        return st.session_state.data_collector.get_technical_metrics()
    
    tech_df = load_technical_data()
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        sorted(tech_df['country'].unique()),
        default=sorted(tech_df['country'].unique())[:5]
    )
    
    if selected_countries:
        tech_filtered = tech_df[tech_df['country'].isin(selected_countries)]
    else:
        tech_filtered = tech_df
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Avg Grid Capacity (MW)",
            f"{tech_filtered['grid_capacity_mw'].mean():,.0f}"
        )
    
    with col2:
        st.metric(
            "Avg Renewable Penetration",
            f"{tech_filtered['renewable_penetration_pct'].mean():.1f}%"
        )
    
    with col3:
        st.metric(
            "Avg System Stability",
            f"{tech_filtered['system_stability_index'].mean():.3f}"
        )
    
    with col4:
        st.metric(
            "Avg Transmission Loss",
            f"{tech_filtered['transmission_losses_pct'].mean():.2f}%"
        )
    
    st.markdown("---")
    
    # Renewable penetration over time
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Renewable Energy Penetration Trend")
        renewable_trend = tech_filtered.groupby('date')['renewable_penetration_pct'].mean().reset_index()
        
        fig_renewable = px.line(
            renewable_trend,
            x='date',
            y='renewable_penetration_pct',
            markers=True,
            labels={
                'renewable_penetration_pct': 'Penetration (%)',
                'date': 'Date'
            }
        )
        st.plotly_chart(fig_renewable, use_container_width=True)
    
    with col2:
        st.subheader("Grid Capacity by Country")
        latest_capacity = tech_filtered.sort_values('date').drop_duplicates('country', keep='last')
        
        fig_capacity = px.bar(
            latest_capacity.sort_values('grid_capacity_mw', ascending=True),
            x='grid_capacity_mw',
            y='country',
            orientation='h',
            labels={'grid_capacity_mw': 'Capacity (MW)', 'country': 'Country'}
        )
        st.plotly_chart(fig_capacity, use_container_width=True)
    
    # System stability vs renewable penetration
    st.subheader("System Stability vs Renewable Penetration")
    fig_stability = px.scatter(
        tech_filtered,
        x='renewable_penetration_pct',
        y='system_stability_index',
        color='country',
        size='grid_capacity_mw',
        hover_data=['country', 'reserve_margin_pct'],
        labels={
            'renewable_penetration_pct': 'Renewable Penetration (%)',
            'system_stability_index': 'System Stability Index'
        }
    )
    st.plotly_chart(fig_stability, use_container_width=True)
    
    # Metrics table
    st.subheader("📊 Technical Metrics Data")
    display_cols = [
        'date', 'country', 'grid_capacity_mw', 'renewable_penetration_pct',
        'average_load_factor_pct', 'reserve_margin_pct', 'system_stability_index'
    ]
    st.dataframe(
        tech_filtered[display_cols].sort_values('date', ascending=False),
        use_container_width=True
    )

# ============================
# ECONOMIC INDICATORS
# ============================
elif page == "Economic Indicators":
    st.title("💰 Economic Indicators & Market Analysis")
    
    @st.cache_data(ttl=3600)
    def load_economic_data():
        return st.session_state.data_collector.get_economic_indicators()
    
    econ_df = load_economic_data()
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Filters")
    
    selected_econ_countries = st.sidebar.multiselect(
        "Select Countries",
        sorted(econ_df['country'].unique()),
        default=sorted(econ_df['country'].unique())[:5]
    )
    
    if selected_econ_countries:
        econ_filtered = econ_df[econ_df['country'].isin(selected_econ_countries)]
    else:
        econ_filtered = econ_df
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Avg Electricity Price (USD/MWh)",
            f"{econ_filtered['avg_electricity_price_usd_mwh'].mean():.2f}"
        )
    
    with col2:
        st.metric(
            "Total Trade Volume (TWh)",
            f"{econ_filtered['cross_border_trade_volume_twh'].sum():.2f}"
        )
    
    with col3:
        st.metric(
            "Avg Cost Savings %",
            f"{econ_filtered['cost_savings_from_trade_pct'].mean():.1f}%"
        )
    
    with col4:
        st.metric(
            "Total Export Revenue (USD Billion)",
            f"{econ_filtered['export_revenue_usd_million'].sum() / 1000:.2f}"
        )
    
    st.markdown("---")
    
    # Electricity price trend
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Electricity Price Trend")
        price_trend = econ_filtered.groupby('date')['avg_electricity_price_usd_mwh'].mean().reset_index()
        
        fig_price = px.area(
            price_trend,
            x='date',
            y='avg_electricity_price_usd_mwh',
            labels={
                'avg_electricity_price_usd_mwh': 'Price (USD/MWh)',
                'date': 'Date'
            }
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        st.subheader("Cost Savings by Country")
        latest_savings = econ_filtered.sort_values('date').drop_duplicates('country', keep='last')
        
        fig_savings = px.bar(
            latest_savings.sort_values('cost_savings_from_trade_pct', ascending=True),
            x='cost_savings_from_trade_pct',
            y='country',
            orientation='h',
            labels={
                'cost_savings_from_trade_pct': 'Cost Savings (%)',
                'country': 'Country'
            }
        )
        st.plotly_chart(fig_savings, use_container_width=True)
    
    # Grid investment analysis
    st.subheader("Grid Investment Trends")
    investment_by_date = econ_filtered.groupby('date')['grid_investment_usd_million'].sum().reset_index()
    
    fig_investment = px.bar(
        investment_by_date,
        x='date',
        y='grid_investment_usd_million',
        labels={
            'grid_investment_usd_million': 'Investment (USD Million)',
            'date': 'Date'
        }
    )
    st.plotly_chart(fig_investment, use_container_width=True)
    
    # Economic data table
    st.subheader("📊 Economic Indicators Data")
    display_cols = [
        'date', 'country', 'avg_electricity_price_usd_mwh',
        'cross_border_trade_volume_twh', 'cost_savings_from_trade_pct',
        'grid_investment_usd_million', 'export_revenue_usd_million'
    ]
    st.dataframe(
        econ_filtered[display_cols].sort_values('date', ascending=False),
        use_container_width=True
    )

# ============================
# ENERGY INTERCONNECTIONS MAP
# ============================
elif page == "Energy Interconnections Map":
    st.title("🗺️ Global Energy Interconnections Map")
    
    # Map type selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        map_type = st.selectbox(
            "Select Map Type",
            ["Global Network", "Trade Flows", "Renewable Penetration", "Grid Capacity", 
             "South Asia", "ASEAN"]
        )
    
    st.markdown("---")
    
    # Load data
    @st.cache_data(ttl=3600)
    def load_map_data():
        collector = st.session_state.data_collector
        return (
            collector.get_bilateral_trade_data(),
            collector.get_technical_metrics(),
            collector.get_economic_indicators()
        )
    
    trade_df, tech_df, econ_df = load_map_data()
    
    # Generate maps based on selection
    mapper = InterconnectionMapper()
    
    if map_type == "Global Network":
        st.subheader("Global Energy Interconnections Network")
        st.write("Shows all ISA member countries and major cross-border energy interconnections.")
        
        countries = list(config.ISA_MEMBER_COUNTRIES.keys())

        # Convert (exporter, importer) to (exporter, importer, capacity_mw)
        connections = [
            (exp, imp, 1000)   # use 1000 MW or any default you like
            for (exp, imp) in config.BILATERAL_TRADE_PAIRS
        ]

        m = mapper.create_interconnection_network_map(countries, connections)
        stf.folium_static(m, width=1200, height=700)
    
    elif map_type == "Trade Flows":
        st.subheader("Cross-Border Electricity Trade Flows")
        st.write("Visualizes bilateral electricity trade flows between countries. Line thickness represents trade volume.")
        
        # Get latest trade data
        latest_trade = trade_df.sort_values('date').drop_duplicates(['exporter', 'importer'], keep='last')
        
        m = mapper.create_trade_flow_map(latest_trade)
        stf.folium_static(m, width=1200, height=700)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Trade Corridors", len(latest_trade))
        with col2:
            st.metric("Total Volume (TWh)", f"{latest_trade['trade_volume_twh'].sum():.2f}")
        with col3:
            st.metric("Total Trade Value (USD B)", f"{latest_trade['trade_value_usd_million'].sum()/1000:.2f}")
    
    elif map_type == "Renewable Penetration":
        st.subheader("Renewable Energy Penetration Heatmap")
        st.write("Intensity of color shows renewable energy penetration percentage. Darker red = lower, green = higher penetration.")
        
        m = mapper.create_renewable_penetration_map(tech_df)
        stf.folium_static(m, width=1200, height=700)
        
        # Show details
        latest_tech = tech_df.sort_values('date').drop_duplicates('country', keep='last')
        top_renewable = latest_tech.nlargest(5, 'renewable_penetration_pct')[['country', 'renewable_penetration_pct']]
        
        st.subheader("Top 5 Countries by Renewable Penetration")
        for idx, row in top_renewable.iterrows():
            st.write(f"🌱 **{row['country']}**: {row['renewable_penetration_pct']:.1f}%")
    
    elif map_type == "Grid Capacity":
        st.subheader("Grid Capacity Distribution")
        st.write("Circle size represents grid capacity. Larger circles = higher capacity (MW).")
        
        m = mapper.create_grid_capacity_map(tech_df)
        stf.folium_static(m, width=1200, height=700)
        
        # Show top capacities
        latest_tech = tech_df.sort_values('date').drop_duplicates('country', keep='last')
        top_capacity = latest_tech.nlargest(5, 'grid_capacity_mw')[['country', 'grid_capacity_mw']]
        
        st.subheader("Top 5 Countries by Grid Capacity")
        for idx, row in top_capacity.iterrows():
            st.write(f"⚡ **{row['country']}**: {row['grid_capacity_mw']:,.0f} MW")
    
    elif map_type == "South Asia":
        st.subheader("South Asia Cross-Border Energy Network")
        st.write("Focus on India, Nepal, Bhutan, Bangladesh, and Pakistan interconnections.")
        
        south_asia_trade = trade_df[
            (trade_df['exporter'].isin(['India', 'Nepal', 'Bhutan', 'Bangladesh', 'Pakistan'])) &
            (trade_df['importer'].isin(['India', 'Nepal', 'Bhutan', 'Bangladesh', 'Pakistan']))
        ]
        
        m = RegionalMapBuilder.create_south_asia_map(south_asia_trade)
        stf.folium_static(m, width=1200, height=700)
        
        # Statistics
        if not south_asia_trade.empty:
            st.subheader("South Asia Trade Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Trade Corridors", len(south_asia_trade.drop_duplicates(['exporter', 'importer'])))
            with col2:
                st.metric("Total Volume (TWh)", f"{south_asia_trade['trade_volume_twh'].sum():.2f}")
    
    elif map_type == "ASEAN":
        st.subheader("ASEAN Cross-Border Energy Network")
        st.write("Focus on Southeast Asia interconnections including Thailand, Laos, Vietnam, Malaysia, Singapore, Indonesia, and Philippines.")
        
        asean_countries = ['Thailand', 'Laos', 'Vietnam', 'Malaysia', 'Singapore', 'Indonesia', 'Philippines']
        asean_trade = trade_df[
            (trade_df['exporter'].isin(asean_countries)) &
            (trade_df['importer'].isin(asean_countries))
        ]
        
        m = RegionalMapBuilder.create_asean_map(asean_trade)
        stf.folium_static(m, width=1200, height=700)
        
        # Statistics
        if not asean_trade.empty:
            st.subheader("ASEAN Trade Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Trade Corridors", len(asean_trade.drop_duplicates(['exporter', 'importer'])))
            with col2:
                st.metric("Total Volume (TWh)", f"{asean_trade['trade_volume_twh'].sum():.2f}")


# ============================
# NEWS & UPDATES
# ============================
elif page == "News & Updates":
    st.title("📰 Real-Time News & Updates")
    st.write("Latest news on renewable energy, grid integration, and cross-border interconnections")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        news_topic = st.text_input(
            "Search for specific topics",
            placeholder="e.g., solar, wind, interconnection, ISA..."
        )
    
    with col2:
        refresh_btn = st.button("🔄 Refresh News", key="refresh_news")
    
    try:
        # Load news with caching
        @st.cache_data(ttl=1800)
        def load_news():
            return st.session_state.web_crawler.crawl_all_sources()
        
        news_articles = load_news()
        
        if refresh_btn:
            st.cache_data.clear()
            news_articles = st.session_state.web_crawler.crawl_all_sources()
            st.success("News refreshed!")
        
        if news_topic:
            news_articles = [
                article for article in news_articles
                if news_topic.lower() in article['title'].lower() or
                   news_topic.lower() in article['summary'].lower()
            ]
        
        if news_articles:
            # Sentiment summary
            sentiment_summary = st.session_state.web_crawler.get_sentiment_summary(news_articles)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Articles", sentiment_summary['total'])
            with col2:
                st.metric("😊 Positive", sentiment_summary['positive'])
            with col3:
                st.metric("😐 Neutral", sentiment_summary['neutral'])
            with col4:
                st.metric("😞 Negative", sentiment_summary['negative'])
            
            st.markdown("---")
            
            # Display articles
            for idx, article in enumerate(news_articles[:20]):
                with st.container():
                    col1, col2 = st.columns([0.9, 0.1])
                    
                    with col1:
                        st.subheader(article['title'][:80] + "...")
                        st.write(f"📅 {article['date'][:10]} | 📍 {article['source']}")
                        st.write(article['summary'])
                        
                        if article['sentiment']:
                            sentiment_emoji = {
                                'positive': '😊',
                                'neutral': '😐',
                                'negative': '😞'
                            }
                            emoji = sentiment_emoji.get(article['sentiment']['sentiment'], '?')
                            st.caption(
                                f"{emoji} Sentiment: {article['sentiment']['sentiment'].upper()} "
                                f"(Polarity: {article['sentiment']['polarity']})"
                            )
                        
                        st.markdown(f"[Read Full Article →]({article['link']})")
                    
                    st.markdown("---")
        
        else:
            st.info("No articles found for your search. Try different keywords.")
    
    except Exception as e:
        st.error(f"Error loading news: {e}")
        st.info("News crawler may be temporarily unavailable. Please try again later.")

# ============================
# DATA EXPORT
# ============================
elif page == "Data Export":
    st.title("📥 Data Export & Download")
    
    st.subheader("Export Available Datasets")
    
    # Trade data export
    st.markdown("### 📊 Bilateral Trade Data")
    if st.button("Export Trade Data (CSV)", key="export_trade_csv"):
        trade_df = st.session_state.data_collector.get_bilateral_trade_data()
        csv = trade_df.to_csv(index=False)
        st.download_button(
            label="Download Trade Data (CSV)",
            data=csv,
            file_name=f"bilateral_trade_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Technical metrics export
    st.markdown("### 🔧 Technical Metrics")
    if st.button("Export Technical Data (CSV)", key="export_tech_csv"):
        tech_df = st.session_state.data_collector.get_technical_metrics()
        csv = tech_df.to_csv(index=False)
        st.download_button(
            label="Download Technical Metrics (CSV)",
            data=csv,
            file_name=f"technical_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Economic indicators export
    st.markdown("### 💰 Economic Indicators")
    if st.button("Export Economic Data (CSV)", key="export_econ_csv"):
        econ_df = st.session_state.data_collector.get_economic_indicators()
        csv = econ_df.to_csv(index=False)
        st.download_button(
            label="Download Economic Indicators (CSV)",
            data=csv,
            file_name=f"economic_indicators_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============================
# CACHE STATUS
# ============================
elif page == "Cache Status":
    st.title("💾 Cache Status & Management")
    
    cache_stats = st.session_state.cache_manager.get_cache_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Cache Entries", cache_stats.get('total_entries', 0))
    
    with col2:
        st.metric("Active Entries", cache_stats.get('active_entries', 0))
    
    with col3:
        st.metric("Expired Entries", cache_stats.get('expired_entries', 0))
    
    st.markdown("---")
    
    st.subheader("Cache Management")
    
    if st.button("🔄 Clear Expired Cache Entries"):
        cleared = st.session_state.cache_manager.clear_expired()
        st.success(f"Cleared {cleared} expired cache entries")
        st.rerun()
    
    if st.button("🗑️ Clear All Cache"):
        st.session_state.cache_manager.clear_expired()
        st.session_state.cache_manager = CacheManager()
        st.success("All cache cleared!")
        st.rerun()
    
    st.markdown("---")
    st.subheader("Cache Database Information")
    st.info(f"""
    **Database Path:** {cache_stats.get('cache_db_path', 'N/A')}
    
    **Cache TTL:** {config.CACHE_EXPIRY_HOURS} hours
    **Crawler Update Interval:** {config.CRAWLER_UPDATE_INTERVAL_MINUTES} minutes
    """)

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: small;'>
    <p>⚡ Cross-Border Energy Interconnections Dashboard v{version}</p>
    <p>Cornerstone Project | International Solar Alliance | Renewable Energy Technologies & Management</p>
    <p>Winter Break 2025</p>
</div>
""".format(version=config.APP_VERSION), unsafe_allow_html=True)
