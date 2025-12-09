"""
ISA MIS: Integrated Solar Energy Global Dashboard
v3.0 - LOCAL DATA STORAGE OPTIMIZATION

New in v3.0:
- Downloads and caches OWID & World Bank data as local CSV files
- Fast startup (loads from local files instead of APIs)
- Optional: Download latest data with one command
- NASA solar data still fetches real-time (stays current)
- Google News still fetches real-time (stays current)
- Setup: Run download_data.py once to populate local cache
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import logging
from pathlib import Path

# ============================================
# LOGGING CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# LOCAL DATA STORAGE CONFIGURATION
# ============================================

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

OWID_FILE = DATA_DIR / "owid_energy_data.csv"
WB_FILE = DATA_DIR / "world_bank_indicators.csv"

# API Configuration
NASA_API_TIMEOUT = 15
NEWS_API_TIMEOUT = 8
MAX_NEWS_ITEMS = 5

# Coordinates for ISA Nations (Lat/Lon for NASA API)
COUNTRY_COORDS = {
    'IND': (28.61, 77.20),    # India - New Delhi
    'FRA': (48.85, 2.35),     # France - Paris
    'AUS': (-35.28, 149.13),  # Australia - Sydney
    'BRA': (-15.78, -47.92),  # Brazil - Brasília
    'JPN': (35.68, 139.69),   # Japan - Tokyo
    'CHL': (-33.44, -70.66),  # Chile - Santiago
    'EGY': (30.04, 31.23),    # Egypt - Cairo
    'NGA': (9.07, 7.39),      # Nigeria - Abuja
    'ARE': (24.45, 54.37),    # UAE - Dubai
    'GBR': (51.50, -0.12)     # UK - London
}

# World Bank Indicators
WB_INDICATORS = {
    'NY.GDP.MKTP.CD': 'GDP_USD',
    'EG.ELC.ACCS.ZS': 'Access_Electricity_Pct',
    'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
}

# Page Configuration
st.set_page_config(
    page_title="ISA MIS: Integrated Global Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "ISA Member Information System v3.0 - Global Energy Dashboard (Local Data)"
    }
)

# ============================================
# DOWNLOAD & UPDATE DATA FUNCTIONS
# ============================================

def download_owid_data():
    """Download OWID energy data and save locally."""
    try:
        logger.info("Downloading OWID energy data...")
        url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
        df = pd.read_csv(url, low_memory=False)
        
        # Select required columns
        cols = ['iso_code', 'country', 'year',
                'solar_share_elec', 'wind_share_elec', 'hydro_share_elec',
                'nuclear_share_elec', 'fossil_share_elec', 'biofuel_share_elec',
                'carbon_intensity_elec', 'per_capita_electricity',
                'primary_energy_consumption']
        
        available_cols = [col for col in cols if col in df.columns]
        df = df[available_cols].copy()
        
        # Save to local file
        df.to_csv(OWID_FILE, index=False)
        logger.info(f"✅ OWID data saved: {len(df)} rows to {OWID_FILE}")
        return True
    except Exception as e:
        logger.error(f"❌ OWID download failed: {str(e)}")
        return False

def download_world_bank_data():
    """Download World Bank indicators and save locally."""
    try:
        logger.info("Downloading World Bank data...")
        import wbgapi as wb
        
        df_wb = wb.data.DataFrame(
            WB_INDICATORS,
            time=range(2010, 2025),
            skipBlanks=False,
            columns='series'
        ).reset_index()
        
        if df_wb.empty:
            logger.warning("World Bank returned empty DataFrame")
            return False
        
        # Rename columns
        df_wb.rename(columns={'economy': 'iso_code', 'time': 'year'}, inplace=True)
        
        # Clean year column
        if 'year' in df_wb.columns:
            df_wb['year'] = df_wb['year'].astype(str).str.replace('YR', '')
            df_wb['year'] = pd.to_numeric(df_wb['year'], errors='coerce')
            df_wb = df_wb.dropna(subset=['year'])
            df_wb['year'] = df_wb['year'].astype(int)
        
        # Apply column rename mapping from indicator codes to friendly names
        rename_map = {
            'NY.GDP.MKTP.CD': 'GDP_USD',
            'EG.ELC.ACCS.ZS': 'Access_Electricity_Pct',
            'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
        }
        df_wb.rename(columns=rename_map, inplace=True)
        
        # Convert numeric columns
        numeric_cols = [col for col in df_wb.columns if col not in ['iso_code', 'year']]
        for col in numeric_cols:
            df_wb[col] = pd.to_numeric(df_wb[col], errors='coerce')
        
        # Save to local file
        df_wb.to_csv(WB_FILE, index=False)
        logger.info(f"✅ World Bank data saved: {len(df_wb)} rows to {WB_FILE}")
        return True
    except Exception as e:
        logger.error(f"❌ World Bank download failed: {str(e)}")
        return False

# ============================================
# LIVE CRAWLER FUNCTIONS
# ============================================

@st.cache_data(ttl=86400)  # 24 hours
def crawl_nasa_solar_data(iso_code):
    """Fetches daily solar irradiance from NASA POWER API."""
    try:
        lat, lon = COUNTRY_COORDS.get(iso_code, (0, 0))
        
        if lat == 0 and lon == 0:
            logger.warning(f"Invalid coordinates for {iso_code}")
            return pd.DataFrame(columns=['Date', 'Irradiance'])
        
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            'parameters': 'ALLSKY_SFC_SW_DWN',
            'community': 'RE',
            'longitude': lon,
            'latitude': lat,
            'start': (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
            'end': datetime.now().strftime('%Y%m%d'),
            'format': 'JSON'
        }
        
        response = requests.get(url, params=params, timeout=NASA_API_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        if 'properties' not in data or 'parameter' not in data['properties']:
            logger.warning(f"Unexpected API response structure for {iso_code}")
            return pd.DataFrame(columns=['Date', 'Irradiance'])
        
        solar_dict = data['properties']['parameter'].get('ALLSKY_SFC_SW_DWN', {})
        
        if not solar_dict:
            logger.warning(f"No solar data available for {iso_code}")
            return pd.DataFrame(columns=['Date', 'Irradiance'])
        
        df_solar = pd.DataFrame(list(solar_dict.items()), columns=['Date', 'Irradiance'])
        df_solar['Date'] = pd.to_datetime(df_solar['Date'], format='%Y%m%d')
        df_solar['Irradiance'] = pd.to_numeric(df_solar['Irradiance'], errors='coerce')
        df_solar = df_solar[df_solar['Irradiance'] > 0].sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Successfully fetched {len(df_solar)} solar data points for {iso_code}")
        return df_solar
        
    except requests.exceptions.RequestException as e:
        logger.error(f"NASA API request failed for {iso_code}: {str(e)}")
        return pd.DataFrame(columns=['Date', 'Irradiance'])
    except (KeyError, ValueError) as e:
        logger.error(f"Data parsing error for {iso_code}: {str(e)}")
        return pd.DataFrame(columns=['Date', 'Irradiance'])


@st.cache_data(ttl=21600)  # 6 hours
def crawl_energy_news(country_name):
    """Crawls Google News RSS for latest energy headlines."""
    try:
        query = f"Energy {country_name}"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        response = requests.get(url, timeout=NEWS_API_TIMEOUT, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        news_items = []
        
        for item in root.findall('./channel/item')[:MAX_NEWS_ITEMS]:
            title_elem = item.find('title')
            link_elem = item.find('link')
            pubdate_elem = item.find('pubDate')
            
            if title_elem is not None and link_elem is not None:
                news = {
                    'title': title_elem.text or 'No title',
                    'link': link_elem.text or '#',
                    'pubDate': pubdate_elem.text if pubdate_elem is not None else 'Unknown date'
                }
                news_items.append(news)
        
        logger.info(f"Successfully fetched {len(news_items)} news items for {country_name}")
        return news_items
        
    except requests.exceptions.RequestException as e:
        logger.error(f"News fetch failed for {country_name}: {str(e)}")
        return []
    except ET.ParseError as e:
        logger.error(f"XML parsing error: {str(e)}")
        return []

# ============================================
# LOAD LOCAL DATA
# ============================================

def load_local_data():
    """Load data from local CSV files."""
    error_messages = []
    df_owid = None
    df_wb = None
    
    # Load OWID
    try:
        if OWID_FILE.exists():
            logger.info(f"Loading OWID from {OWID_FILE}...")
            df_owid = pd.read_csv(OWID_FILE)
            logger.info(f"✅ OWID loaded: {len(df_owid)} rows")
        else:
            logger.warning(f"OWID file not found: {OWID_FILE}")
            error_messages.append(f"OWID file not found. Run: python download_data.py")
    except Exception as e:
        logger.error(f"Error loading OWID: {str(e)}")
        error_messages.append(f"OWID loading error: {str(e)}")
    
    # Load World Bank
    try:
        if WB_FILE.exists():
            logger.info(f"Loading World Bank from {WB_FILE}...")
            df_wb = pd.read_csv(WB_FILE)
            logger.info(f"✅ World Bank loaded: {len(df_wb)} rows")
        else:
            logger.warning(f"World Bank file not found: {WB_FILE}")
            error_messages.append(f"World Bank file not found. Run: python download_data.py")
    except Exception as e:
        logger.error(f"Error loading World Bank: {str(e)}")
        error_messages.append(f"World Bank loading error: {str(e)}")
    
    # Merge if both available
    try:
        if df_owid is not None and df_wb is not None:
            logger.info("Merging OWID and World Bank data...")
            df_merged = pd.merge(df_owid, df_wb, on=['iso_code', 'year'], how='outer')
            logger.info(f"✅ Merged: {len(df_merged)} rows")
        elif df_owid is not None:
            logger.warning("Using OWID only")
            df_merged = df_owid.copy()
            error_messages.append("World Bank data unavailable; using OWID only")
        elif df_wb is not None:
            logger.warning("Using World Bank only")
            df_merged = df_wb.copy()
            error_messages.append("OWID data unavailable; using World Bank only")
        else:
            logger.error("No data sources available!")
            st.error("❌ FATAL: No local data files found.")
            st.error("Setup: Run `python download_data.py` to download data files first.")
            st.stop()
        
        # Ensure required columns exist
        for col in list(WB_INDICATORS.values()):
            if col not in df_merged.columns:
                df_merged[col] = np.nan
        
        # Filter for ISA nations
        isa_codes = ['IND', 'FRA', 'AUS', 'BRA', 'JPN', 'CHL', 'EGY', 'NGA', 'ARE', 'GBR']
        df_final = df_merged[df_merged['iso_code'].isin(isa_codes)].copy()
        
        if df_final.empty:
            st.error("❌ No data found for ISA countries")
            st.stop()
        
        # Calculate renewable share
        renewable_cols = ['solar_share_elec', 'wind_share_elec', 'hydro_share_elec', 'biofuel_share_elec']
        renewable_cols = [col for col in renewable_cols if col in df_final.columns]
        
        if renewable_cols:
            df_final['total_renewable_share'] = df_final[renewable_cols].sum(axis=1)
            df_final['total_renewable_share'] = df_final['total_renewable_share'].replace(0, np.nan)
        else:
            df_final['total_renewable_share'] = np.nan
        
        logger.info(f"✅ Data ready for {len(isa_codes)} ISA countries")
        
        return df_final, isa_codes, error_messages
        
    except Exception as e:
        logger.exception(f"Merge/processing failed: {str(e)}")
        st.error(f"❌ Data processing error: {str(e)}")
        st.stop()

# ============================================
# INITIALIZE
# ============================================

try:
    df, isa_codes, data_warnings = load_local_data()
    
    if df.empty:
        st.error("⚠️ No data available.")
        st.stop()
    
    iso_to_name = df[['iso_code', 'country']].drop_duplicates().set_index('iso_code')['country'].to_dict()
    
    # Show warnings if any
    if data_warnings:
        with st.expander("⚠️ Setup Required"):
            st.warning("Local data files not found. To set up:")
            st.code("""
# Create file: download_data.py
import subprocess
subprocess.run(['python', 'download_data.py'])

# Or run in terminal:
python download_data.py
            """)
            for warning in data_warnings:
                st.info(warning)

except Exception as e:
    logger.exception(f"FATAL: {str(e)}")
    st.error(f"❌ Failed to initialize: {str(e)}")
    st.stop()

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    selected_iso = st.selectbox(
        "Select ISA Member Nation",
        isa_codes,
        format_func=lambda x: iso_to_name.get(x, x),
        help="Choose a country to analyze"
    )
    
    country_name = iso_to_name.get(selected_iso, selected_iso)
    min_year, max_year = int(df['year'].min()), int(df['year'].max())
    
    selected_year_range = st.slider(
        "Historical Analysis Period",
        min_year, max_year, (2015, max_year),
        help="Select years to analyze"
    )
    
    st.divider()
    st.subheader("📊 Display Options")
    chart_style = st.radio("Chart Theme", ["Default", "Dark", "Light"], horizontal=True)
    
    # Data management
    st.divider()
    st.subheader("🔄 Data Management")
    
    if st.button("📥 Download Latest Data"):
        st.info("Downloading latest data (this may take a minute)...")
        owid_ok = download_owid_data()
        wb_ok = download_world_bank_data()
        
        if owid_ok and wb_ok:
            st.success("✅ Data updated! Refresh page to reload.")
            st.balloons()
        else:
            st.error("❌ Some downloads failed. Check terminal for errors.")
    
    st.divider()
    st.caption(f"""
📁 Data Location: `./data/`
    
- `owid_energy_data.csv` (~100 MB)
- `world_bank_indicators.csv` (~10 MB)

🚀 Fast load from disk instead of APIs!
    """)
    
    # Filter data
    mask = ((df['iso_code'] == selected_iso) &
            (df['year'] >= selected_year_range[0]) &
            (df['year'] <= selected_year_range[1]))
    
    country_df = df[mask].copy()
    
    if country_df.empty:
        st.warning(f"No data available for {country_name} in {selected_year_range[0]}-{selected_year_range[1]}")
        st.stop()
    
    latest_data = country_df.iloc[-1]
    first_data = country_df.iloc[0]

# ============================================
# MAIN DASHBOARD
# ============================================

st.title(f"🌍 ISA MIS: {country_name} Energy Dashboard")
st.markdown("### Complete Energy Mix Analysis & Economic Impact (v3.0 - Local Data)")
st.markdown(f"*Last updated: {datetime.now().strftime('%B %d, %Y at %H:%M')} UTC*")

# KEY METRICS
st.subheader("📊 Key Energy Indicators")

metric_cols = st.columns(5)

with metric_cols[0]:
    total_renewable = latest_data.get('total_renewable_share', np.nan)
    renewable_delta = total_renewable - first_data.get('total_renewable_share', np.nan) if pd.notna(total_renewable) and pd.notna(first_data.get('total_renewable_share')) else None
    st.metric(
        "Total Renewable %",
        f"{total_renewable:.1f}%" if pd.notna(total_renewable) else "N/A",
        delta=f"{renewable_delta:.1f}%" if renewable_delta is not None else None
    )

with metric_cols[1]:
    fossil_pct = latest_data.get('fossil_share_elec', np.nan)
    fossil_delta = fossil_pct - first_data.get('fossil_share_elec', np.nan) if pd.notna(fossil_pct) and pd.notna(first_data.get('fossil_share_elec')) else None
    st.metric(
        "Fossil Fuel %",
        f"{fossil_pct:.1f}%" if pd.notna(fossil_pct) else "N/A",
        delta=f"{fossil_delta:.1f}%" if fossil_delta is not None else None,
        delta_color="inverse"
    )

with metric_cols[2]:
    nuclear_pct = latest_data.get('nuclear_share_elec', np.nan)
    st.metric(
        "Nuclear %",
        f"{nuclear_pct:.1f}%" if pd.notna(nuclear_pct) else "N/A"
    )

with metric_cols[3]:
    carbon_intensity = latest_data.get('carbon_intensity_elec', np.nan)
    carbon_delta = first_data.get('carbon_intensity_elec', np.nan) - carbon_intensity if pd.notna(carbon_intensity) and pd.notna(first_data.get('carbon_intensity_elec')) else None
    st.metric(
        "Carbon Intensity",
        f"{carbon_intensity:.0f} gCO₂/kWh" if pd.notna(carbon_intensity) else "N/A",
        delta=f"{carbon_delta:.0f}" if carbon_delta is not None else None,
        delta_color="inverse"
    )

with metric_cols[4]:
    access_pct = latest_data.get('Access_Electricity_Pct', np.nan)
    st.metric(
        "Electricity Access",
        f"{access_pct:.1f}%" if pd.notna(access_pct) else "N/A",
        help="Population with access"
    )

st.divider()

# TABS
tab_energy_mix, tab_renewable, tab_economic, tab_live = st.tabs([
    "⚡ Complete Energy Mix",
    "♻️ Renewable Focus",
    "💰 Economic Impact",
    "☀️ Live Intelligence"
])

# --- TAB 1: COMPLETE ENERGY MIX ---
with tab_energy_mix:
    st.subheader(f"All Energy Sources - {country_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Energy Mix Evolution (Stacked Area)")
        
        energy_data = country_df[['year', 'solar_share_elec', 'wind_share_elec',
                                   'hydro_share_elec', 'nuclear_share_elec',
                                   'fossil_share_elec']].copy()
        energy_data = energy_data.fillna(0)
        
        if not energy_data.empty and energy_data[['solar_share_elec', 'wind_share_elec', 'hydro_share_elec', 'nuclear_share_elec', 'fossil_share_elec']].sum().sum() > 0:
            fig_stacked = go.Figure()
            
            fig_stacked.add_trace(go.Scatter(
                x=energy_data['year'], y=energy_data['solar_share_elec'],
                mode='lines', name='Solar', stackgroup='one', fillcolor='rgba(255, 193, 7, 0.8)'
            ))
            fig_stacked.add_trace(go.Scatter(
                x=energy_data['year'], y=energy_data['wind_share_elec'],
                mode='lines', name='Wind', stackgroup='one', fillcolor='rgba(76, 175, 80, 0.8)'
            ))
            fig_stacked.add_trace(go.Scatter(
                x=energy_data['year'], y=energy_data['hydro_share_elec'],
                mode='lines', name='Hydropower', stackgroup='one', fillcolor='rgba(33, 150, 243, 0.8)'
            ))
            fig_stacked.add_trace(go.Scatter(
                x=energy_data['year'], y=energy_data['nuclear_share_elec'],
                mode='lines', name='Nuclear', stackgroup='one', fillcolor='rgba(255, 87, 34, 0.8)'
            ))
            fig_stacked.add_trace(go.Scatter(
                x=energy_data['year'], y=energy_data['fossil_share_elec'],
                mode='lines', name='Fossil Fuels', stackgroup='one', fillcolor='rgba(158, 158, 158, 0.8)'
            ))
            
            template_map = {
                "Default": "plotly",
                "Dark": "plotly_dark",
                "Light": "plotly_white",  # note: 'plotly_white', not 'plotly_light'
            }
            
            fig_stacked.update_layout(
                title="Energy Mix Evolution (Stacked %)",
                xaxis_title="Year",
                yaxis_title="Share of Electricity (%)",
                hovermode='x unified',
                template=template_map.get(chart_style, "plotly"),
                height=500
            )
            
            st.plotly_chart(fig_stacked, use_container_width=True)
        else:
            st.info("No energy mix data available")
    
    with col2:
        st.markdown("#### Individual Energy Sources")
        
        fig_multi = go.Figure()
        
        sources = [
            ('solar_share_elec', 'Solar', 'orange'),
            ('wind_share_elec', 'Wind', 'green'),
            ('hydro_share_elec', 'Hydropower', 'blue'),
            ('nuclear_share_elec', 'Nuclear', 'red'),
            ('fossil_share_elec', 'Fossil Fuels', 'gray')
        ]
        
        for col_name, label, color in sources:
            if col_name in energy_data.columns:
                fig_multi.add_trace(go.Scatter(
                    x=energy_data['year'], y=energy_data[col_name],
                    mode='lines+markers', name=label, line=dict(color=color, width=2)
                ))
        
        fig_multi.update_layout(
            title="Energy Sources Comparison",
            xaxis_title="Year",
            yaxis_title="Share of Electricity (%)",
            hovermode='x unified',
            template=template_map.get(chart_style, "plotly"),
            height=500
        )
        
        st.plotly_chart(fig_multi, use_container_width=True)

# --- TAB 2: RENEWABLE FOCUS ---
with tab_renewable:
    st.subheader(f"Renewable Energy Analysis - {country_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Renewable Energy Growth")
        
        renewable_data = country_df[['year', 'solar_share_elec', 'wind_share_elec',
                                       'hydro_share_elec', 'biofuel_share_elec',
                                       'total_renewable_share']].copy()
        renewable_data = renewable_data.fillna(0)
        
        if not renewable_data.empty:
            fig_renewable = go.Figure()
            
            fig_renewable.add_trace(go.Scatter(
                x=renewable_data['year'], y=renewable_data['total_renewable_share'],
                mode='lines+markers', name='Total Renewable', line=dict(color='green', width=3),
                fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.2)'
            ))
            
            fig_renewable.update_layout(
                title="Total Renewable Energy Share",
                xaxis_title="Year",
                yaxis_title="Share (%)",
                template=template_map.get(chart_style, "plotly"),
                height=450
            )
            
            st.plotly_chart(fig_renewable, use_container_width=True)
    
    with col2:
        st.markdown("#### Renewable Composition")
        
        latest_renewable = country_df.iloc[-1]
        
        renewable_sources = {
            'Solar': latest_renewable.get('solar_share_elec', 0),
            'Wind': latest_renewable.get('wind_share_elec', 0),
            'Hydropower': latest_renewable.get('hydro_share_elec', 0),
            'Biofuel': latest_renewable.get('biofuel_share_elec', 0)
        }
        
        renewable_sources = {k: v for k, v in renewable_sources.items() if v > 0}
        
        if renewable_sources:
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(renewable_sources.keys()),
                values=list(renewable_sources.values()),
                marker=dict(colors=['orange', 'green', 'blue', 'brown'])
            )])
            
            fig_pie.update_layout(
                title=f"Renewable Mix ({selected_year_range[1]})",
                height=450
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No renewable data available")

# --- TAB 3: ECONOMIC IMPACT ---
with tab_economic:
    st.subheader(f"Economic & Energy Access Indicators - {country_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### GDP Growth Trend")
        
        gdp_data = country_df[['year', 'GDP_USD']].copy()
        gdp_data_clean = gdp_data.dropna(subset=['GDP_USD'])
        
        if not gdp_data_clean.empty and gdp_data_clean['GDP_USD'].sum() > 0:
            fig_gdp = px.line(
                gdp_data_clean, x='year', y='GDP_USD',
                title='GDP (Current US$)',
                markers=True
            )
            
            fig_gdp.update_yaxes(tickformat='$,.0f')
            fig_gdp.update_layout(height=400, template=template_map.get(chart_style, "plotly"))
            
            st.plotly_chart(fig_gdp, use_container_width=True)
        else:
            st.info("💡 GDP data not available")
    
    with col2:
        st.markdown("#### Electricity Access Progress")
        
        access_data = country_df[['year', 'Access_Electricity_Pct']].copy()
        access_data_clean = access_data.dropna(subset=['Access_Electricity_Pct'])
        
        if not access_data_clean.empty and access_data_clean['Access_Electricity_Pct'].sum() > 0:
            fig_access = px.bar(
                access_data_clean, x='year', y='Access_Electricity_Pct',
                title='Population with Electricity Access',
                color='Access_Electricity_Pct',
                color_continuous_scale='Greens'
            )
            
            fig_access.update_layout(height=400, template=template_map.get(chart_style, "plotly"))
            
            st.plotly_chart(fig_access, use_container_width=True)
        else:
            st.info("💡 Electricity access data not available")
    
    # Economic table
    st.markdown("#### Economic Indicators Summary")
    
    econ_data = country_df[['year', 'GDP_USD', 'GDP_Per_Capita', 'Access_Electricity_Pct']].copy()
    econ_data.columns = ['Year', 'GDP (USD)', 'GDP Per Capita (USD)', 'Electricity Access %']
    
    def format_currency(x):
        if pd.isna(x):
            return "N/A"
        if x > 1e12:
            return f"${x/1e12:.2f}T"
        elif x > 1e9:
            return f"${x/1e9:.2f}B"
        elif x > 1e6:
            return f"${x/1e6:.2f}M"
        else:
            return f"${x:,.0f}"
    
    econ_display = econ_data.copy()
    econ_display['GDP (USD)'] = econ_data['GDP (USD)'].apply(format_currency)
    econ_display['GDP Per Capita (USD)'] = econ_data['GDP Per Capita (USD)'].apply(format_currency)
    econ_display['Electricity Access %'] = econ_data['Electricity Access %'].apply(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A"
    )
    
    st.dataframe(econ_display, use_container_width=True, hide_index=True)

# --- TAB 4: LIVE INTELLIGENCE ---
with tab_live:
    st.subheader("Real-Time Solar Data & Latest News")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📡 Current Solar Irradiance (Last 30 Days)")
        
        solar_df = crawl_nasa_solar_data(selected_iso)
        
        if not solar_df.empty:
            fig_solar = px.line(
                solar_df, x='Date', y='Irradiance',
                title=f"Solar Irradiance - {country_name}",
                markers=True
            )
            
            fig_solar.update_traces(fill='tozeroy')
            fig_solar.update_layout(height=400, template=template_map.get(chart_style, "plotly"))
            
            st.plotly_chart(fig_solar, use_container_width=True)
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Avg Daily", f"{solar_df['Irradiance'].mean():.2f} kWh/m²")
            with col_stat2:
                st.metric("Peak", f"{solar_df['Irradiance'].max():.2f} kWh/m²")
            with col_stat3:
                st.metric("Min", f"{solar_df['Irradiance'].min():.2f} kWh/m²")
            with col_stat4:
                st.metric("Std Dev", f"{solar_df['Irradiance'].std():.2f}")
        else:
            st.warning("⚠️ No solar data available")
    
    with col2:
        st.markdown("#### 📰 Latest Energy News")
        
        news_items = crawl_energy_news(country_name)
        
        if news_items:
            for idx, news in enumerate(news_items, 1):
                with st.expander(f"📑 {idx}. {news['title'][:60]}..."):
                    st.write(f"**Published:** {news['pubDate']}")
                    st.markdown(f"[Read →]({news['link']})")
        else:
            st.info("No news available")

# FOOTER
st.divider()

st.markdown(f"""
---

**ISA Member Information System (MIS) v3.0 - Local Data Edition**

📊 Data Sources:
- 📁 Local CSV files (OWID + World Bank) - **FAST** ⚡
- 🔬 NASA POWER API - Real-time solar irradiance
- 📰 Google News RSS - Latest energy news

**v3.0 Features:**
- ✅ Data stored locally in `./data/` folder
- ✅ Instant startup (no API waits)
- ✅ One-click data refresh button
- ✅ Reduces API dependency & costs
- ✅ Works offline (except NASA & News)

📁 **Local Data:** {OWID_FILE.stat().st_size / 1e6:.1f} MB + {WB_FILE.stat().st_size / 1e6:.1f} MB

""")

st.caption("© 2024 International Solar Alliance. All rights reserved.")
