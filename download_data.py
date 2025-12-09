#!/usr/bin/env python3
"""
ISA MIS Data Downloader
Downloads and saves OWID & World Bank data locally for fast dashboard access

Usage:
    python download_data.py

Output:
    ./data/owid_energy_data.csv         (~100 MB)
    ./data/world_bank_indicators.csv    (~10 MB)
"""

import pandas as pd
import wbgapi as wb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

OWID_FILE = DATA_DIR / "owid_energy_data.csv"
WB_FILE = DATA_DIR / "world_bank_indicators.csv"

WB_INDICATORS = {
    'NY.GDP.MKTP.CD': 'GDP_USD',
    'EG.ELC.ACCS.ZS': 'Access_Electricity_Pct',
    'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
}

def download_owid():
    """Download OWID energy data."""
    try:
        logger.info("📥 Downloading OWID energy data...")
        print("   This may take 1-2 minutes (large CSV file)...\n")
        
        url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
        df = pd.read_csv(url, low_memory=False)
        
        cols = ['iso_code', 'country', 'year',
                'solar_share_elec', 'wind_share_elec', 'hydro_share_elec',
                'nuclear_share_elec', 'fossil_share_elec', 'biofuel_share_elec',
                'carbon_intensity_elec', 'per_capita_electricity',
                'primary_energy_consumption']
        
        available_cols = [col for col in cols if col in df.columns]
        df = df[available_cols].copy()
        
        df.to_csv(OWID_FILE, index=False)
        
        logger.info(f"✅ OWID data saved: {len(df)} rows to {OWID_FILE}")
        print(f"   📁 File size: {OWID_FILE.stat().st_size / 1e6:.1f} MB\n")
        return True
    except Exception as e:
        logger.error(f"❌ OWID download failed: {str(e)}")
        return False

def download_world_bank():
    """Download World Bank indicators."""
    try:
        logger.info("📥 Downloading World Bank data...")
        print("   Fetching GDP, electricity access, GDP per capita...\n")
        
        df_wb = wb.data.DataFrame(
            WB_INDICATORS,
            time=range(2010, 2025),
            skipBlanks=False,
            columns='series'
        ).reset_index()
        
        if df_wb.empty:
            logger.warning("World Bank returned empty data")
            return False
        
        # Rename columns
        df_wb.rename(columns={'economy': 'iso_code', 'time': 'year'}, inplace=True)
        
        # Clean year
        if 'year' in df_wb.columns:
            df_wb['year'] = df_wb['year'].astype(str).str.replace('YR', '')
            df_wb['year'] = pd.to_numeric(df_wb['year'], errors='coerce')
            df_wb = df_wb.dropna(subset=['year'])
            df_wb['year'] = df_wb['year'].astype(int)
        
        # Apply rename map
        rename_map = {
            'NY.GDP.MKTP.CD': 'GDP_USD',
            'EG.ELC.ACCS.ZS': 'Access_Electricity_Pct',
            'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
        }
        df_wb.rename(columns=rename_map, inplace=True)
        
        # Convert to numeric
        numeric_cols = [col for col in df_wb.columns if col not in ['iso_code', 'year']]
        for col in numeric_cols:
            df_wb[col] = pd.to_numeric(df_wb[col], errors='coerce')
        
        df_wb.to_csv(WB_FILE, index=False)
        
        logger.info(f"✅ World Bank data saved: {len(df_wb)} rows for {df_wb['iso_code'].nunique()} countries to {WB_FILE}")
        print(f"   📁 File size: {WB_FILE.stat().st_size / 1e6:.1f} MB\n")
        return True
    except Exception as e:
        logger.error(f"❌ World Bank download failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  ISA MIS Data Downloader")
    print("="*60 + "\n")
    
    owid_ok = download_owid()
    wb_ok = download_world_bank()
    
    print("="*60)
    if owid_ok and wb_ok:
        print("\n✅ SUCCESS! Data downloaded and saved.")
        print(f"📁 Location: {DATA_DIR.absolute()}/\n")
        print("You can now run the dashboard:")
        print("   streamlit run app_v3_0_local_data.py\n")
        print("="*60 + "\n")
    else:
        print("\n❌ FAILED! Some downloads didn't complete.")
        print("Check the errors above and try again.\n")
        print("="*60 + "\n")
