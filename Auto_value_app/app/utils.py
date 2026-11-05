# app/utils.py
import json
import pandas as pd

def load_dataset(file_path):
    """Loads the JSON dataset into a Pandas DataFrame."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def get_dropdown_options(df):
    """Extracts unique sorted lists for UI dropdowns using exact dataset column names."""
    options = {
        "makes": sorted(df['Make / Brand'].dropna().unique().tolist()) if 'Make / Brand' in df.columns else [],
        "models": sorted(df['Model'].dropna().unique().tolist()) if 'Model' in df.columns else [],
        "transmissions": sorted(df['Transmission'].dropna().unique().tolist()) if 'Transmission' in df.columns else [],
        "fuel_types": sorted(df['Fuel Type'].dropna().unique().tolist()) if 'Fuel Type' in df.columns else []
    }
    return options