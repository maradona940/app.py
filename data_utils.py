import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import Tuple, List

def load_monthly_crime_data(data_dir) -> Tuple[gpd.GeoDataFrame, List[str]]:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        # Standardize column names
        df.columns = [c.strip().replace(' ', '_').replace('-', '_').lower() for c in df.columns]
        # Map CSV columns to standardized filter columns
        if 'location' in df.columns:
            df['street_name'] = df['location']
        else:
            df['street_name'] = 'Unknown'
        if 'month' in df.columns:
            df['date'] = pd.to_datetime(df['month'], format='%Y-%m', errors='coerce').dt.date
        else:
            df['date'] = pd.NaT
        if 'lsoa_name' in df.columns:
            df['region'] = df['lsoa_name']
        elif 'lsoa_code' in df.columns:
            df['region'] = df['lsoa_code']
        else:
            df['region'] = 'Unknown'
        if 'crime_type' not in df.columns and 'crime_type' in df:
            df['crime_type'] = df['crime_type']
        if 'last_outcome_category' in df.columns:
            df['outcome_category'] = df['last_outcome_category']
        else:
            df['outcome_category'] = 'Unknown'
        # Ensure required columns exist
        for col in ['street_name', 'outcome_category', 'date', 'crime_type', 'longitude', 'latitude', 'region']:
            if col not in df.columns:
                df[col] = 'Unknown'
        # Fill missing values
        df = df.fillna('Unknown')
        df_list.append(df)
    all_crimes = pd.concat(df_list, ignore_index=True)
    # Drop rows with missing coordinates
    all_crimes = all_crimes[(all_crimes['longitude'] != 'Unknown') & (all_crimes['latitude'] != 'Unknown')]
    geometry = [Point(float(xy[0]), float(xy[1])) for xy in zip(all_crimes['longitude'], all_crimes['latitude'])]
    gdf = gpd.GeoDataFrame(all_crimes, geometry=geometry, crs="EPSG:4326")
    # List available filter columns
    filter_columns = [col for col in gdf.columns if col in ['crime_type', 'date', 'street_name', 'outcome_category', 'region']]
    return gdf, filter_columns
