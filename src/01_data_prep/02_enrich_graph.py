"""
02_enrich_graph.py

This script enriches the base OSM road network with accurate speed limits.
If the external Vejdirektoratet dataset is present in the raw data folder, it 
applies a spatial join and prior cascade logic to assign official speed limits.
If the external dataset is missing, it gracefully degrades to a fallback logic 
based purely on OSM road categories.

External Data Source: https://du-portal-ui.dataudveksler.app.vd.dk/data/112/overview
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--map-file", required=True,
                     help="e.g. osm_dk_20161101_map.parquet or osm_dk_20140101_map.parquet") # DO NOT USE ON PORTO! 
parser.add_argument("--tag", required=True,
                     help="short label for this run, e.g. '2014' or '2016'")
args = parser.parse_args()


# warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 0. PATHS & SETUP
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / 'data' / '01_raw'          # define BEFORE using it
INTERIM_DATA_DIR = BASE_DIR / 'data' / '02_interim'
INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. BASE OSM DATA LOAD
# ---------------------------------------------------------
print("Loading base OSM map data...")
osm_path = RAW_DATA_DIR / args.map_file
df_osm = pd.read_parquet(osm_path)

# Normalize 0s to NA so they can be seamlessly targeted by fill logic
df_osm['speedlimit_forward'] = df_osm['speedlimit_forward'].replace(0, pd.NA)

# ---------------------------------------------------------
# 2. EXTERNAL DATA ENRICHMENT (Vejdirektoratet)
# ---------------------------------------------------------
speed_limits_external = RAW_DATA_DIR / "speed_limits_danish_state_roads_vejdirektorater.gml"

if speed_limits_external.exists():
    print(f"\nExternal dataset found. Loading GML file... (This might take a moment)")
    gdf = gpd.read_file(speed_limits_external)
    print(f"Loaded {len(gdf)} official road segments.")

    # Projection Alignment
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        print("Converting projection to EPSG:4326 to match OpenStreetMap...")
        gdf = gdf.to_crs(epsg=4326)

    # Apply Updated Prior Speed Cascade Logic
    print("Applying string mapping and cascade logic...")
    speed_cols = ['HAST_GAELDENDE_HAST', 'HAST_LOKAL_HAST', 'HAST_GENEREL_HAST']
    for col in speed_cols:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

    # Cascade 1: Direct explicit speeds
    gdf['prior_speed'] = gdf['HAST_GAELDENDE_HAST'] \
        .combine_first(gdf.get('HAST_LOKAL_HAST', pd.Series(dtype=float))) \
        .combine_first(gdf.get('HAST_GENEREL_HAST', pd.Series(dtype=float)))

    # Cascade 2-6: Rule-based overrides
    if 'VEJTYPESKILTET' in gdf.columns:
        gdf['prior_speed'] = np.where(gdf['prior_speed'].isna() & (gdf['VEJTYPESKILTET'] == 'Motorvej'), 130, gdf['prior_speed'])
        gdf['prior_speed'] = np.where(gdf['prior_speed'].isna() & (gdf['VEJTYPESKILTET'] == 'Motortrafikvej'), 80, gdf['prior_speed'])
        
    if 'HAST_BYKODE' in gdf.columns:
        gdf['prior_speed'] = np.where(gdf['prior_speed'].isna() & (gdf['HAST_BYKODE'].str.contains('Byzone', na=False, case=False)), 50, gdf['prior_speed'])

    if 'VEJTYPESKILTET' in gdf.columns:
        gdf['prior_speed'] = np.where(gdf['prior_speed'].isna() & (gdf['VEJTYPESKILTET'] == 'Øvrige veje'), 80, gdf['prior_speed'])
        gdf['prior_speed'] = np.where(gdf['prior_speed'].isna() & gdf['VEJTYPESKILTET'].isna(), 50, gdf['prior_speed'])

    # Cleanup Redundant Columns
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    columns_to_keep = ['geometry', 'prior_speed', 'HAST_ANBEFALET_HAST', 'VEJTYPESKILTET', 'CPR_VEJNAVN']
    final_columns = [col for col in columns_to_keep if col in gdf.columns]
    gdf_vej = gdf[final_columns]

    # Spatial Join Execution
    print("Preparing spatial join with OSM data (projecting to meters)...")
    df_osm['geometry'] = gpd.GeoSeries.from_wkb(df_osm['segmentgeo'])
    gdf_osm = gpd.GeoDataFrame(df_osm, geometry='geometry', crs="EPSG:4326")

    gdf_osm = gdf_osm.to_crs("EPSG:25832")
    gdf_vej = gdf_vej.to_crs("EPSG:25832")

    gdf_merged = gpd.sjoin_nearest(
        gdf_osm, 
        gdf_vej[['geometry', 'prior_speed']], 
        how='left', 
        max_distance=15.0
    )

    gdf_merged = gdf_merged[~gdf_merged.index.duplicated(keep='first')]
    
    # Overwrite OSM speed limits with external priors where missing
    gdf_merged['speedlimit_forward'] = gdf_merged['speedlimit_forward'].fillna(gdf_merged['prior_speed'])
    
    # Revert back to standard pandas DataFrame, dropping heavy geometries
    df_osm = pd.DataFrame(gdf_merged).drop(columns=['geometry', 'index_right'], errors='ignore')

else:
    print("\n--- WARNING: External Dataset Missing ---")
    print(f"Vejdirektoratet GML file not found at: {speed_limits_external}")
    print("Skipping spatial enrichment. The graph will be built using baseline OSM data only.")
    print("To enable full enrichment, place the required GML file in the 01_raw folder.\n")

# ---------------------------------------------------------
# 3. OSM CATEGORY FALLBACK (Executes regardless of external data presence)
# ---------------------------------------------------------
print("Applying local OSM category fallbacks...")

osm_speed_fallbacks = {
    'residential': 50.0,
    'living_street': 15.0,
    'unclassified': 80.0,
    'tertiary': 50.0,
    'secondary': 80.0,
    'primary': 80.0,
    'service': 30.0,
    "motorway" : 130
}
print(f'amount of na speedlimits: {df_osm["speedlimit_forward"].isna().sum()} rows ')

df_osm['speedlimit_forward'] = df_osm['speedlimit_forward'].fillna(
    df_osm['category'].map(osm_speed_fallbacks)
)

# Final safety net: If absolutely everything else fails, assume 50 km/h
df_osm['speedlimit_forward'] = df_osm['speedlimit_forward'].fillna(50.0) # for track, road, ferry etc where it is very difficult to tell.

# ---------------------------------------------------------
# 4. FINAL STATISTICS & EXPORT
# ---------------------------------------------------------
export_path = INTERIM_DATA_DIR / f"osm_map_enriched_{args.tag}.parquet"
df_osm.to_parquet(export_path, index=False)

print("\n--- FINAL SPEED LIMIT DISTRIBUTION ---")
speed_counts = df_osm['speedlimit_forward'].value_counts(dropna=False).sort_index()
speed_pcts = df_osm['speedlimit_forward'].value_counts(dropna=False, normalize=True).sort_index() * 100

stats_df = pd.DataFrame({
    'Count': speed_counts,
    'Percentage (%)': speed_pcts.round(2)
})
stats_df.index.name = 'Speed Limit (km/h)'
print(stats_df)
print(f"\nTotal Segments: {len(df_osm):,}")
print(f"Data successfully exported to: {export_path}")
print("--------------------------------------\n")