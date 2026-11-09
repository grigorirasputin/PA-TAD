"""
01_extract.py

This script connects to the PostgreSQL database, extracts the map skeleton 
and matched trajectory data, and exports them as compressed Parquet files 
for downstream processing.
"""

import pandas as pd
from sqlalchemy import create_engine
import os
import time
import warnings
from pathlib import Path 
from dotenv import load_dotenv

load_dotenv() # loading password
warnings.filterwarnings('ignore') 

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
# script should run from p10 dir

BASE_DIR = Path(__file__).resolve().parents[2]

# 3. Safely join the path to the raw data folder
RAW_DATA_DIR = BASE_DIR / 'data' / '01_raw'

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Point directly to the .env file located in the exact same folder as this script
current_dir = Path(__file__).resolve().parent
env_path = current_dir / '.env'

# Load the specific file
load_dotenv(dotenv_path=env_path)

# Database connection parameters securely loaded
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
DBNAME = os.getenv("DB_NAME")
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")

if not all([HOST, PORT, DBNAME, USER, PASSWORD]):
    raise ValueError(f"Set DB_HOST, DB_PORT, DB_NAME, DB_USER and DB_PASSWORD in the .env file at: {env_path}")


# ==========================================
# 2. DATABASE CONNECTION
# ==========================================
print("Establishing database connection...")
engine = create_engine(f'postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}')

# ==========================================
# 3. EXTRACTION QUERIES 
# ==========================================

map_cols = "segmentkey, direction, startpoint, endpoint, category, segangle, meters, segmentid, segmentgeo, speedlimit_forward" #Inferred necessary cols
traj_cols = "id,trip_id,segmentkey,direction, trip_segmentno,seconds,timekey, datekey,primehash" # Tilføj primehash her 


query_map = f"SELECT {map_cols} FROM maps.osm_dk_20140101;"
query_traj = f"SELECT {traj_cols}  FROM mapmatched_data.viterbi_match_osm_dk_20140101;"

# ==========================================
# 4. EXECUTE & EXPORT: MAP SKELETON
# ==========================================
print("\n--- Extracting Map Skeleton ---")
start_time = time.time()
df_map = pd.read_sql(query_map, engine)

map_export_path = os.path.join(RAW_DATA_DIR, "osm_dk_20140101_map.parquet")

print(f"Loaded {df_map.shape[0]:,} rows. Compressing and saving to Parquet...")
# engine='pyarrow' is highly recommended for speed and memory efficiency

df_map.to_parquet(map_export_path, index=False, engine='pyarrow') 
print(f"Map extraction complete! ({time.time() - start_time:.2f} seconds)")

# Clear from RAM before loading the trajectory table (Not a significant issue for us) (can be problematic with Select *)
del df_map 

# ==========================================
# 5. EXECUTE & EXPORT: TRAJECTORIES
# ==========================================
print("\n--- Extracting Trajectories (~14.5M rows, this will take a few minutes) ---")
start_time = time.time()
df_traj = pd.read_sql(query_traj, engine)

traj_export_path = os.path.join(RAW_DATA_DIR, "viterbi_match_trajectories.parquet")

print(f"Loaded {df_traj.shape[0]:,} rows. Compressing and saving to Parquet...")
df_traj.to_parquet(traj_export_path, index=False, engine='pyarrow')
print(f"Trajectory extraction complete! ({time.time() - start_time:.2f} seconds)")

print("\n Data Extraction Finished Successfully!")