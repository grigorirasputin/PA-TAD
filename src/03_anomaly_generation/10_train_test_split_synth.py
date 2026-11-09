"""
Splits the generated synthetic trajectory data into 
60% train trip_ids
20% validation trip_ids 
20% test trip_ids

Preserves segmentgeo and segangle in the regional map files, while keeping 
trajectory files lean (free of heavy geometries).
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True, help="dataset pair label, e.g. '2014', '2016_itsp', or 'porto'")
parser.add_argument("--regions", nargs="+", default=None, help="regions to split (e.g. 'porto', 'aalborg copenhagen denmark')")
args = parser.parse_args()

SYNTH_DIR = BASE_DIR / "data" / "05_synthetic" / args.tag
OUT_DIR = BASE_DIR / "data" / "06_model_ready" / args.tag
OUT_DIR.mkdir(parents=True, exist_ok=True)

if args.regions is not None:
    active_regions = args.regions
else:
    if "porto" in str(args.tag).lower():
        active_regions = ["porto"]
    else:
        active_regions = ["aalborg", "copenhagen", "denmark"]

# Target splits: 60% Train, 20% Validation, 20% Test
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
RANDOM_SEED = 42 


def process_splits():
    for region in active_regions:
        print(f"\n{'='*50}")
        print(f"Processing Train/Val/Test Split for: {region.upper()}")
        print(f"{'='*50}")
        
        traj_path = SYNTH_DIR / f"synthetic_trajectories_{region}.parquet"
        if not traj_path.exists():
            print(f"Skipping {region} - file not found at {traj_path}")
            continue
            
        # 1. Load the Trajectories
        print("Loading data...")
        df = pd.read_parquet(traj_path)
        
        # Ensure heavy geometry columns are kept out of trajectory files
        cols_to_drop = [c for c in ["segmentgeo", "geometry"] if c in df.columns]
        if cols_to_drop:
            print(f"Dropping trajectory-level geometry column(s) {cols_to_drop} to preserve minimal storage footprint.")
            df.drop(columns=cols_to_drop, inplace=True)

        # Harmonize spatial_subtype for normal trajectories
        if "spatial_subtype" in df.columns and "anomaly_class" in df.columns:
            is_normal = df["anomaly_class"].astype(str).str.strip().str.lower() == "normal"
            df.loc[is_normal, "spatial_subtype"] = "normal"
        
        # 2. Extract Trip-Level Metadata
        print("Extracting trip-level classes...")
        trip_meta = df[["trip_id", "anomaly_class"]].drop_duplicates(subset=["trip_id"])
        
        # 3. Perform Stratified Splits
        temp_ratio = VAL_RATIO + TEST_RATIO
        train_trips, temp_trips = train_test_split(
            trip_meta, 
            test_size=temp_ratio, 
            stratify=trip_meta["anomaly_class"], 
            random_state=RANDOM_SEED
        )
        
        test_fraction_of_temp = TEST_RATIO / temp_ratio
        val_trips, test_trips = train_test_split(
            temp_trips, 
            test_size=test_fraction_of_temp, 
            stratify=temp_trips["anomaly_class"], 
            random_state=RANDOM_SEED
        )
        
        # 4. Map the Splits back to the Original DataFrame
        print("Mapping splits back to segment data...")
        split_mapping = {}
        split_mapping.update({t: "train" for t in train_trips["trip_id"]})
        split_mapping.update({t: "val" for t in val_trips["trip_id"]})
        split_mapping.update({t: "test" for t in test_trips["trip_id"]})
        
        df["split"] = df["trip_id"].map(split_mapping)
        
        # 5. Validation and Reporting
        print("\nTrip Distribution by Split:")
        trip_counts = df.drop_duplicates(subset=["trip_id"])["split"].value_counts()
        print(trip_counts)
        
        print("\nClass Distribution in Test Set:")
        test_class_counts = df[df["split"] == "test"].drop_duplicates(subset=["trip_id"])["anomaly_class"].value_counts(normalize=True)
        print(test_class_counts)
        
        # 6. Save the Model-Ready Data
        out_path = OUT_DIR / f"model_ready_trajectories_{region}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"\nSaved model-ready dataset to: {out_path.name}")
        
        # Copy map_df and graph to the new directory for model access
        map_df_path = SYNTH_DIR / f"map_df_{region}.parquet"
        if map_df_path.exists():
            dest_map_path = OUT_DIR / f"model_ready_map_df_{region}.parquet"
            shutil.copy(map_df_path, dest_map_path)
            
            # Verify required map columns
            map_peek = pd.read_parquet(dest_map_path)
            has_geo = "segmentgeo" in map_peek.columns
            has_angle = "segangle" in map_peek.columns
            print(f"Copied map_df_{region}.parquet -> model_ready_map_df_{region}.parquet "
                  f"(has segmentgeo: {has_geo}, has segangle: {has_angle})")

            # Referential integrity check
            map_nodes = set(map_peek["u"].unique()).union(set(map_peek["v"].unique()))
            traj_nodes = set(df["u"].unique()).union(set(df["v"].unique()))
            missing_nodes = traj_nodes - map_nodes
            if missing_nodes:
                print(f"WARNING: Found {len(missing_nodes)} trajectory nodes not present in map_df for {region}!")
            else:
                print(f"Referential integrity verified: All {len(traj_nodes):,} trajectory nodes exist in map_df.")

        graph_path = SYNTH_DIR / f"graph_{region}.pkl"
        if graph_path.exists():
            shutil.copy(graph_path, OUT_DIR / f"graph_{region}.pkl")
            print(f"Copied graph_{region}.pkl -> model_ready directory")


if __name__ == "__main__":
    process_splits()
