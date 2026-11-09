"""
04_cold_start_split.py

This script prepares the dataset for a strict "Cold Start" anomaly detection 
evaluation by creating spatially disjoint sets of trajectories on the 
largest connected component of the road network.

Key Transformations:
- Largest Connected Component Filtering: Discards disconnected islands and 
  outlier road segments by retaining only the largest connected component of the graph.
- Partial Traversal Removal: Drops the first and last segments of every trip. 
  GPS recordings often begin or end halfway through a road segment; removing 
  these prevents partial durations from skewing the travel-time distributions.
- Zero-duration Filtering: Drops any records where duration (seconds) <= 0.
- Speed Bound Filtering: Drops traversals with speeds exceeding 150 km/h to prevent
  severe map-matching or GPS artifacts (e.g. 200 - 2000 km/h) from skewing prior distributions.
- Geographic Edge Split: Extracts all unique physical segments in the largest connected component
  and partitions them strictly into Train (60%), Validation (20%), and Test (20%).
- Trajectory Allocation: Uses Polars lazy evaluation and streaming sink to filter 
  and route trajectories to splits efficiently without memory saturation.

Outputs:
- cold_start_graph_{tag}.parquet: The static map graph (largest connected component).
- train_traj_{tag}.parquet, val_traj_{tag}.parquet, test_traj_{tag}.parquet: The disjoint datasets.
"""

import pandas as pd
import numpy as np
import polars as pl
import networkx as nx
from pathlib import Path
import os
import argparse

# ==========================================
# 0. CONFIGURATION & SETUP
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True,
                    help="Dataset identifier tag, e.g. 'porto' or '2014'")
parser.add_argument("--max-speed-kmh", type=float, default=150.0,
                    help="Maximum speed in km/h allowed before filtering outliers (default: 150.0)")
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = BASE_DIR / 'data' / '03_processed'
COLD_START_DIR = BASE_DIR / 'data' / '04_cold_start'

COLD_START_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. LOAD PROCESSED MAP & EXTRACT LARGEST CONNECTED COMPONENT
# ==========================================
print("Loading processed map data...")
map_path = PROCESSED_DATA_DIR / f"map_directed_{args.tag}.parquet"
traj_path = PROCESSED_DATA_DIR / f"traj_mapped_{args.tag}.parquet"

if not map_path.exists():
    raise FileNotFoundError(f"Processed map not found at {map_path}. Please run 03_preprocess.py first.")
if not traj_path.exists():
    raise FileNotFoundError(f"Processed trajectories not found at {traj_path}. Please run 03_preprocess.py first.")

df_map = pd.read_parquet(map_path)

# Ensure uniform types for key identifier and node columns
df_map['u'] = df_map['u'].astype(np.int64)
df_map['v'] = df_map['v'].astype(np.int64)
df_map['segmentkey'] = df_map['segmentkey'].astype(str)

# Ensure length column exists in map
length_col = 'segment_length' if 'segment_length' in df_map.columns else ('meters' if 'meters' in df_map.columns else None)
if length_col is None:
    raise KeyError("Map does not contain 'segment_length' or 'meters' column.")

print(f"Original map shape: {len(df_map):,} directed edges across {df_map['segmentkey'].nunique():,} physical segments.")

# Build undirected graph to find connected components
print("Extracting largest connected component (filtering out isolated islands)...")
g_undirected = nx.Graph()
for _, row in df_map[['u', 'v']].iterrows():
    g_undirected.add_edge(row['u'], row['v'])

connected_components = list(nx.connected_components(g_undirected))
n_components = len(connected_components)
print(f"Detected {n_components:,} connected components in network.")

largest_cc_nodes = max(connected_components, key=len)
print(f"Largest connected component contains {len(largest_cc_nodes):,} nodes "
      f"({100.0 * len(largest_cc_nodes) / g_undirected.number_of_nodes():.2f}% of all nodes).")

# Retain only edges whose endpoints both belong to the largest component
mask_in_largest_cc = df_map['u'].isin(largest_cc_nodes) & df_map['v'].isin(largest_cc_nodes)
n_dropped_edges = int((~mask_in_largest_cc).sum())
df_map = df_map[mask_in_largest_cc].reset_index(drop=True)

print(f"Filtered out {n_dropped_edges:,} island edges. "
      f"Remaining in largest component: {len(df_map):,} directed edges.")

# Ensure directed_edge_id exists
if 'directed_edge_id' not in df_map.columns:
    df_map['directed_edge_id'] = df_map['segmentkey'].astype(str) + '_' + df_map['traj_direction'].astype(str).str[0]
df_map['directed_edge_id'] = df_map['directed_edge_id'].astype(str)

# ==========================================
# 2. PERFORM SEGMENT-BASED SPLIT
# ==========================================
print("\nPerforming Geographic Cold Start Split on Largest Connected Component...")
# Split on segmentkey (not u,v) so FORWARD/BACKWARD of the same physical
# road segment always land in the same split.
unique_segments = df_map[['segmentkey']].drop_duplicates()

# Shuffle segments reproducibly
np.random.seed(42)
shuffled_segments = unique_segments.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split indices (60% Train, 20% Val, 20% Test)
n_total = len(shuffled_segments)
train_end = int(n_total * 0.60)
val_end = int(n_total * 0.80)

train_segments = shuffled_segments.iloc[:train_end]
val_segments = shuffled_segments.iloc[train_end:val_end]
test_segments = shuffled_segments.iloc[val_end:]

# Build edge sets with directed_edge_id
train_edges = df_map.merge(train_segments, on='segmentkey')[['directed_edge_id', 'u', 'v', 'segmentkey']].drop_duplicates()
val_edges = df_map.merge(val_segments, on='segmentkey')[['directed_edge_id', 'u', 'v', 'segmentkey']].drop_duplicates()
test_edges = df_map.merge(test_segments, on='segmentkey')[['directed_edge_id', 'u', 'v', 'segmentkey']].drop_duplicates()

# Defensive check: confirm the three edge sets are truly disjoint
_train_keys = set(train_edges['directed_edge_id'])
_val_keys = set(val_edges['directed_edge_id'])
_test_keys = set(test_edges['directed_edge_id'])
assert not (_train_keys & _val_keys), "Train/Val edge sets overlap on directed_edge_id — split leakage!"
assert not (_train_keys & _test_keys), "Train/Test edge sets overlap on directed_edge_id — split leakage!"
assert not (_val_keys & _test_keys), "Val/Test edge sets overlap on directed_edge_id — split leakage!"

print(f"Total Unique Segments in CC: {n_total:,}")
print(f"Train Directed Edges:       {len(train_edges):,}")
print(f"Val Directed Edges:         {len(val_edges):,}")
print(f"Test Directed Edges:        {len(test_edges):,}")

# Export fixed cold start graph
graph_out = COLD_START_DIR / f"cold_start_graph_{args.tag}.parquet"
df_map.to_parquet(graph_out, index=False)

# ==========================================
# 3. FILTER BOUNDARIES & MAP TRIPS VIA POLARS
# ==========================================
print("\nPreparing trajectory split with Polars (Lazy)...")
pl_train_edges = pl.from_pandas(train_edges[['directed_edge_id']].astype({'directed_edge_id': str}))
pl_val_edges = pl.from_pandas(val_edges[['directed_edge_id']].astype({'directed_edge_id': str}))
pl_test_edges = pl.from_pandas(test_edges[['directed_edge_id']].astype({'directed_edge_id': str}))

# Map lookup for segment length to calculate speed
edge_lengths = (
    df_map[['directed_edge_id', length_col]]
    .drop_duplicates(subset=['directed_edge_id'])
    .rename(columns={length_col: 'meters'})
)
edge_lengths['directed_edge_id'] = edge_lengths['directed_edge_id'].astype(str)
edge_lengths['meters'] = edge_lengths['meters'].astype(np.float64)
pl_edge_lengths = pl.from_pandas(edge_lengths[['directed_edge_id', 'meters']])

# Ensure strict positive traversal time (> 0) and filter trip boundary segments
traj_lazy = (
    pl.scan_parquet(traj_path)
    .filter(pl.col("seconds") > 0)
)

schema_names = traj_lazy.collect_schema().names()
traj_exprs = []
if 'directed_edge_id' not in schema_names:
    dir_col = 'traj_direction' if 'traj_direction' in schema_names else 'direction'
    traj_exprs.append(
        (pl.col('segmentkey').cast(pl.String) + '_' + pl.col(dir_col).cast(pl.String).str.slice(0, 1)).alias('directed_edge_id')
    )
else:
    traj_exprs.append(pl.col('directed_edge_id').cast(pl.String))

if 'u' in schema_names:
    traj_exprs.append(pl.col('u').cast(pl.Int64))
if 'v' in schema_names:
    traj_exprs.append(pl.col('v').cast(pl.Int64))
if 'segmentkey' in schema_names:
    traj_exprs.append(pl.col('segmentkey').cast(pl.String))

traj_lazy = traj_lazy.with_columns(traj_exprs)

print(f"Removing boundary segments per trip and filtering records exceeding {args.max_speed_kmh} km/h...")
traj_filtered = (
    traj_lazy
    .join(pl_edge_lengths.lazy(), on='directed_edge_id', how='inner')
    .with_columns([
        pl.col("trip_segmentno").min().over("trip_id").alias("_min_seg"),
        pl.col("trip_segmentno").max().over("trip_id").alias("_max_seg"),
        ((pl.col("meters") / pl.col("seconds")) * 3.6).alias("_speed_kmh"),
    ])
    .filter(
        (pl.col("trip_segmentno") != pl.col("_min_seg")) &
        (pl.col("trip_segmentno") != pl.col("_max_seg")) &
        (pl.col("_speed_kmh") <= args.max_speed_kmh)
    )
    .drop(["_min_seg", "_max_seg", "_speed_kmh", "meters"])
)

# Allocate to Train, Val, Test by inner join on directed_edge_id
train_out = COLD_START_DIR / f"train_traj_{args.tag}.parquet"
val_out = COLD_START_DIR / f"val_traj_{args.tag}.parquet"
test_out = COLD_START_DIR / f"test_traj_{args.tag}.parquet"

print("Writing Train trajectory split...")
(
    traj_filtered
    .join(pl_train_edges.lazy(), on='directed_edge_id', how='inner')
    .sink_parquet(train_out, compression='snappy')
)

print("Writing Val trajectory split...")
(
    traj_filtered
    .join(pl_val_edges.lazy(), on='directed_edge_id', how='inner')
    .sink_parquet(val_out, compression='snappy')
)

print("Writing Test trajectory split...")
(
    traj_filtered
    .join(pl_test_edges.lazy(), on='directed_edge_id', how='inner')
    .sink_parquet(test_out, compression='snappy')
)

# Counts report
train_count = pl.scan_parquet(train_out).select(pl.len()).collect().item()
val_count = pl.scan_parquet(val_out).select(pl.len()).collect().item()
test_count = pl.scan_parquet(test_out).select(pl.len()).collect().item()

print(f"\nCold Start Split Complete!")
print(f"Train: {train_count:,} segment traversals")
print(f"Val:   {val_count:,} segment traversals")
print(f"Test:  {test_count:,} segment traversals")
