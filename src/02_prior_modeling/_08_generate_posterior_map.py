"""
08_generate_posterior_map.py

This script bridges the gap between the zero-shot topological prior and the 
sparse historical ground truth. It utilizes Bayesian conjugate updates to 
calculate localized travel-time adjustments for every observed edge in the network.

Key Operations:
- Update Strategies: Computes Additive (L1), Proportional (L2), and 
  Feature-Weighted (L3) Bayesian posteriors.
- Sparsity-Resistant Extraction: Specifically isolates and exports the Level 2 
  (L2) proportional shift. This multiplicative scalar corrects the amplitude 
  of the travel-time expectation while preserving the continuous 24-hour phase 
  (shape) predicted by the GAT, preventing overfitting on highly sparse edges.
- Memory-Safe Execution: Uses PyArrow streaming batch reader and chunked GAT 
  inference so that large datasets (such as Porto) can be processed with constant 
  low RAM overhead without OOM errors.

Outputs:
- master_posterior_map.parquet: A unified lookup table containing physical edge 
  attributes, the GAT prior parameters, and the L2 Bayesian shift, serving as 
  the definitive environment map for the procedural anomaly generator.
"""

import os
import gc
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True, help="dataset pair label, e.g. 'porto' or '2014'")
parser.add_argument("--batch-size", type=int, default=32_768,
                    help="Batch size for streaming GAT inference (default: 32,768 to prevent OOM)")
args = parser.parse_args()

from _05_train_prior import (
    BayesianTrafficGAT, TEMPORAL_FEATURE_COLUMNS, evaluate_global_weekly_prior,
)

# ==========================================
# 0. CONFIGURATION & PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = BASE_DIR / 'data' / '03_processed'
COLD_START_DIR = BASE_DIR / 'data' / '04_cold_start'

POSSIBLE_MODELS_DIRS = [
    BASE_DIR / 'models' / 'prior_weights' / args.tag,
    BASE_DIR / 'models' / 'prior_weights',
    BASE_DIR / 'models' / args.tag,
]

POSSIBLE_RESULTS_DIRS = [
    BASE_DIR / 'results' / 'prior_modeling' / args.tag / 'bayesian_update',
    BASE_DIR / 'results' / 'prior_modeling' / args.tag,
    BASE_DIR / 'results' / 'prior_modeling',
]

SYNTH_DIR = BASE_DIR / 'data' / '05_synthetic' / args.tag
SYNTH_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_PRIOR_VAR = 1.940
DEFAULT_OBS_VAR = 1.531


def find_artifact_file(filenames, search_dirs):
    """Search for the first existing file among candidates across directories."""
    for d in search_dirs:
        for fname in filenames:
            p = d / fname
            if p.exists():
                return p
    return None


def compute_temporal_features_np(datetimes):
    """Vectorized calculation of continuous harmonic features without Polars overhead."""
    dt_series = pd.to_datetime(datetimes)
    hour_of_week = (
        dt_series.dt.dayofweek * 24.0
        + dt_series.dt.hour
        + dt_series.dt.minute / 60.0
    ).to_numpy(dtype=np.float64)

    two_pi = 2.0 * np.pi
    week_sin = np.sin(hour_of_week * (two_pi / 168.0)).astype(np.float32)
    week_cos = np.cos(hour_of_week * (two_pi / 168.0)).astype(np.float32)
    day_sin = np.sin(hour_of_week * (two_pi / 24.0)).astype(np.float32)
    day_cos = np.cos(hour_of_week * (two_pi / 24.0)).astype(np.float32)
    half_day_sin = np.sin(hour_of_week * (two_pi / 12.0)).astype(np.float32)
    half_day_cos = np.cos(hour_of_week * (two_pi / 12.0)).astype(np.float32)

    feats = np.column_stack([week_sin, week_cos, day_sin, day_cos, half_day_sin, half_day_cos])
    return hour_of_week, feats


# ==========================================
# 1. MAIN MAP GENERATOR
# ==========================================
def generate_master_map():
    start_time = datetime.now()
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] --- GENERATING MASTER POSTERIOR MAP ({args.tag.upper()}) ---")

    # ---------------------------------------------------------
    # A. LOAD GRAPH ARTIFACTS & GAT MODEL
    # ---------------------------------------------------------
    print("Locating and loading Graph Artifacts and GAT Model...")
    
    artifacts_candidates = [
        f"graph_artifacts_{args.tag}.pkl",
        "graph_artifacts.pkl",
    ]
    artifacts_path = find_artifact_file(artifacts_candidates, POSSIBLE_MODELS_DIRS)
    if artifacts_path is None:
        raise FileNotFoundError(
            f"Could not find graph artifacts among {artifacts_candidates} in {POSSIBLE_MODELS_DIRS}"
        )
    print(f"  Loaded graph artifacts from: {artifacts_path}")
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)

    graph_data = artifacts['graph_data'].to(device)
    log_base_time = artifacts['log_base_time'].to(device)
    edge_lookup = artifacts['edge_lookup']

    prior_json_candidates = [
        f"global_weekly_prior_{args.tag}.json",
        "global_weekly_prior.json",
    ]
    prior_json_path = find_artifact_file(prior_json_candidates, POSSIBLE_MODELS_DIRS)
    if prior_json_path is None:
        raise FileNotFoundError(
            f"Could not find global weekly prior JSON among {prior_json_candidates} in {POSSIBLE_MODELS_DIRS}"
        )
    print(f"  Loaded global weekly prior from: {prior_json_path}")
    with open(prior_json_path, encoding="utf-8") as file:
        global_weekly_prior = json.load(file)

    weights_candidates = [
        f"GAT_prior_weights_{args.tag}.pth",
        "GAT_prior_weights.pth",
        f"gat_prior_{args.tag}.pth",
        "gat_prior.pth",
    ]
    weights_path = find_artifact_file(weights_candidates, POSSIBLE_MODELS_DIRS)
    if weights_path is None:
        raise FileNotFoundError(
            f"Could not find GAT model weights among {weights_candidates} in {POSSIBLE_MODELS_DIRS}"
        )
    print(f"  Loaded GAT weights from: {weights_path}")
    
    gat_config = artifacts.get("model_config", {})
    gat_model = BayesianTrafficGAT(
        static_in_channels=graph_data.x.shape[1],
        **gat_config
    ).to(device)

    try:
        gat_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except Exception:
        gat_model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    gat_model.eval()

    prior_var = DEFAULT_PRIOR_VAR
    obs_var = DEFAULT_OBS_VAR
    param_candidates = [
        f"tuned_parameters_{args.tag}.csv",
        "tuned_parameters.csv",
    ]
    tuned_params_file = find_artifact_file(param_candidates, POSSIBLE_RESULTS_DIRS)
    if tuned_params_file is not None:
        try:
            tp_df = pd.read_csv(tuned_params_file)
            tp_df["model_clean"] = tp_df["model"].astype(str).str.strip().str.upper()
            tp_df["level_clean"] = tp_df["update_level"].astype(str).str.strip().str.upper()
            tp_df["cluster_clean"] = tp_df["variance_cluster"].astype(str).str.strip().str.lower()

            gat_l2 = tp_df[
                (tp_df["model_clean"].isin(["GAT", "GAT (OURS)"])) &
                (tp_df["level_clean"].isin(["L2", "L2_L3", "LEVEL 2 (PROPORTIONAL)"])) &
                (tp_df["cluster_clean"] == "global")
            ]
            if not gat_l2.empty:
                prior_var = float(gat_l2["prior_variance"].iloc[0])
                obs_var = float(gat_l2["observation_variance"].iloc[0])
                print(f"  Loaded tuned L2 hyperparameters from {tuned_params_file.name}: "
                      f"prior_var={prior_var:.4f}, obs_var={obs_var:.4f}")
            else:
                print(f"  Tuned parameters found in {tuned_params_file.name} but GAT L2 global row not matched. Using defaults.")
        except Exception as e:
            print(f"  Could not parse tuned hyperparameters ({e}), using defaults.")
    else:
        print(f"  No tuned parameters CSV found. Using default hyperparameters: prior_var={prior_var:.4f}, obs_var={obs_var:.4f}")

    # ---------------------------------------------------------
    # B. BUILD MASTER MAP DATAFRAME & LOOKUP
    # ---------------------------------------------------------
    map_candidates = [
        COLD_START_DIR / f"cold_start_graph_{args.tag}.parquet",
        PROCESSED_DATA_DIR / f"map_clean_{args.tag}.parquet",
        BASE_DIR / 'data' / '02_intermediate' / f"map_clean_{args.tag}.parquet",
    ]
    map_parquet_path = None
    for p in map_candidates:
        if p.exists():
            map_parquet_path = p
            break
            
    if map_parquet_path is None:
        raise FileNotFoundError(f"Could not find network graph parquet among {map_candidates}")

    print(f"Loading base network map from {map_parquet_path}...")
    map_df = pd.read_parquet(map_parquet_path)
    
    if "segmentkey" in map_df.columns:
        map_df["segmentkey"] = map_df["segmentkey"].astype(str)
    else:
        map_df["segmentkey"] = ""

    if "speed_limit" not in map_df.columns:
        map_df["speed_limit"] = 50.0
    else:
        map_df["speed_limit"] = map_df["speed_limit"].fillna(50.0).clip(lower=5.0)

    if "segment_length" not in map_df.columns:
        if "length" in map_df.columns:
            map_df["segment_length"] = map_df["length"]
        else:
            map_df["segment_length"] = 50.0
    map_df["segment_length"] = map_df["segment_length"].fillna(50.0).clip(lower=1.0)

    if "tortuosity" not in map_df.columns:
        map_df["tortuosity"] = 1.0
    else:
        map_df["tortuosity"] = map_df["tortuosity"].fillna(1.0)

    if "category" not in map_df.columns:
        map_df["category"] = "unclassified"

    map_df["log_base_time"] = np.log(
        (map_df["segment_length"] / (map_df["speed_limit"] / 3.6)).clip(lower=1.0)
    )

    sample_key = next(iter(edge_lookup.keys()))
    key_arity = len(sample_key) if isinstance(sample_key, tuple) else 1

    if key_arity == 3:
        map_df['edge_id'] = pd.MultiIndex.from_arrays(
            [map_df['u'], map_df['v'], map_df['segmentkey']]
        ).map(edge_lookup)
    elif key_arity == 2:
        map_df['edge_id'] = pd.MultiIndex.from_arrays(
            [map_df['u'], map_df['v']]
        ).map(edge_lookup)
    else:
        map_df['edge_id'] = map_df['u'].map(edge_lookup)

    map_df = map_df.dropna(subset=['edge_id']).drop_duplicates(subset=['edge_id']).copy()
    map_df['edge_id'] = map_df['edge_id'].astype(int)

    num_edges = len(map_df)
    max_edge_id = int(map_df['edge_id'].max())

    # Build memory-efficient lookup dicts for fast per-batch streaming resolution
    if key_arity == 3:
        stream_edge_lookup = {
            (u, v, str(s)): eid
            for u, v, s, eid in zip(map_df['u'], map_df['v'], map_df['segmentkey'], map_df['edge_id'])
        }
    elif key_arity == 2:
        stream_edge_lookup = {
            (u, v): eid
            for u, v, eid in zip(map_df['u'], map_df['v'], map_df['edge_id'])
        }
    else:
        stream_edge_lookup = {
            u: eid
            for u, eid in zip(map_df['u'], map_df['edge_id'])
        }

    # ---------------------------------------------------------
    # C. STREAM HISTORICAL TRAJECTORIES & ACCUMULATE RESIDUALS
    # ---------------------------------------------------------
    traj_candidates = [
        PROCESSED_DATA_DIR / f"traj_mapped_{args.tag}.parquet",
        COLD_START_DIR / f"test_traj_{args.tag}.parquet",
        PROCESSED_DATA_DIR / f"trajectories_{args.tag}.parquet",
    ]
    traj_path = None
    for p in traj_candidates:
        if p.exists():
            traj_path = p
            break

    if traj_path is None:
        raise FileNotFoundError(f"Could not find historical trajectories among {traj_candidates}")

    print(f"Streaming historical trajectories directly from {traj_path}...")

    # True constant memory accumulators
    edge_residual_sum = np.zeros(max_edge_id + 1, dtype=np.float64)
    edge_obs_count = np.zeros(max_edge_id + 1, dtype=np.int64)

    # Encode static graph once on GPU
    with torch.no_grad():
        encoded_graph = gat_model.encode(graph_data.x, graph_data.edge_index)

    # Inspect parquet metadata without loading full table
    parquet_file = pq.ParquetFile(traj_path)
    total_rows = parquet_file.metadata.num_rows
    schema_cols = parquet_file.schema.names
    print(f"Total rows in parquet file: {total_rows:,}")

    # Determine date/time column name
    dt_col_name = "datetime" if "datetime" in schema_cols else ("timestamp" if "timestamp" in schema_cols else None)
    if dt_col_name is None:
        raise KeyError(f"Neither 'datetime' nor 'timestamp' found in {traj_path}")

    needed_columns = ['u', 'v', 'seconds', dt_col_name]
    if 'segmentkey' in schema_cols and key_arity == 3:
        needed_columns.append('segmentkey')

    STREAM_CHUNK_SIZE = args.batch_size
    n_processed = 0

    with tqdm(total=total_rows, desc="Evaluating Prior & Residuals") as pbar:
        for batch_record in parquet_file.iter_batches(batch_size=STREAM_CHUNK_SIZE, columns=needed_columns):
            b_df = batch_record.to_pandas()
            if len(b_df) == 0:
                continue

            # Filter valid seconds
            sec_valid = (b_df['seconds'] > 0) & (b_df['seconds'] < 1800)
            if not sec_valid.any():
                pbar.update(len(b_df))
                continue
            b_df = b_df[sec_valid]

            # Fast edge_id mapping
            if key_arity == 3 and 'segmentkey' in b_df.columns:
                keys = zip(b_df['u'], b_df['v'], b_df['segmentkey'].astype(str))
            elif key_arity == 2:
                keys = zip(b_df['u'], b_df['v'])
            else:
                keys = b_df['u']

            b_edge_ids = np.fromiter((stream_edge_lookup.get(k, -1) for k in keys), dtype=np.int64, count=len(b_df))
            
            valid_edge_mask = b_edge_ids >= 0
            if not np.any(valid_edge_mask):
                pbar.update(len(batch_record))
                del b_df, keys, b_edge_ids, valid_edge_mask
                continue

            b_edge_np = b_edge_ids[valid_edge_mask]
            b_sec_np = b_df['seconds'].to_numpy(dtype=np.float32)[valid_edge_mask]
            b_dt = b_df[dt_col_name].iloc[valid_edge_mask]

            b_how_np, b_time_np = compute_temporal_features_np(b_dt)

            b_edges = torch.as_tensor(b_edge_np, dtype=torch.long, device=device)
            b_time = torch.as_tensor(b_time_np, dtype=torch.float32, device=device)
            b_delay = torch.as_tensor(
                evaluate_global_weekly_prior(b_how_np, global_weekly_prior),
                dtype=torch.float32,
                device=device,
            )

            with torch.no_grad():
                mu, _ = gat_model.predict(
                    encoded_graph, graph_data.edge_index, b_edges, b_time, log_base_time, b_delay
                )
                prior_gat = np.exp(mu.cpu().numpy())

            log_residuals = np.log(b_sec_np) - np.log(prior_gat)

            # Vectorized in-place buffer accumulation
            np.add.at(edge_residual_sum, b_edge_np, log_residuals)
            np.add.at(edge_obs_count, b_edge_np, 1)

            n_raw_batch = len(batch_record)
            n_processed += n_raw_batch
            pbar.update(n_raw_batch)

            # Explicit release of memory buffers
            del b_edges, b_time, b_delay, mu, prior_gat, log_residuals, b_df, b_edge_np, b_sec_np, b_how_np, b_time_np
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gc.collect()

    # ---------------------------------------------------------
    # D. APPLY BAYESIAN LEVEL 2 SHIFTS TO MASTER MAP
    # ---------------------------------------------------------
    print("\nCalculating Bayesian Level 2 Proportional Shifts...")
    map_edge_ids = map_df['edge_id'].values
    counts = edge_obs_count[map_edge_ids]
    sums = edge_residual_sum[map_edge_ids]

    l2_shifts = np.zeros(len(map_df), dtype=np.float64)
    valid = counts > 0
    l2_shifts[valid] = (prior_var / (counts[valid] * prior_var + obs_var)) * sums[valid]

    map_df['Bayesian_L2_Shift'] = l2_shifts
    map_df['Obs_Count'] = counts

    # ---------------------------------------------------------
    # E. EXPORT MASTER MAP
    # ---------------------------------------------------------
    export_cols = [
        'edge_id', 'u', 'v', 'segment_length', 'speed_limit', 'category', 'tortuosity',
        'log_base_time', 'Bayesian_L2_Shift', 'Obs_Count'
    ]
    
    out_path = SYNTH_DIR / "master_posterior_map.parquet"
    map_df[export_cols].to_parquet(out_path, index=False)
    
    warm_count = int(np.sum(valid))
    end_time = datetime.now()
    print("\n" + "=" * 80)
    print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] --- MASTER MAP GENERATED ---")
    print(f"Total Edges in Map: {num_edges:,}")
    print(f"Edges with observed data (Warm): {warm_count:,} ({(warm_count / max(num_edges, 1) * 100):.1f}%)")
    print(f"Edges relying strictly on GAT Prior (Cold): {num_edges - warm_count:,}")
    print(f"Saved to: {out_path}")
    print(f"Total Execution Time: {end_time - start_time}")
    print("=" * 80)


if __name__ == "__main__":
    generate_master_map()
