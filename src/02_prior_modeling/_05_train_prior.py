"""
05_train_prior.py

This script trains and establishes the continuous-time travel priors 
required for the PA-TAD framework. It computes and exports three distinct 
baselines to solve the zero-shot cold-start problem on sparse networks:

1. Kinematic Baseline (Static): A strictly physics-based free-flow travel time 
   derived from segment length and statutory speed limits, independent of time.
2. Tabular Baseline (Time-Aware XGBoost): A machine learning baseline optimized 
   on raw traversal seconds using physical edge features and continuous harmonic 
   time variables, but evaluating segments in spatial isolation.
3. Topological Prior (Continuous-Time GAT): A Bayesian Graph Attention Network 
   that fuses spatial connectivity with continuous temporal representations. 
   It learns to predict localized travel-time residuals (adjustments upon free-flow) 
   and dynamic predictive uncertainties for any edge at any time of day.

Outputs:
- xgboost_weights.json: The trained Tabular Baseline model.
- GAT_prior_weights.pth: The trained Topological GAT Prior.
- graph_artifacts.pkl: Saved tensors and scalers for downstream evaluation.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

import pandas as pd 
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
import time
import copy
import pickle
from pathlib import Path 
import xgboost as xgb
import torch.nn.functional as F
import argparse
import json
import gc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="e.g. 'porto' or '2014'")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_048_576,
        help="Observation batch size. Transferred from CPU to GPU in chunks.",
    )
    parser.add_argument(
        "--max-xgb-samples",
        type=int,
        default=2_000_000,
        help="Maximum training observations sampled for XGBoost baseline.",
    )
    return parser.parse_args()

BASE_DIR = Path(__file__).resolve().parents[2]
COLD_START_DIR = BASE_DIR / 'data' / '04_cold_start'

class BayesianTrafficGAT(nn.Module):
    def __init__(self, static_in_channels, time_channels=6, hidden_dim=64, num_heads=2, dropout_rate=0.2):
        super(BayesianTrafficGAT, self).__init__()
        self.gat1 = GATConv(static_in_channels, hidden_dim, heads=num_heads, concat=True, dropout=dropout_rate)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        
        combined_dim = hidden_dim + time_channels
        fusion_hidden_dim = 32
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        self.mu_head = nn.Linear(fusion_hidden_dim, 1)
        self.log_var_head = nn.Linear(fusion_hidden_dim, 1)

    def encode(self, x_static, edge_index):
        x = F.leaky_relu(self.gat1(x_static, edge_index))
        x = self.dropout(x)
        x = F.leaky_relu(self.gat2(x, edge_index))
        return x

    def predict(self, x, edge_index, target_edge_ids, time_features, log_base_time, global_log_delay=None):
        row, col = edge_index
        queried_edge_embeddings = x[col][target_edge_ids]
        fused_features = torch.cat([queried_edge_embeddings, time_features], dim=-1)
        hidden_interaction = self.fusion_mlp(fused_features)

        mu_adj = self.mu_head(hidden_interaction).squeeze(-1)
        log_var = self.log_var_head(hidden_interaction).squeeze(-1)
        log_var = torch.clamp(log_var, min=-4.0, max=4.0)
        mu_adj = torch.clamp(mu_adj, min=-1.0, max=4.0)
        if global_log_delay is None:
            global_log_delay = torch.zeros_like(mu_adj)
        mu_final = log_base_time[target_edge_ids] + global_log_delay + mu_adj
        return mu_final, log_var

    def forward(self, x_static, edge_index, target_edge_ids, time_features, log_base_time, global_log_delay=None):
        """End-to-end forward pass encoding static topology and predicting target edge distributions."""
        x_enc = self.encode(x_static, edge_index)
        return self.predict(x_enc, edge_index, target_edge_ids, time_features, log_base_time, global_log_delay)

def gaussian_nll_loss(mu, log_var, target):
    var = torch.exp(log_var)
    loss = 0.5 * np.log(2 * np.pi) + 0.5 * log_var + ((target - mu) ** 2) / (2 * var)
    return loss.mean()

TEMPORAL_FEATURE_COLUMNS = [
    "week_sin_168", "week_cos_168", "day_sin_24", "day_cos_24", "half_day_sin_12", "half_day_cos_12",
]

def add_temporal_features_polars(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    hour_of_week = (
        (pl.col("datetime").dt.weekday() - 1) * 24.0
        + pl.col("datetime").dt.hour()
        + pl.col("datetime").dt.minute() / 60.0
    ).alias("hour_of_week")

    two_pi = 2.0 * np.pi
    return (
        df_lazy
        .with_columns(hour_of_week)
        .with_columns([
            (pl.col("hour_of_week") * (two_pi / 168.0)).sin().alias("week_sin_168"),
            (pl.col("hour_of_week") * (two_pi / 168.0)).cos().alias("week_cos_168"),
            (pl.col("hour_of_week") * (two_pi / 24.0)).sin().alias("day_sin_24"),
            (pl.col("hour_of_week") * (two_pi / 24.0)).cos().alias("day_cos_24"),
            (pl.col("hour_of_week") * (two_pi / 12.0)).sin().alias("half_day_sin_12"),
            (pl.col("hour_of_week") * (two_pi / 12.0)).cos().alias("half_day_cos_12"),
        ])
    )

def fit_global_weekly_prior(sampled_edges, sampled_seconds, sampled_how, log_base_time, ridge=1e-3):
    valid = (sampled_edges >= 0) & (sampled_seconds > 0)
    edge_ids = sampled_edges[valid]
    tau = sampled_how[valid]
    seconds = sampled_seconds[valid]

    edge_tensor = torch.as_tensor(edge_ids, dtype=torch.long, device=log_base_time.device)
    target = np.log(seconds) - log_base_time[edge_tensor].cpu().numpy()

    harmonics = np.arange(1, 9, dtype=np.float64)
    design = np.column_stack([np.ones(len(tau)), *(
        val for harmonic in harmonics
        for val in (np.sin(2 * np.pi * harmonic * tau / 168.0), np.cos(2 * np.pi * harmonic * tau / 168.0))
    )])
    regularizer = np.eye(design.shape[1]) * ridge
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return {
        "period_hours": 168,
        "harmonics": harmonics.astype(int).tolist(),
        "coefficients": coefficients.tolist(),
        "ridge": ridge,
        "sample_size": int(len(target)),
        "seed": 42
    }

def evaluate_global_weekly_prior(hour_of_week, artifact):
    tau = np.asarray(hour_of_week, dtype=np.float64)
    result = np.full(tau.shape, artifact["coefficients"][0], dtype=np.float64)
    for index, harmonic in enumerate(artifact["harmonics"]):
        result += artifact["coefficients"][1 + index * 2] * np.sin(2 * np.pi * harmonic * tau / 168.0)
        result += artifact["coefficients"][2 + index * 2] * np.cos(2 * np.pi * harmonic * tau / 168.0)
    return result

def build_graph_and_train(tag: str, batch_size: int = 1_048_576, max_xgb_samples: int = 2_000_000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_dir = BASE_DIR / 'models' / 'prior_weights' / tag
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- STARTING PRIOR TRAINING PIPELINE ON {device.type.upper()} ({tag.upper()}) ---")

    print("Loading static graph data...")
    map_parquet_path = COLD_START_DIR / f"cold_start_graph_{tag}.parquet"
    map_df = pd.read_parquet(map_parquet_path)
    map_df["segmentkey"] = map_df["segmentkey"].astype(str)

    map_df['base_time_sec'] = map_df['segment_length'] / (map_df['speed_limit'] / 3.6)
    map_df['log_base_time'] = np.log(map_df['base_time_sec'].clip(lower=1.0))

    cat_dummies = pd.get_dummies(map_df['category'], prefix='cat', dtype=float)
    map_df = pd.concat([map_df, cat_dummies], axis=1)

    cont_cols = ['speed_limit', 'segment_length', 'tortuosity']
    feature_cols = cont_cols + list(cat_dummies.columns)

    scaler_graph = StandardScaler()
    map_df[feature_cols] = scaler_graph.fit_transform(map_df[feature_cols])

    unique_nodes = pd.concat([map_df['u'], map_df['v']]).unique()
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)

    u_idx_array = map_df['u'].map(node_mapping).values
    v_idx_array = map_df['v'].map(node_mapping).values
    feats_array = map_df[feature_cols].values.astype(float)
    
    edge_index = torch.tensor(np.vstack((u_idx_array, v_idx_array)), dtype=torch.long)
    node_features = np.zeros((num_nodes, len(feature_cols)))
    node_counts = np.zeros(num_nodes)
    
    for i in range(len(map_df)):
        n1, n2 = u_idx_array[i], v_idx_array[i]
        node_features[n1] += feats_array[i]
        node_features[n2] += feats_array[i]
        node_counts[n1] += 1
        node_counts[n2] += 1

    node_features = node_features / np.maximum(node_counts[:, None], 1)
    x_static = torch.tensor(node_features, dtype=torch.float).to(device)
    log_base_time = torch.tensor(map_df['log_base_time'].values, dtype=torch.float).to(device)
    
    map_df['edge_id'] = np.arange(len(map_df), dtype=np.int32)
    edge_lookup = map_df.set_index(['u', 'v', 'segmentkey'])['edge_id'].to_dict()

    # Create edge lookup directly via Polars without ArrowStringArray conversion error
    pl_edge_lookup = (
        pl.read_parquet(map_parquet_path)
        .with_columns(pl.col("segmentkey").cast(pl.String))
        .select(['u', 'v', 'segmentkey'])
        .with_columns(pl.int_range(0, pl.len(), dtype=pl.Int32).alias("edge_id"))
    )

    print("Loading trajectory splits with Polars (selective projection)...")
    train_path = COLD_START_DIR / f"train_traj_{tag}.parquet"
    val_path = COLD_START_DIR / f"val_traj_{tag}.parquet"

    def load_and_prep_split(path: Path) -> pl.DataFrame:
        needed_cols = ['edge_id', 'seconds', 'hour_of_week'] + TEMPORAL_FEATURE_COLUMNS
        return (
            pl.scan_parquet(path)
            .with_columns(pl.col("segmentkey").cast(pl.String))
            .pipe(add_temporal_features_polars)
            .join(pl_edge_lookup.lazy(), on=['u', 'v', 'segmentkey'], how='inner')
            .select(needed_cols)
            .collect()
        )

    print("Processing Train trajectory features...")
    train_pl = load_and_prep_split(train_path)
    print(f"Train split loaded: {len(train_pl):,} observations")

    print("Processing Val trajectory features...")
    val_pl = load_and_prep_split(val_path)
    print(f"Val split loaded: {len(val_pl):,} observations")

    # Fit Global Weekly Prior
    sample_size = min(len(train_pl), 200_000)
    sample_idx = np.random.RandomState(42).choice(len(train_pl), size=sample_size, replace=False)
    s_edges = train_pl['edge_id'].to_numpy()[sample_idx]
    s_seconds = train_pl['seconds'].to_numpy()[sample_idx]
    s_how = train_pl['hour_of_week'].to_numpy()[sample_idx]

    global_weekly_prior = fit_global_weekly_prior(s_edges, s_seconds, s_how, log_base_time)
    with open(models_dir / "global_weekly_prior.json", "w", encoding="utf-8") as file:
        json.dump(global_weekly_prior, file, indent=2)

    # Extract calibration and XGBoost artifacts before dropping dataframes
    calibration_columns = ['edge_id', 'hour_of_week', 'seconds'] + TEMPORAL_FEATURE_COLUMNS
    calibration_df = val_pl.select(calibration_columns).to_pandas()
    unique_train_edges = set(train_pl['edge_id'].unique().to_list())

    train_sample_xgb = train_pl.sample(n=min(len(train_pl), max_xgb_samples), seed=42).to_pandas()

    print("Preparing observation tensors on CPU...")
    def prep_tensors_polars(df: pl.DataFrame):
        edge_ids = torch.from_numpy(df['edge_id'].to_numpy().astype(np.int64))
        time_feats = torch.from_numpy(df.select(TEMPORAL_FEATURE_COLUMNS).to_numpy().astype(np.float32))
        
        how = df['hour_of_week'].to_numpy()
        g_delay = evaluate_global_weekly_prior(how, global_weekly_prior)
        global_delay = torch.from_numpy(g_delay.astype(np.float32))
        
        secs = np.maximum(df['seconds'].to_numpy().astype(np.float32), 1.0)
        targets = torch.from_numpy(np.log(secs))
        return edge_ids, time_feats, global_delay, targets

    train_edges, train_time, train_global_delay, y_train = prep_tensors_polars(train_pl)
    val_edges, val_time, val_global_delay, y_val = prep_tensors_polars(val_pl)
    edge_index = edge_index.to(device)

    # Immediately release Polars DataFrames from RAM
    del train_pl, val_pl
    gc.collect()

    print("Initializing Model...")
    model_config = {
        'time_channels': len(TEMPORAL_FEATURE_COLUMNS),
        'hidden_dim': 64,
        'num_heads': 2,
        'dropout_rate': 0.2,
    }
    model = BayesianTrafficGAT(
        static_in_channels=len(feature_cols),
        **model_config
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    epochs, patience = 300, 20
    best_val_loss, patience_counter = float('inf'), 0
    best_model_weights = None
    start_time = time.time()

    def batch_iterate(*tensors, b_size, shuffle=True):
        n = tensors[0].shape[0]
        idx_all = torch.randperm(n) if shuffle else torch.arange(n)
        for i in range(0, n, b_size):
            idx = idx_all[i:i + b_size]
            yield tuple(t[idx] for t in tensors)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        x = model.encode(x_static, edge_index)
        x_detached = x.detach().requires_grad_(True)

        total_loss, n_batches = 0.0, 0
        for b_edges, b_time, b_global_delay, b_y in batch_iterate(
            train_edges, train_time, train_global_delay, y_train, b_size=batch_size
        ):
            b_edges = b_edges.to(device, non_blocking=True)
            b_time = b_time.to(device, non_blocking=True)
            b_global_delay = b_global_delay.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)
            
            mu, log_var = model.predict(x_detached, edge_index, b_edges, b_time, log_base_time, b_global_delay)
            loss = gaussian_nll_loss(mu, log_var, b_y)
            
            loss.backward()
            total_loss += loss.item()
            n_batches += 1
            del b_edges, b_time, b_global_delay, b_y, mu, log_var, loss

        if x_detached.grad is not None:
            x.backward(x_detached.grad)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        avg_train_loss = total_loss / max(n_batches, 1)
        del x, x_detached

        # Validation
        model.eval()
        with torch.no_grad():
            x_val = model.encode(x_static, edge_index)
            val_total, val_batches = 0.0, 0
            for b_edges, b_time, b_global_delay, b_y in batch_iterate(
                val_edges, val_time, val_global_delay, y_val, b_size=batch_size, shuffle=False
            ):
                b_edges = b_edges.to(device, non_blocking=True)
                b_time = b_time.to(device, non_blocking=True)
                b_global_delay = b_global_delay.to(device, non_blocking=True)
                b_y = b_y.to(device, non_blocking=True)
                val_mu, val_log_var = model.predict(x_val, edge_index, b_edges, b_time, log_base_time, b_global_delay)
                val_total += gaussian_nll_loss(val_mu, val_log_var, b_y).item()
                val_batches += 1
                del b_edges, b_time, b_global_delay, b_y, val_mu, val_log_var
            val_loss = val_total / max(val_batches, 1)
            del x_val
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if epoch % 1 == 0:
            print(f"Epoch {epoch:03d} | Train NLL: {avg_train_loss:.4f} | Val NLL: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    if best_model_weights:
        model.load_state_dict(best_model_weights)
        
    print(f"Training Complete. Time: {time.time() - start_time:.2f}s")

    # Train XGBoost Baseline
    print("\nTraining XGBoost Tabular Baseline...")
    xgb_feat_cols = ['speed_limit', 'segment_length', 'tortuosity'] + TEMPORAL_FEATURE_COLUMNS + list(cat_dummies.columns)
    cols_to_merge = ['edge_id', 'speed_limit', 'segment_length', 'tortuosity'] + list(cat_dummies.columns)

    train_xgb_df = train_sample_xgb.merge(map_df[cols_to_merge], on='edge_id', how='left')
    X_train_xgb = train_xgb_df[xgb_feat_cols]
    
    s_edge_tensor = torch.as_tensor(train_xgb_df['edge_id'].values, dtype=torch.long, device=log_base_time.device)
    y_train_xgb = (
        np.log(train_xgb_df['seconds'].clip(lower=1.0).values)
        - log_base_time[s_edge_tensor].cpu().numpy()
        - evaluate_global_weekly_prior(train_xgb_df["hour_of_week"], global_weekly_prior)
    )
    
    xgb_model = xgb.XGBRegressor(   
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1, 
        n_jobs=-2,               
        random_state=42
    )
    xgb_model.fit(X_train_xgb, y_train_xgb)
    xgb_model.save_model(os.path.join(models_dir, "xgboost_weights.json"))
    print("XGBoost Training Complete.")

    # Export Artifacts
    print("\nSaving Model and Graph Artifacts...")
    torch.save(model.state_dict(), os.path.join(models_dir, "GAT_prior_weights.pth"))

    with open(os.path.join(models_dir, "graph_artifacts.pkl"), 'wb') as f:
        pickle.dump({
            'graph_data': Data(x=x_static.cpu(), edge_index=edge_index.cpu()),
            'log_base_time': log_base_time.cpu(),
            'edge_lookup': edge_lookup,
            'scaler': scaler_graph,
            'train_edge_ids': unique_train_edges,
            'temporal_feature_columns': TEMPORAL_FEATURE_COLUMNS,
            'global_weekly_prior_file': "global_weekly_prior.json",
            'calibration_df': calibration_df,
            'model_config': model_config,
        }, f)
        
    print(f"Success! Artifacts saved to {models_dir}.")

if __name__ == "__main__":
    args = parse_args()
    build_graph_and_train(tag=args.tag, batch_size=args.batch_size, max_xgb_samples=args.max_xgb_samples)
