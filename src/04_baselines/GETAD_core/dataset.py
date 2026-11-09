"""
GETAD_core/dataset.py
Data processing, graph extraction, and PyTorch dataset definitions for GETAD.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from .model import EOT_TOKEN, NUM_SPECIAL_TOKENS, PAD_TOKEN, SOT_TOKEN


def build_graph_and_mapping(map_df_path, d_max, t_path=None, logger=None):
    """
    Constructs road network graph, node features, empirical edge transition frequencies,
    and GPE shortest path matrix.
    Uses int8 SciPy sparse BFS to eliminate dictionary pointer memory bloat.
    """
    map_df_path = Path(map_df_path)
    log_fn = logger.info if logger else print
    log_fn(f"Loading graph from: {map_df_path.name}")
    df = pd.read_parquet(map_df_path)

    unique_nodes = sorted(list(set(df["u"].unique()).union(set(df["v"].unique()))))
    num_nodes = len(unique_nodes)
    node2idx = {node: idx for idx, node in enumerate(unique_nodes)}

    u_idx = df["u"].map(node2idx).to_numpy(dtype=np.int64)
    v_idx = df["v"].map(node2idx).to_numpy(dtype=np.int64)
    edge_index = torch.tensor(np.vstack((u_idx, v_idx)), dtype=torch.long)

    # ----------------------------------------------------
    # Empirical Transition Frequencies from Training Set
    # count(u -> v) / count(u) per Section 4.1
    # ----------------------------------------------------
    edge_weights = np.zeros(len(u_idx), dtype=np.float32)
    if t_path is not None and Path(t_path).exists():
        log_fn(f"Computing empirical transition frequencies from: {Path(t_path).name}...")
        try:
            t_df = pd.read_parquet(t_path)
            if "split" in t_df.columns:
                t_df = t_df[t_df["split"] == "train"]

            pair_counts = Counter()
            out_counts = Counter()

            for _, group in t_df.groupby("trip_id", sort=False):
                raw_u = group["u"].map(node2idx).dropna().values.astype(np.int64)
                if len(raw_u) < 2:
                    continue
                u_seq = raw_u[:-1]
                v_seq = raw_u[1:]
                for src, dst in zip(u_seq, v_seq):
                    pair_counts[(src, dst)] += 1
                    out_counts[src] += 1

            edge_map = {(u, v): idx for idx, (u, v) in enumerate(zip(u_idx, v_idx))}
            for (src, dst), cnt in pair_counts.items():
                if (src, dst) in edge_map:
                    total_out = max(1, out_counts[src])
                    edge_idx = edge_map[(src, dst)]
                    edge_weights[edge_idx] = float(cnt) / float(total_out)

            log_fn(f"Calculated transition frequencies for {len(pair_counts):,} observed directed transitions.")
        except Exception as e:
            log_fn(f"Warning: Failed to compute empirical transition probabilities ({e}). Using uniform priors.")
            edge_weights.fill(1.0)
    else:
        edge_weights.fill(1.0)

    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

    # ----------------------------------------------------
    # GPE Precomputation via SciPy Compressed Sparse Graph
    # ----------------------------------------------------
    log_fn(f"Precomputing GPE shortest-path distances with SciPy (num_nodes={num_nodes:,}, d_max={d_max})...")
    adj_weights = np.ones(len(u_idx), dtype=np.float32)
    adj_csr = csr_matrix((adj_weights, (u_idx, v_idx)), shape=(num_nodes, num_nodes))

    try:
        # Unweighted shortest paths with depth cutoff limit
        dist_matrix = shortest_path(
            csgraph=adj_csr,
            directed=True,
            unweighted=True,
        )

        # Distances beyond d_max or unreachable (inf) are mapped to d_max + 1
        unreachable_mask = np.isinf(dist_matrix) | (dist_matrix > d_max)
        dist_matrix = np.clip(dist_matrix, 0, d_max + 1)
        dist_matrix[unreachable_mask] = d_max + 1
        global_D = dist_matrix.astype(np.int8)
        np.fill_diagonal(global_D, 0)
    except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
        log_fn(f"MemoryError encountered while allocating GPE distance matrix: {e}")
        raise

    log_fn(
        f"GPE distance matrix created: shape={global_D.shape} | dtype={global_D.dtype} | "
        f"Memory: ~{global_D.nbytes / (1024 ** 3):.2f} GB"
    )

    # ----------------------------------------------------
    # Node Feature Aggregation & Standardization
    # ----------------------------------------------------
    log_fn("Building node features...")
    df_feat = df.copy()
    df_feat["u_idx"] = df_feat["u"].map(node2idx).fillna(0).astype(int)
    df_feat["segment_length"] = df_feat["segment_length"].fillna(0.0).astype(float)
    df_feat["speed_limit"] = df_feat["speed_limit"].fillna(50.0).astype(float)
    df_feat["category"] = df_feat["category"].fillna("unknown").astype(str)

    node_stats = (
        df_feat.groupby("u_idx")
        .agg({
            "segment_length": "mean",
            "speed_limit": "mean",
        })
        .reindex(range(num_nodes), fill_value=0.0)
        .fillna(0.0)
    )

    cat_mode = (
        df_feat.groupby("u_idx")["category"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown")
        .reindex(range(num_nodes), fill_value="unknown")
        .fillna("unknown")
    )

    cat_onehot = pd.get_dummies(cat_mode)

    real_features = torch.tensor(node_stats.values, dtype=torch.float32)
    cat_tensor = torch.tensor(cat_onehot.values, dtype=torch.float32)

    mean = torch.nan_to_num(real_features.mean(dim=0), nan=0.0)
    std = torch.nan_to_num(real_features.std(dim=0), nan=1.0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    real_features = (real_features - mean) / std
    real_features = torch.nan_to_num(real_features, nan=0.0)

    final_node_features = torch.cat([real_features, cat_tensor], dim=1)
    log_fn(
        f"Node features created: {final_node_features.shape} "
        f"({real_features.shape[1]} real + {cat_tensor.shape[1]} categorical)"
    )

    return edge_index, edge_attr, num_nodes, node2idx, final_node_features, global_D


class TrajectoryDataset(Dataset):
    def __init__(self, data_path, node2idx, split="train", logger=None):
        data_path = Path(data_path)
        log_fn = logger.info if logger else print
        log_fn(f"Loading {split} trajectories from {data_path.name}...")
        df = pd.read_parquet(data_path)

        if "split" in df.columns:
            df = df[df["split"] == split]

        self.trips = []
        self.trip_ids = []
        # Preserve file appearance order strictly
        grouped = df.groupby("trip_id", sort=False)
        log_fn(f"Grouping sequences for {split} set ({len(grouped):,} trips)...")

        for trip_id, group in tqdm(grouped, total=len(grouped), leave=False):
            # Shift node indices by NUM_SPECIAL_TOKENS (PAD=0, SOT=1, EOT=2)
            raw_nodes = group["u"].map(node2idx).dropna().values.astype(np.int64)
            if len(raw_nodes) < 2:
                continue

            # Trajectory sequence with SOT and EOT
            token_nodes = np.concatenate(([SOT_TOKEN], raw_nodes + NUM_SPECIAL_TOKENS, [EOT_TOKEN]))
            raw_node_seq = raw_nodes

            spatial_labels = (
                group["spatial_anomaly"].values.astype(np.float32)
                if "spatial_anomaly" in group.columns
                else np.zeros(len(raw_nodes), dtype=np.float32)
            )
            temporal_labels = (
                group["temporal_anomaly"].values.astype(np.float32)
                if "temporal_anomaly" in group.columns
                else np.zeros(len(raw_nodes), dtype=np.float32)
            )

            trip_is_spatial = int((spatial_labels > 0).any())
            trip_is_temporal = int((temporal_labels > 0).any())
            trip_label = int(trip_is_spatial or trip_is_temporal)

            tid_str = str(trip_id)
            self.trips.append({
                "trip_id": tid_str,
                "tokens": token_nodes,
                "raw_nodes": raw_node_seq,
                "label": trip_label,
                "spatial_label": trip_is_spatial,
                "temporal_label": trip_is_temporal,
            })
            self.trip_ids.append(tid_str)

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        item = self.trips[idx]
        return {
            "trip_id": item["trip_id"],
            "tokens": torch.from_numpy(item["tokens"]).long(),
            "raw_nodes": torch.from_numpy(item["raw_nodes"]).long(),
            "label": torch.tensor(item["label"], dtype=torch.int32),
            "spatial_label": torch.tensor(item["spatial_label"], dtype=torch.int32),
            "temporal_label": torch.tensor(item["temporal_label"], dtype=torch.int32),
        }


class GETADCollateFn:
    """
    Top-level callable class for collating batches.
    Preserves strict sample ordering without internal sorting permutations.
    """
    def __init__(self, global_dist_matrix, d_max):
        if isinstance(global_dist_matrix, np.ndarray):
            self.dist_tensor = torch.from_numpy(global_dist_matrix)
        else:
            self.dist_tensor = global_dist_matrix
        self.d_max = d_max

    def __call__(self, batch):
        batch_size = len(batch)
        trip_ids = [item["trip_id"] for item in batch]

        tokens = pad_sequence([item["tokens"] for item in batch], batch_first=True, padding_value=PAD_TOKEN)
        seq_len = tokens.size(1)

        mask = (tokens != PAD_TOKEN)
        lengths = torch.tensor([len(item["tokens"]) for item in batch], dtype=torch.long)

        trip_labels = torch.stack([item["label"] for item in batch])
        spatial_labels = torch.stack([item["spatial_label"] for item in batch])
        temporal_labels = torch.stack([item["temporal_label"] for item in batch])

        # Build GPE matrix for the batch
        # Disconnected or special tokens map to index (d_max + 1)
        gpe_matrix = torch.full((batch_size, seq_len, seq_len), self.d_max + 1, dtype=torch.long)

        for b_idx in range(batch_size):
            raw_nodes = batch[b_idx]["raw_nodes"]
            num_raw = len(raw_nodes)
            if num_raw == 0:
                continue

            grid_i, grid_j = torch.meshgrid(raw_nodes, raw_nodes, indexing="ij")
            dist_sub = self.dist_tensor[grid_i, grid_j].long()

            # Tokens: [SOT (idx 0), raw_node_0 (idx 1), ..., raw_node_{N-1} (idx N), EOT (idx N+1)]
            gpe_matrix[b_idx, 1:1 + num_raw, 1:1 + num_raw] = dist_sub
            # Distance from node to itself is 0
            for pos in range(1, 1 + num_raw):
                gpe_matrix[b_idx, pos, pos] = 0

        return tokens, gpe_matrix, mask, lengths, trip_labels, spatial_labels, temporal_labels, trip_ids


def make_collate_fn(global_dist_matrix, d_max):
    """Backwards-compatible helper returning GETADCollateFn."""
    return GETADCollateFn(global_dist_matrix, d_max)
