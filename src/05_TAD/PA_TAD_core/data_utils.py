import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

ANOMALY_CLASS_MAP = {
    "normal": 0,
    "spatial": 1,
    "temporal": 2,
    "spatio_temporal": 3,
}


def extract_spatial_subtype(series):
    for val in series:
        if pd.notna(val) and str(val).strip() and str(val).strip().lower() != "none":
            return str(val).strip()
    return "None"


def extract_linestring_geometry_features(df):
    """
    Extracts geometric centroid (lon, lat) and directional azimuth (bearing sin/cos)
    from segment geometries.

    Incorporating directional azimuth directly resolves the cold-start struggle
    with overshoots (where vehicles drive past their destination, U-turn, and re-traverse),
    because reversal of traversal direction presents a ~180 degree heading inversion.
    """
    geom_col = None
    for cand in ["segmentgeo", "geometry", "geom", "wkb"]:
        if cand in df.columns:
            geom_col = cand
            break

    n = len(df)
    lons = np.zeros(n, dtype=np.float32)
    lats = np.zeros(n, dtype=np.float32)
    sin_bearing = np.zeros(n, dtype=np.float32)
    cos_bearing = np.zeros(n, dtype=np.float32)

    # Check if segangle is already precomputed from extraction pipeline
    has_segangle = "segangle" in df.columns

    if geom_col is None:
        lon_c = pd.to_numeric(df["lon"], errors="coerce") if "lon" in df.columns else pd.Series(0.0, index=df.index)
        lat_c = pd.to_numeric(df["lat"], errors="coerce") if "lat" in df.columns else pd.Series(0.0, index=df.index)
        lons = lon_c.fillna(0.0).values.astype(np.float32)
        lats = lat_c.fillna(0.0).values.astype(np.float32)
        if has_segangle:
            rad = np.radians(pd.to_numeric(df["segangle"], errors="coerce").fillna(0.0).values)
            sin_bearing = np.sin(rad).astype(np.float32)
            cos_bearing = np.cos(rad).astype(np.float32)
        return lons, lats, sin_bearing, cos_bearing

    try:
        from shapely import wkb as shapely_wkb
        from shapely.geometry.base import BaseGeometry
        has_shapely = True
    except ImportError:
        has_shapely = False

    raw_geoms = df[geom_col].values
    for idx, val in enumerate(raw_geoms):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        try:
            if has_shapely:
                geom_obj = None
                if isinstance(val, BaseGeometry):
                    geom_obj = val
                elif isinstance(val, (bytes, bytearray)):
                    geom_obj = shapely_wkb.loads(val)
                elif isinstance(val, str):
                    val_s = val.strip()
                    if val_s.startswith("00") or val_s.startswith("01"):
                        geom_obj = shapely_wkb.loads(bytes.fromhex(val_s))
                    elif "(" in val_s:
                        from shapely import wkt
                        geom_obj = wkt.loads(val_s)

                if geom_obj is not None and not geom_obj.is_empty:
                    c = geom_obj.centroid
                    lons[idx] = c.x
                    lats[idx] = c.y

                    if not has_segangle and hasattr(geom_obj, "coords") and len(geom_obj.coords) >= 2:
                        coords = list(geom_obj.coords)
                        x1, y1 = coords[0][0], coords[0][1]
                        x2, y2 = coords[-1][0], coords[-1][1]
                        dx = x2 - x1
                        dy = y2 - y1
                        angle = math.atan2(dy, dx)
                        sin_bearing[idx] = math.sin(angle)
                        cos_bearing[idx] = math.cos(angle)
                    continue
        except Exception:
            pass

        if isinstance(val, (list, tuple)) and len(val) > 0:
            arr = np.array(val)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                lons[idx] = np.mean(arr[:, 0])
                lats[idx] = np.mean(arr[:, 1])
                if not has_segangle and len(arr) >= 2:
                    dx = arr[-1, 0] - arr[0, 0]
                    dy = arr[-1, 1] - arr[0, 1]
                    angle = math.atan2(dy, dx)
                    sin_bearing[idx] = math.sin(angle)
                    cos_bearing[idx] = math.cos(angle)

    if has_segangle:
        rad = np.radians(pd.to_numeric(df["segangle"], errors="coerce").fillna(0.0).values)
        sin_bearing = np.sin(rad).astype(np.float32)
        cos_bearing = np.cos(rad).astype(np.float32)

    return lons, lats, sin_bearing, cos_bearing


def build_graph_and_mapping(map_df_path):
    """
    Builds the PyTorch Geometric graph edge_index, node mappings, and normalized node feature matrix.
    Includes segment length, speed limit, centroid lon/lat, and directional angle components.
    """
    df = pd.read_parquet(map_df_path)
    nodes = np.unique(df[["u", "v"]].values)
    num_nodes, node2idx = len(nodes), {n: i for i, n in enumerate(nodes)}

    edge_index = torch.tensor(
        np.array([df["u"].map(node2idx).values, df["v"].map(node2idx).values]),
        dtype=torch.long,
    )

    c_lon, c_lat, s_bear, c_bear = extract_linestring_geometry_features(df)
    df["centroid_lon"] = c_lon
    df["centroid_lat"] = c_lat
    df["bearing_sin"] = s_bear
    df["bearing_cos"] = c_bear

    cat_col = "category" if "category" in df.columns else ("highway" if "highway" in df.columns else None)
    df["category_code"] = df[cat_col].fillna("unknown").astype("category").cat.codes if cat_col else 0

    length_col = "segment_length" if "segment_length" in df.columns else ("length" if "length" in df.columns else None)
    speed_col = "speed_limit" if "speed_limit" in df.columns else ("maxspeed" if "maxspeed" in df.columns else None)

    df["agg_length"] = pd.to_numeric(df[length_col], errors="coerce").fillna(50.0) if length_col else 50.0
    df["agg_speed"] = pd.to_numeric(df[speed_col], errors="coerce").fillna(50.0) if speed_col else 50.0

    stats = df.groupby(df["u"].map(node2idx)).agg({
        "agg_length": "mean",
        "agg_speed": "mean",
        "centroid_lon": "mean",
        "centroid_lat": "mean",
        "bearing_sin": "mean",
        "bearing_cos": "mean",
        "category_code": "first",
    }).reindex(range(num_nodes), fill_value=0)

    geo_real = stats[["agg_length", "agg_speed", "centroid_lon", "centroid_lat", "bearing_sin", "bearing_cos"]].values
    real_feats = torch.tensor(geo_real, dtype=torch.float32)
    real_feats = (real_feats - real_feats.mean(dim=0)) / (real_feats.std(dim=0) + 1e-6)

    num_classes = int(df["category_code"].max()) + 1
    cat_feats = F.one_hot(torch.tensor(stats["category_code"].values, dtype=torch.long), num_classes=num_classes).float()

    return edge_index, num_nodes, node2idx, torch.cat([real_feats, cat_feats], dim=1)


class TrajectoryDataset(Dataset):
    def __init__(self, path, node2idx, split="train"):
        df = pd.read_parquet(path)

        if "split" in df.columns:
            df = df[df["split"] == split].copy()
        else:
            all_trips = df["trip_id"].drop_duplicates().values
            rng = np.random.default_rng(42)
            perm = rng.permutation(all_trips)
            n_trips = len(perm)
            n_train = int(0.70 * n_trips)
            n_val = int(0.15 * n_trips)
            if split == "train":
                chosen = set(perm[:n_train])
            elif split == "val":
                chosen = set(perm[n_train:n_train + n_val])
            else:
                chosen = set(perm[n_train + n_val:])
            df = df[df["trip_id"].isin(chosen)].copy()

        self.trips = []
        if len(df) == 0:
            return

        df["n_idx"] = df["u"].map(node2idx).fillna(0).astype(np.int64)
        df["log_secs"] = np.log1p(np.maximum(0.0, df["seconds"].values.astype(np.float32)))

        if "timestamp" in df.columns:
            dt_col = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        elif "datetime" in df.columns:
            dt_col = pd.to_datetime(df["datetime"], errors="coerce")
        else:
            dt_col = pd.to_datetime("2024-01-01 08:00:00") + pd.to_timedelta(np.arange(len(df)), unit="s")

        if dt_col.isna().any():
            dt_col = dt_col.fillna(pd.to_datetime("2024-01-01 08:00:00"))

        hour_of_week = (
            dt_col.dt.dayofweek * 24.0
            + dt_col.dt.hour
            + (dt_col.dt.minute / 60.0)
            + (dt_col.dt.second / 3600.0)
        ).values

        two_pi = 2.0 * np.pi
        time_arr = np.column_stack([
            np.sin(two_pi * hour_of_week / 168.0), np.cos(two_pi * hour_of_week / 168.0),
            np.sin(two_pi * hour_of_week / 24.0),  np.cos(two_pi * hour_of_week / 24.0),
            np.sin(two_pi * hour_of_week / 12.0),  np.cos(two_pi * hour_of_week / 12.0),
        ]).astype(np.float32)

        if "spatial_subtype" in df.columns:
            s = df["spatial_subtype"].astype(str).str.upper()
            df["sub_lbl"] = np.select(
                [s.str.contains("SPRW|1"), s.str.contains("DETOUR|LOOP|2"), s.str.contains("OVERSHOOT|3")],
                [1, 2, 3],
                default=0,
            )
        else:
            df["sub_lbl"] = np.zeros(len(df), dtype=np.int64)

        if "anomaly_class" in df.columns:
            df["class_code"] = df["anomaly_class"].map(ANOMALY_CLASS_MAP).fillna(0).astype(np.int64)
        else:
            s_anom = df["spatial_anomaly"].values if "spatial_anomaly" in df.columns else 0
            t_anom = df["temporal_anomaly"].values if "temporal_anomaly" in df.columns else 0
            df["class_code"] = np.select(
                [(s_anom > 0) & (t_anom > 0), (s_anom > 0), (t_anom > 0)],
                [3, 1, 2],
                default=0,
            )

        n_idx_arr = df["n_idx"].values
        secs_arr = df["log_secs"].values
        s_lbl_arr = df["spatial_anomaly"].values.astype(np.float32) if "spatial_anomaly" in df.columns else np.zeros(len(df), dtype=np.float32)
        t_lbl_arr = df["temporal_anomaly"].values.astype(np.float32) if "temporal_anomaly" in df.columns else np.zeros(len(df), dtype=np.float32)
        sub_arr = df["sub_lbl"].values
        class_arr = df["class_code"].values

        for tid, indices in tqdm(df.groupby("trip_id", sort=False).indices.items(), desc=f"Loading {split}", leave=False):
            if len(indices) < 2:
                continue
            self.trips.append({
                "trip_id": str(tid),
                "nodes": torch.tensor(n_idx_arr[indices], dtype=torch.long),
                "seconds": torch.tensor(secs_arr[indices], dtype=torch.float32).unsqueeze(1),
                "time_feats": torch.tensor(time_arr[indices], dtype=torch.float32),
                "s_lbl": torch.tensor(s_lbl_arr[indices], dtype=torch.float32),
                "t_lbl": torch.tensor(t_lbl_arr[indices], dtype=torch.float32),
                "sub_lbl": torch.tensor(sub_arr[indices], dtype=torch.long),
                "cls_lbl": torch.tensor(int(class_arr[indices[0]]), dtype=torch.long),
            })

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        return self.trips[idx]


def collate_fn(batch):
    batch.sort(key=lambda x: len(x["nodes"]), reverse=True)
    keys = ["nodes", "seconds", "time_feats", "s_lbl", "t_lbl", "sub_lbl"]
    padded = [pad_sequence([b[k] for b in batch], batch_first=True) for k in keys]
    mask = pad_sequence([torch.ones(len(b["nodes"])) for b in batch], batch_first=True).bool()
    cls_labels = torch.stack([b["cls_lbl"] for b in batch], dim=0)
    trip_ids = [b["trip_id"] for b in batch]
    return *padded, mask, torch.tensor([len(b["nodes"]) for b in batch]), cls_labels, trip_ids
