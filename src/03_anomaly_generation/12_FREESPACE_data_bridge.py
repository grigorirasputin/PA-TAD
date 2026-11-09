"""
Generate continuous free-space trajectory datasets for CeTrajad from synthetic
trajectory pipelines.

Extracts point-level coordinate sequences using edge geometries, guarantees strict
1:1 trip ordering with split definitions from 10_train_test_split_synth.py, and
generates:
  1. Route view: Origin-centered coordinates resampled uniformly every 50m.
  2. Speed view: Point-to-point Haversine speed sequence (m/s).
  3. Shape view: Total-distance-normalized coordinates resampled uniformly every 24s,
     transformed with StandardScaler + PCA(n_components=2) fitted exclusively on train.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_READY_DIR = BASE_DIR / "data" / "06_model_ready"
SYNTH_DIR = BASE_DIR / "data" / "05_synthetic"

EARTH_RADIUS_METERS = 6371008.8


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CeTrajad free-space multi-view trajectory data.")
    parser.add_argument("--tag", required=True, help="dataset pair label, e.g. '2014', 'porto'")
    parser.add_argument("--region", required=True, help="region name, e.g. 'porto', 'aalborg', 'copenhagen'")
    parser.add_argument("--resample_dist_m", type=float, default=50.0, help="route view resampling distance (m)")
    parser.add_argument("--resample_time_s", type=float, default=24.0, help="shape view resampling time interval (s)")
    return parser.parse_args()


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points in meters."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def load_edge_geometries(tag: str, region: str):
    """
    Load segment geometries from model ready or synthetic directory.
    Maps (u, v) -> list of (lon, lat) coordinates.
    """
    candidates = [
        MODEL_READY_DIR / tag / f"model_ready_map_df_{region}.parquet",
        SYNTH_DIR / tag / f"map_df_{region}.parquet",
    ]

    map_path = None
    for cand in candidates:
        if cand.exists():
            map_path = cand
            break

    if map_path is None:
        raise FileNotFoundError(f"Could not find map dataframe for {region} under tag {tag}")

    print(f"Loading edge geometries from: {map_path}")
    map_df = pd.read_parquet(map_path, columns=["u", "v", "segmentgeo"])
    map_df["u"] = map_df["u"].astype(np.int64)
    map_df["v"] = map_df["v"].astype(np.int64)
    map_df = map_df.drop_duplicates(subset=["u", "v"], keep="first")

    edge_coords = {}
    for row in map_df.itertuples(index=False):
        val = row.segmentgeo
        if val is None or pd.isna(val):
            continue
        try:
            geom = wkb.loads(bytes.fromhex(val)) if isinstance(val, str) else wkb.loads(val)
        except Exception:
            continue

        if geom is None or geom.is_empty:
            continue

        coords = list(geom.coords)
        if len(coords) >= 2:
            edge_coords[(int(row.u), int(row.v))] = coords

    print(f"Extracted coordinate sequences for {len(edge_coords):,} edges.")
    return edge_coords


def reconstruct_trip_points(trip_df, edge_coords):
    """
    Reconstruct continuous (lat, lon, cumulative_seconds) trajectory points
    by traversing each edge linestring.
    """
    lats = []
    lons = []
    cum_times = []

    current_t = 0.0

    for i, row in enumerate(trip_df.itertuples(index=False)):
        key = (int(row.u), int(row.v))
        edge_pts = edge_coords.get(key)
        duration = float(row.seconds)

        if edge_pts is None or len(edge_pts) < 2:
            continue

        coords_arr = np.asarray(edge_pts, dtype=np.float64)
        edge_lons = coords_arr[:, 0]
        edge_lats = coords_arr[:, 1]

        dists = haversine_distance_m(edge_lats[:-1], edge_lons[:-1], edge_lats[1:], edge_lons[1:])
        cum_dist = np.concatenate(([0.0], np.cumsum(dists)))
        total_len = cum_dist[-1]

        if total_len > 0:
            edge_times = current_t + (cum_dist / total_len) * duration
        else:
            edge_times = current_t + np.linspace(0.0, duration, len(edge_pts))

        start_idx = 1 if len(lats) > 0 else 0
        lats.extend(edge_lats[start_idx:].tolist())
        lons.extend(edge_lons[start_idx:].tolist())
        cum_times.extend(edge_times[start_idx:].tolist())

        current_t += duration

    # Fallback if edge geometries could not be loaded: use dummy points to avoid dropping trips
    if len(lats) < 2:
        lats = [0.0, 0.0001]
        lons = [0.0, 0.0001]
        cum_times = [0.0, max(current_t, 1.0)]

    return (
        np.asarray(lats, dtype=np.float64),
        np.asarray(lons, dtype=np.float64),
        np.asarray(cum_times, dtype=np.float64),
    )


def compute_route_view(lats, lons, resample_dist_m=50.0):
    """
    Route View (input_size=2):
      - Origin-centered coordinates (dlat, dlon).
      - Resampled at uniform cumulative distance steps (resample_dist_m).
    """
    dlat = lats - lats[0]
    dlon = lons - lons[0]

    step_dists = haversine_distance_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    cum_s = np.concatenate(([0.0], np.cumsum(step_dists)))

    total_s = cum_s[-1]
    if total_s < resample_dist_m:
        target_s = np.array([0.0, max(total_s, 1e-3)], dtype=np.float64)
    else:
        target_s = np.arange(0.0, total_s, resample_dist_m, dtype=np.float64)
        if target_s[-1] < total_s:
            target_s = np.append(target_s, total_s)

    unique_mask = np.concatenate(([True], np.diff(cum_s) > 1e-6))
    cum_s_clean = cum_s[unique_mask]
    dlat_clean = dlat[unique_mask]
    dlon_clean = dlon[unique_mask]

    if len(cum_s_clean) < 2:
        interp_lat = np.repeat(dlat[0], len(target_s))
        interp_lon = np.repeat(dlon[0], len(target_s))
    else:
        interp_lat = np.interp(target_s, cum_s_clean, dlat_clean)
        interp_lon = np.interp(target_s, cum_s_clean, dlon_clean)

    route_features = np.column_stack((interp_lat, interp_lon)).astype(np.float32)
    return route_features, total_s


def compute_speed_view(lats, lons, times):
    """
    Speed View (input_size=1):
      - Point-to-point speed: displacement / dt.
    """
    step_dists = haversine_distance_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
    dt = np.diff(times)
    dt = np.maximum(dt, 1e-3)

    speeds = step_dists / dt
    all_speeds = np.concatenate(([speeds[0]], speeds)).astype(np.float32)
    return all_speeds.reshape(-1, 1)


def compute_shape_raw(lats, lons, times, total_s, resample_time_s=24.0):
    """
    Shape View raw representation:
      - Origin-centered coordinates.
      - Normalized by total cumulative distance S_total.
      - Resampled at uniform time intervals (resample_time_s).
    """
    scale = max(float(total_s), 1.0)
    dlat_norm = (lats - lats[0]) / scale
    dlon_norm = (lons - lons[0]) / scale

    total_time = times[-1]
    if total_time < resample_time_s:
        target_t = np.array([0.0, max(total_time, 1e-3)], dtype=np.float64)
    else:
        target_t = np.arange(0.0, total_time, resample_time_s, dtype=np.float64)
        if target_t[-1] < total_time:
            target_t = np.append(target_t, total_time)

    unique_mask = np.concatenate(([True], np.diff(times) > 1e-6))
    times_clean = times[unique_mask]
    lat_clean = dlat_norm[unique_mask]
    lon_clean = dlon_norm[unique_mask]

    if len(times_clean) < 2:
        interp_lat = np.repeat(dlat_norm[0], len(target_t))
        interp_lon = np.repeat(dlon_norm[0], len(target_t))
    else:
        interp_lat = np.interp(target_t, times_clean, lat_clean)
        interp_lon = np.interp(target_t, times_clean, lon_clean)

    return np.column_stack((interp_lat, interp_lon)).astype(np.float32)


def extract_spatial_subtype(series):
    for val in series:
        if pd.notna(val) and str(val).strip() and str(val).strip().lower() != "none":
            return str(val).strip()
    return "None"


def main():
    args = parse_args()
    tag = args.tag
    region = args.region.lower()

    output_dir = MODEL_READY_DIR / tag / f"freespace_data_{region}"
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_path = MODEL_READY_DIR / tag / f"model_ready_trajectories_{region}.parquet"
    if not traj_path.exists():
        raise FileNotFoundError(f"Model-ready trajectories not found at {traj_path}. Run 10_train_test_split_synth.py first.")

    print(f"Loading trajectories from {traj_path}")
    traj_df = pd.read_parquet(traj_path)

    # Edge geometries for reconstruction
    edge_coords = load_edge_geometries(tag, region)

    splits = ["train", "val", "test"]
    grouped = dict(tuple(traj_df.groupby("trip_id", sort=False)))

    # Unique trip order per split preserving exact original order from traj_df
    trips_by_split = {
        s: traj_df[traj_df["split"] == s]["trip_id"].drop_duplicates().tolist()
        for s in splits
    }

    route_dict = {s: {} for s in splits}
    speed_dict = {s: {} for s in splits}
    shape_raw_dict = {s: {} for s in splits}

    print("Reconstructing continuous trajectories and calculating views...")
    for split in splits:
        trip_ids = trips_by_split[split]
        for tid in tqdm(trip_ids, desc=f"Processing {split} trips"):
            trip_data = grouped[tid]
            res = reconstruct_trip_points(trip_data, edge_coords)

            lats, lons, times = res
            route_feat, total_s = compute_route_view(lats, lons, resample_dist_m=args.resample_dist_m)
            speed_feat = compute_speed_view(lats, lons, times)
            shape_raw = compute_shape_raw(lats, lons, times, total_s, resample_time_s=args.resample_time_s)

            route_dict[split][tid] = route_feat
            speed_dict[split][tid] = speed_feat
            shape_raw_dict[split][tid] = shape_raw

    # Fit StandardScaler and PCA strictly on train split
    print("\nFitting StandardScaler and PCA(n_components=2) on train shape view...")
    train_shape_matrices = list(shape_raw_dict["train"].values())
    if not train_shape_matrices:
        raise RuntimeError("No valid trajectories reconstructed in the train split!")

    train_shape_stacked = np.concatenate(train_shape_matrices, axis=0)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_shape_stacked)

    pca = PCA(n_components=2, random_state=42)
    pca.fit(train_scaled)

    # Save fitted transformer pipeline
    pca_pipeline = {"scaler": scaler, "pca": pca}
    with open(output_dir / "shape_pca_model.pkl", "wb") as f:
        pickle.dump(pca_pipeline, f)
    print(f"Saved fitted PCA pipeline to {output_dir / 'shape_pca_model.pkl'}")

    # Transform all splits using fitted scaler and PCA
    shape_dict = {s: {} for s in splits}
    for split in splits:
        for tid in trips_by_split[split]:
            raw_arr = shape_raw_dict[split][tid]
            scaled = scaler.transform(raw_arr)
            transformed = pca.transform(scaled).astype(np.float32)
            shape_dict[split][tid] = transformed

    # Persist multi-view pickle datasets
    for split in splits:
        with open(output_dir / f"route_{split}.pkl", "wb") as f:
            pickle.dump(route_dict[split], f)
        with open(output_dir / f"speed_{split}.pkl", "wb") as f:
            pickle.dump(speed_dict[split], f)
        with open(output_dir / f"shape_{split}.pkl", "wb") as f:
            pickle.dump(shape_dict[split], f)
        print(f"Saved {split} views: {len(route_dict[split]):,} trips.")

    # Export metadata_test.csv with complete aggregations, preserving exact test order
    test_meta_rows = []
    test_trip_ids = trips_by_split["test"]

    for tid in test_trip_ids:
        group = grouped[tid]
        anomaly_class = group["anomaly_class"].iloc[0]
        spatial_anomaly = int(group["spatial_anomaly"].max()) if "spatial_anomaly" in group.columns else 0
        temporal_anomaly = int(group["temporal_anomaly"].max()) if "temporal_anomaly" in group.columns else 0
        spatial_subtype = extract_spatial_subtype(group["spatial_subtype"]) if "spatial_subtype" in group.columns else "None"
        label = 0 if str(anomaly_class).lower() == "normal" else 1

        test_meta_rows.append({
            "trip_id": str(tid),
            "anomaly_class": str(anomaly_class),
            "spatial_anomaly": spatial_anomaly,
            "temporal_anomaly": temporal_anomaly,
            "spatial_subtype": str(spatial_subtype),
            "label": label,
        })

    metadata_test_df = pd.DataFrame(test_meta_rows)
    metadata_test_path = output_dir / "metadata_test.csv"
    metadata_test_df.to_csv(metadata_test_path, index=False)
    print(f"Saved metadata test ground truth with {len(metadata_test_df):,} trips to {metadata_test_path}")


if __name__ == "__main__":
    main()
