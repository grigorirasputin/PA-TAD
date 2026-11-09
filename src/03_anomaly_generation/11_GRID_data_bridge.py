"""
11_GRID_data_bridge.py

Transforms model_ready trajectory data into spatio-temporal grid representations.
Supports:
  1. GM-VSAE / GRID-GRU: Sequential spatial grid cell tokens.
  2. FoTraj: Spatio-temporal triplets (Cell Token, Dwell Duration in Seconds, Entry Timestamp, Anomaly Flags).

Allocates continuous edge travel times into intersected grid cells proportionately 
by sub-segment geometric length, guaranteeing time conservation across cell boundaries
and enforcing 4-connected (Manhattan) contiguity without diagonal corner cutting or gaps.
Guarantees 100% trip preservation without dropping trips.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkb
from shapely.geometry import LineString
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "03_processed"
SYNTH_DIR = BASE_DIR / "data" / "05_synthetic"
MODEL_READY_DIR = BASE_DIR / "data" / "06_model_ready"

# Regional target bounding dimensions to achieve ~100x100 grid cells (~100m cell size)
# Aalborg (~57.05 N): 1 deg lat ~ 111.3 km, 1 deg lon ~ 60.5 km
# Porto (~41.15 N):   1 deg lat ~ 111.1 km, 1 deg lon ~ 83.6 km
REGIONAL_CELL_SPECS = {
    "aalborg": {
        "cell_w": 0.001650,  # ~100m lon
        "cell_h": 0.000900,  # ~100m lat
        "target_grid": (100, 100),
    },
    "porto": {
        "cell_w": 0.001196,  # ~100m lon
        "cell_h": 0.000901,  # ~100m lat
        "target_grid": (100, 100),
    },
    "copenhagen": {
        "cell_w": 0.3800 / 150,  # ~0.002533 deg lon (~158m)
        "cell_h": 0.2900 / 200,  # ~0.001450 deg lat (~161m)
        "target_grid": (150, 200),  # 30,000 tokens (manageable for GM-VSAE & FoTraj)
    },
    "denmark": {
        "cell_w": 0.008000,
        "cell_h": 0.005000,
        "target_grid": None,
    }
}


def parse_wkb_geometry(val):
    if val is None or pd.isna(val):
        return None
    try:
        if isinstance(val, str):
            val = bytes.fromhex(val)
        return wkb.loads(val)
    except Exception:
        return None


def _trace_ray_manhattan(p1, p2, min_lon, min_lat, cell_w, cell_h, grid_m, grid_n):
    """
    Traces a 2D line segment between p1=(x1, y1) and p2=(x2, y2) through a regular grid
    using Amanatides & Woo's Fast Voxel Traversal algorithm.
    Guarantees strictly 4-connected (Manhattan) steps across cell boundaries so that
    no diagonal corner-crossings or skipped intermediate cells occur.
    Returns:
        List of tuples: (cx, cy, length_in_cell)
    """
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    seg_length = float(np.hypot(dx, dy))

    # Cell coordinates
    cx1 = int(np.clip((x1 - min_lon) / cell_w, 0, grid_m - 1))
    cy1 = int(np.clip((y1 - min_lat) / cell_h, 0, grid_n - 1))
    cx2 = int(np.clip((x2 - min_lon) / cell_w, 0, grid_m - 1))
    cy2 = int(np.clip((y2 - min_lat) / cell_h, 0, grid_n - 1))

    if seg_length == 0.0:
        return [(cx1, cy1, 0.0)]

    if cx1 == cx2 and cy1 == cy2:
        return [(cx1, cy1, seg_length)]

    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

    # Next boundary coordinate in parametric t [0, 1]
    if step_x > 0:
        next_boundary_x = min_lon + (cx1 + 1) * cell_w
        t_max_x = (next_boundary_x - x1) / dx
        t_delta_x = cell_w / dx
    elif step_x < 0:
        next_boundary_x = min_lon + cx1 * cell_w
        t_max_x = (next_boundary_x - x1) / dx
        t_delta_x = cell_w / (-dx)
    else:
        t_max_x = float('inf')
        t_delta_x = float('inf')

    if step_y > 0:
        next_boundary_y = min_lat + (cy1 + 1) * cell_h
        t_max_y = (next_boundary_y - y1) / dy
        t_delta_y = cell_h / dy
    elif step_y < 0:
        next_boundary_y = min_lat + cy1 * cell_h
        t_max_y = (next_boundary_y - y1) / dy
        t_delta_y = cell_h / (-dy)
    else:
        t_max_y = float('inf')
        t_delta_y = float('inf')

    cx, cy = cx1, cy1
    curr_t = 0.0
    cells = []
    max_steps = (abs(cx2 - cx1) + abs(cy2 - cy1) + 2) * 2
    step_count = 0

    while step_count < max_steps:
        step_count += 1
        if t_max_x < t_max_y:
            next_t = min(t_max_x, 1.0)
            dist_in_cell = max(0.0, (next_t - curr_t) * seg_length)
            cells.append((cx, cy, dist_in_cell))
            curr_t = next_t
            if cx == cx2 and cy == cy2:
                break
            cx += step_x
            if cx < 0 or cx >= grid_m:
                break
            t_max_x += t_delta_x
        elif t_max_y < t_max_x:
            next_t = min(t_max_y, 1.0)
            dist_in_cell = max(0.0, (next_t - curr_t) * seg_length)
            cells.append((cx, cy, dist_in_cell))
            curr_t = next_t
            if cx == cx2 and cy == cy2:
                break
            cy += step_y
            if cy < 0 or cy >= grid_n:
                break
            t_max_y += t_delta_y
        else:
            # Reached exact corner intersection: t_max_x == t_max_y
            next_t = min(t_max_x, 1.0)
            dist_in_cell = max(0.0, (next_t - curr_t) * seg_length)
            cells.append((cx, cy, dist_in_cell))
            curr_t = next_t
            if cx == cx2 and cy == cy2:
                break
            # Split diagonal into Manhattan 4-connected step via dominant axis first
            if abs(dx) >= abs(dy):
                cx += step_x
                if 0 <= cx < grid_m:
                    cells.append((cx, cy, 0.0))  # zero-duration transit cell to preserve continuity
                cy += step_y
            else:
                cy += step_y
                if 0 <= cy < grid_n:
                    cells.append((cx, cy, 0.0))
                cx += step_x
            if cx < 0 or cx >= grid_m or cy < 0 or cy >= grid_n:
                break
            t_max_x += t_delta_x
            t_max_y += t_delta_y

        if curr_t >= 1.0 - 1e-9:
            break

    if not cells or (cells[-1][0] != cx2 or cells[-1][1] != cy2):
        rem_dist = max(0.0, (1.0 - curr_t) * seg_length)
        cells.append((cx2, cy2, rem_dist))

    return cells


def slice_edge_into_grid(geom, edge_duration, min_lon, min_lat, cell_w, cell_h, grid_m, grid_n):
    """
    Slices a LineString into the grid cells it traverses using exact grid boundary ray-casting.
    Guarantees strict 4-connected Manhattan cell adjacency without skipping cells or
    jumping diagonally through corners.
    Returns:
        List of tuples: (cell_token, cell_duration_seconds)
    """
    if geom is None or geom.is_empty or geom.length == 0:
        return []

    coords = list(geom.coords)
    if len(coords) < 2:
        return []

    raw_traversals = []
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        traced = _trace_ray_manhattan(p1, p2, min_lon, min_lat, cell_w, cell_h, grid_m, grid_n)
        raw_traversals.extend(traced)

    if not raw_traversals:
        return []

    # Merge consecutive identical cells and bridge any accidental non-Manhattan gaps
    sanitized = []
    for cx, cy, d in raw_traversals:
        cx = int(np.clip(cx, 0, grid_m - 1))
        cy = int(np.clip(cy, 0, grid_n - 1))
        if sanitized:
            last_cx, last_cy, last_d = sanitized[-1]
            if last_cx == cx and last_cy == cy:
                sanitized[-1] = (last_cx, last_cy, last_d + d)
                continue

            # Ensure 4-connected Manhattan path between consecutive cells
            step_gap_x = cx - last_cx
            step_gap_y = cy - last_cy
            if abs(step_gap_x) + abs(step_gap_y) > 1:
                cur_x, cur_y = last_cx, last_cy
                while cur_x != cx:
                    cur_x += 1 if step_gap_x > 0 else -1
                    if cur_x != cx or cur_y != cy:
                        sanitized.append((cur_x, cur_y, 0.0))
                while cur_y != cy:
                    cur_y += 1 if step_gap_y > 0 else -1
                    if cur_x != cx or cur_y != cy:
                        sanitized.append((cur_x, cur_y, 0.0))

        sanitized.append((cx, cy, d))

    # Convert to tokens and assign proportional duration
    total_traced_len = sum(d for _, _, d in sanitized)
    result = []
    for cx, cy, d in sanitized:
        token = cy * grid_m + cx
        if total_traced_len > 0:
            frac = d / total_traced_len
            dur = edge_duration * frac
        else:
            dur = edge_duration / len(sanitized)
        result.append((token, dur))

    # Final pass to merge consecutive identical tokens if any intermediate steps collapsed
    merged = []
    for token, dur in result:
        if merged and merged[-1][0] == token:
            merged[-1] = (token, merged[-1][1] + dur)
        else:
            merged.append((token, dur))

    return merged


def bridge_manhattan_gap(prev_token, next_token, grid_m, grid_n):
    """
    Generates intermediate tokens between two non-adjacent tokens such that
    the step follows a strictly 4-connected Manhattan path.
    """
    if prev_token == next_token:
        return []

    c_x1 = prev_token % grid_m
    c_y1 = prev_token // grid_m
    c_x2 = next_token % grid_m
    c_y2 = next_token // grid_m

    intermediates = []
    curr_x, curr_y = c_x1, c_y1

    step_x = 1 if c_x2 > c_x1 else -1
    step_y = 1 if c_y2 > c_y1 else -1

    while curr_x != c_x2:
        curr_x += step_x
        intermediates.append(curr_y * grid_m + curr_x)

    while curr_y != c_y2:
        curr_y += step_y
        intermediates.append(curr_y * grid_m + curr_x)

    if intermediates and intermediates[-1] == next_token:
        intermediates.pop()

    return intermediates


def extract_spatial_subtype(series):
    for val in series:
        if pd.notna(val) and str(val).strip() and str(val).strip().lower() != "none":
            return str(val).strip()
    return "None"


def generate_grid_dataset(df_traj_with_spatial, cell_w, cell_h, dataset_name, output_dir):
    print(f"\n--- Generating Spatio-Temporal Grid Dataset for {dataset_name.upper()} ---")

    bounds = df_traj_with_spatial["geometry"].apply(lambda g: g.bounds if g else None).dropna().tolist()
    if not bounds:
        print(f"Warning: No valid geometries found for {dataset_name}. Skipping grid conversion.")
        return

    min_lon = min(b[0] for b in bounds)
    min_lat = min(b[1] for b in bounds)
    max_lon = max(b[2] for b in bounds) + 1e-5
    max_lat = max(b[3] for b in bounds) + 1e-5

    M = max(1, int(np.ceil((max_lon - min_lon) / cell_w)))
    N = max(1, int(np.ceil((max_lat - min_lat) / cell_h)))
    vocab_size = M * N
    print(f"Canvas: {M} columns x {N} rows | Total Tokens: {vocab_size:,}")

    data_splits = {"train": [], "val": [], "test": []}
    fotraj_splits = {"train": [], "val": [], "test": []}
    metadata_splits = {"train": [], "val": [], "test": []}

    df_sorted = df_traj_with_spatial.sort_values(by=["trip_id", "datetime"], kind="stable")
    grouped = df_sorted.groupby("trip_id", sort=False)

    for trip_id, group in tqdm(grouped, total=len(grouped), desc=f"Rasterizing {dataset_name}"):
        split = group["split"].iloc[0]
        anomaly_class = group["anomaly_class"].iloc[0]
        departure_time = pd.to_datetime(group["datetime"].iloc[0])

        spatial_max = int(group["spatial_anomaly"].max()) if "spatial_anomaly" in group.columns else 0
        temporal_max = int(group["temporal_anomaly"].max()) if "temporal_anomaly" in group.columns else 0
        spatial_subtype = extract_spatial_subtype(group["spatial_subtype"]) if "spatial_subtype" in group.columns else "None"

        trip_tokens = []
        trip_durations = []
        current_time = departure_time
        trip_fotraj_events = []

        for _, row in group.iterrows():
            geom = row["geometry"]
            e_sec = float(row["seconds"])
            spatial_flag = int(row.get("spatial_anomaly", 0))
            temporal_flag = int(row.get("temporal_anomaly", 0))

            slices = slice_edge_into_grid(geom, e_sec, min_lon, min_lat, cell_w, cell_h, M, N)
            for token, cell_sec in slices:
                if trip_tokens:
                    last_tok = trip_tokens[-1]
                    lx, ly = last_tok % M, last_tok // M
                    tx, ty = token % M, token // M
                    if abs(tx - lx) + abs(ty - ly) > 1:
                        fill_tokens = bridge_manhattan_gap(last_tok, token, M, N)
                        for fill_t in fill_tokens:
                            if not trip_tokens or trip_tokens[-1] != fill_t:
                                trip_tokens.append(int(fill_t))
                                trip_durations.append(0.0)
                                trip_fotraj_events.append({
                                    "token": int(fill_t),
                                    "duration_s": 0.0,
                                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "spatial_anomaly": spatial_flag,
                                    "temporal_anomaly": temporal_flag,
                                })

                if not trip_tokens or trip_tokens[-1] != token:
                    trip_tokens.append(int(token))
                    trip_durations.append(cell_sec)
                    trip_fotraj_events.append({
                        "token": int(token),
                        "duration_s": round(cell_sec, 2),
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "spatial_anomaly": spatial_flag,
                        "temporal_anomaly": temporal_flag,
                    })
                else:
                    trip_durations[-1] += cell_sec
                    trip_fotraj_events[-1]["duration_s"] = round(trip_durations[-1], 2)
                    if spatial_flag:
                        trip_fotraj_events[-1]["spatial_anomaly"] = 1
                    if temporal_flag:
                        trip_fotraj_events[-1]["temporal_anomaly"] = 1

                current_time += pd.Timedelta(seconds=cell_sec)

        # Ensure length >= 2 to prevent any short-sequence dropping in baselines
        if len(trip_tokens) == 1:
            tok = trip_tokens[0]
            trip_tokens = [tok, tok]
            half_dur = trip_durations[0] / 2.0
            trip_durations = [half_dur, half_dur]
            if trip_fotraj_events:
                ev1 = trip_fotraj_events[0]
                ev1["duration_s"] = round(half_dur, 2)
                ev2 = dict(ev1)
                ev2_time = pd.to_datetime(ev1["timestamp"]) + pd.Timedelta(seconds=half_dur)
                ev2["timestamp"] = ev2_time.strftime("%Y-%m-%d %H:%M:%S")
                trip_fotraj_events = [ev1, ev2]
        elif len(trip_tokens) == 0:
            trip_tokens = [0, 0]
            trip_durations = [0.0, 0.0]
            trip_fotraj_events = [
                {
                    "token": 0,
                    "duration_s": 0.0,
                    "timestamp": departure_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "spatial_anomaly": spatial_max,
                    "temporal_anomaly": temporal_max,
                },
                {
                    "token": 0,
                    "duration_s": 0.0,
                    "timestamp": departure_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "spatial_anomaly": spatial_max,
                    "temporal_anomaly": temporal_max,
                }
            ]

        data_splits[split].append(trip_tokens)
        fotraj_splits[split].append(trip_fotraj_events)
        metadata_splits[split].append({
            "trip_id": str(trip_id),
            "anomaly_class": str(anomaly_class),
            "spatial_anomaly": int(spatial_max),
            "temporal_anomaly": int(temporal_max),
            "spatial_subtype": str(spatial_subtype),
            "total_time_s": float(sum(trip_durations)),
            "grid_tokens_count": int(len(trip_tokens))
        })

    out_dir = Path(output_dir) / f"grid_data_{dataset_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for s_name in ["train", "val", "test"]:
        if len(data_splits[s_name]) > 0:
            with open(out_dir / f"grid_tokens_{dataset_name}_{s_name}.csv", "w", encoding="utf-8") as f:
                for traj in data_splits[s_name]:
                    f.write(str(traj) + "\n")

            with open(out_dir / f"fotraj_{dataset_name}_{s_name}.jsonl", "w", encoding="utf-8") as f:
                for events in fotraj_splits[s_name]:
                    f.write(json.dumps(events) + "\n")

            pd.DataFrame(metadata_splits[s_name]).to_csv(
                out_dir / f"metadata_{dataset_name}_{s_name}.csv", index=False
            )

    config = {
        "dataset_name": dataset_name,
        "grid_size": {"columns_M": M, "rows_N": N},
        "cell_dimensions": {"width_lon": cell_w, "height_lat": cell_h},
        "vocab_size": vocab_size,
        "bbox": [min_lon, min_lat, max_lon, max_lat]
    }
    with open(out_dir / "grid_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"Exported {dataset_name.upper()} grid data to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="dataset tag, e.g. '2014', '2016_itsp', or 'porto'")
    parser.add_argument("--regions", nargs="+", default=None, help="regions to convert (e.g. 'porto' or 'aalborg')")
    args = parser.parse_args()

    active_regions = args.regions or (["porto"] if "porto" in args.tag.lower() else ["aalborg", "copenhagen", "denmark"])
    model_ready_tag_dir = MODEL_READY_DIR / args.tag

    map_file = PROCESSED_DIR / f"map_directed_{args.tag}.parquet"
    print(f"Loading road network geometries from {map_file}...")
    map_df = pd.read_parquet(map_file, columns=["u", "v", "segmentgeo"])
    map_df["u"] = map_df["u"].astype(np.int64)
    map_df["v"] = map_df["v"].astype(np.int64)
    map_df["geometry"] = map_df["segmentgeo"].apply(parse_wkb_geometry)
    map_spatial = map_df.dropna(subset=["geometry"])[["u", "v", "geometry"]].drop_duplicates(subset=["u", "v"])
    del map_df

    for region in active_regions:
        traj_file = model_ready_tag_dir / f"model_ready_trajectories_{region}.parquet"
        if not traj_file.exists():
            print(f"Trajectory file {traj_file.name} not found. Skipping.")
            continue

        spec = REGIONAL_CELL_SPECS.get(region, REGIONAL_CELL_SPECS["porto"])

        print(f"\nProcessing region {region} with cell size lon={spec['cell_w']:.6f}, lat={spec['cell_h']:.6f}...")
        traj_df = pd.read_parquet(traj_file)
        traj_df["u"] = traj_df["u"].astype(np.int64)
        traj_df["v"] = traj_df["v"].astype(np.int64)

        traj_with_geom = traj_df.merge(map_spatial, on=["u", "v"], how="left")
        missing_geoms = traj_with_geom["geometry"].isna().sum()
        if missing_geoms > 0:
            print(f"Warning: {missing_geoms:,} rows out of {len(traj_with_geom):,} have missing geometry.")

        generate_grid_dataset(
            traj_with_geom,
            cell_w=spec["cell_w"],
            cell_h=spec["cell_h"],
            dataset_name=region,
            output_dir=model_ready_tag_dir
        )

    print("\nGrid data bridge completed successfully.")


if __name__ == "__main__":
    main()
