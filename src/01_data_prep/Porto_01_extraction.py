import gc
import logging
import math
import os
import pickle
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    import ujson as json_parser
except ImportError:
    try:
        import orjson

        class _OrjsonWrapper:
            @staticmethod
            def loads(s):
                return orjson.loads(s)

            @staticmethod
            def dumps(obj):
                return orjson.dumps(obj).decode("utf-8")

        json_parser = _OrjsonWrapper()
    except ImportError:
        import json as json_parser

import osmnx as ox
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pa_ds
import requests
from shapely import wkb
from shapely.geometry import LineString, Point, box
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "01_raw"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"porto_extraction_{_run_stamp}.log"

_logger = logging.getLogger("porto_extraction")
_logger.setLevel(logging.INFO)
_logger.propagate = False
_stream_handler = logging.StreamHandler(sys.stdout)
_file_handler = logging.FileHandler(LOG_PATH, mode="a")
for _h in (_stream_handler, _file_handler):
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(_h)


def _log(msg):
    _logger.info(msg)
    _file_handler.flush()


_log(f"Logging to console and to: {LOG_PATH}")
traj_path = os.path.join(RAW_DATA_DIR, "train.csv")

_log(f"Reading {traj_path} ...")
_t0 = time.time()
df = pd.read_csv(traj_path)
_log(f"Loaded {len(df):,} rows in {time.time() - _t0:.1f}s")

# ==========================================
# 1. LOAD OSM DATA & PREPARE INDEXES
# ==========================================
ox.settings.log_console = False
ox.settings.requests_timeout = 300

PORTO_BBOX = {
    "north": 41.2600,
    "south": 41.1000,
    "east": -8.5300,
    "west": -8.7200,
}

GRAPH_BBOX = {
    "min_lat": PORTO_BBOX["south"],
    "max_lat": PORTO_BBOX["north"],
    "min_lon": PORTO_BBOX["west"],
    "max_lon": PORTO_BBOX["east"],
}

GRAPH_CACHE_PATH = RAW_DATA_DIR / "porto_osmnx_graph.pkl"
map_extracted_path = os.path.join(RAW_DATA_DIR, "porto_map_extracted.parquet")

edge_categories = {}
edge_oneways = {}
edge_maxspeeds = {}
edge_geometries = {}
edge_lengths = {}
node_coords = {}

if os.path.exists(map_extracted_path):
    _log(f"Found existing map at {map_extracted_path}. Pre-indexing known road network...")
    try:
        prior_map_df = pd.read_parquet(map_extracted_path)
        for _, r in prior_map_df.iterrows():
            u = int(r["startpoint"])
            v = int(r["endpoint"])
            seg_k = str(r["segmentkey"])
            cat = str(r.get("category", "residential"))
            edge_categories[(u, v)] = cat
            edge_oneways[seg_k] = str(r.get("direction", "BOTH"))
            spd = r.get("speedlimit_forward")
            if spd is not None and not pd.isna(spd):
                edge_maxspeeds[(u, v)] = float(spd)
            geom_raw = r.get("segmentgeo")
            if geom_raw is not None:
                try:
                    g = wkb.loads(bytes(geom_raw)) if not isinstance(geom_raw, str) else wkb.loads(geom_raw, hex=True)
                    coords = list(g.coords)
                    if coords:
                        node_coords[u] = (float(coords[0][0]), float(coords[0][1]))
                        node_coords[v] = (float(coords[-1][0]), float(coords[-1][1]))
                        edge_geometries[(u, v)] = g
                except Exception:
                    pass
        _log(f"Pre-indexed {len(node_coords):,} nodes and {len(edge_categories):,} directed edges from local map.")
    except Exception as e:
        _log(f"Could not parse existing map: {e}")

if GRAPH_CACHE_PATH.exists():
    _log(f"Loading cached Porto OSMnx graph from {GRAPH_CACHE_PATH}...")
    try:
        with open(GRAPH_CACHE_PATH, "rb") as f:
            G = pickle.load(f)
        for n, d in G.nodes(data=True):
            node_coords[n] = (float(d["x"]), float(d["y"]))
        for u, v, d in G.edges(data=True):
            h = d.get("highway", "residential")
            cat = h[0] if isinstance(h, list) else h
            edge_categories[(u, v)] = cat
            edge_oneways[f"{u}_{v}"] = "FORWARD" if d.get("oneway", False) else "BOTH"
            if "geometry" in d:
                edge_geometries[(u, v)] = d["geometry"]
            if "length" in d:
                edge_lengths[(u, v)] = float(d["length"])
            if "maxspeed" in d:
                ms = d["maxspeed"]
                if isinstance(ms, list):
                    ms = ms[0]
                try:
                    edge_maxspeeds[(u, v)] = float(str(ms).split()[0])
                except (ValueError, TypeError):
                    pass
        _log(f"OSMnx cache loaded ({len(G.nodes):,} nodes, {len(G.edges):,} edges).")
    except Exception as e:
        _log(f"Could not load graph cache: {e}. Relying on OSRM dynamic geometries.")

DEFAULT_SPEED_BY_CAT = {
    "motorway": 100.0,
    "motorway_link": 60.0,
    "trunk": 90.0,
    "trunk_link": 50.0,
    "primary": 60.0,
    "primary_link": 40.0,
    "secondary": 50.0,
    "secondary_link": 40.0,
    "tertiary": 50.0,
    "tertiary_link": 30.0,
    "residential": 40.0,
    "living_street": 20.0,
    "unclassified": 40.0,
    "service": 20.0,
}


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_bearing(lon1, lat1, lon2, lat2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.cos(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    return round((math.degrees(math.atan2(y, x)) + 360) % 360, 2)


def _haversine_m(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


def _dedupe_stationary_points(coords_list, min_move_deg=3e-5):
    kept = [(0, coords_list[0])]
    for idx in range(1, len(coords_list)):
        lon, lat = coords_list[idx]
        plon, plat = kept[-1][1]
        if abs(lon - plon) < min_move_deg and abs(lat - plat) < min_move_deg:
            continue
        kept.append((idx, coords_list[idx]))
    return kept


def _trip_prefilter_flags(coords_list, min_trip_points=4, max_trip_duration_s=7200, bbox=None):
    n = len(coords_list)
    if n < min_trip_points:
        return False, "too_few_points"
    if 15 * (n - 1) > max_trip_duration_s:
        return False, "trip_too_long"
    if bbox is not None:
        min_lon, max_lon = bbox["min_lon"], bbox["max_lon"]
        min_lat, max_lat = bbox["min_lat"], bbox["max_lat"]
        for lon, lat in coords_list:
            if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                return False, "outside_bbox"
    return True, None


def _build_session(pool_size=16):
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=0,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _fetch_match(session, base_url, coords_str, ts_str, radiuses_str, timeout=30, max_retries=3):
    url = f"{base_url}/match/v1/driving/{coords_str}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "annotations": "nodes,duration,distance",
        "timestamps": ts_str,
        "radiuses": radiuses_str,
    }
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait_time = (2 ** attempt) * 2.0 + random.uniform(0.5, 1.5)
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait_time = max(wait_time, float(retry_after))
                    except ValueError:
                        pass
                time.sleep(wait_time)
                continue
            if resp.status_code != 200:
                time.sleep(1.0 + attempt * 0.5)
                continue
            return resp.json()
        except (requests.exceptions.RequestException, ConnectionError, TimeoutError):
            if attempt == max_retries:
                return None
            time.sleep(1.5 * (2 ** attempt))
    return None


# ==========================================
# 3. EXTRACTION PIPELINE
# ==========================================
def process_porto_to_pipeline(
    df_sample,
    traj_output_path,
    map_output_path,
    base_url=os.getenv("OSRM_URL", "http://localhost:5000"),
    wkb_hex=True,
    flush_every_trips=10000,
    match_radius_m=25.0,
    max_edge_speed_kmh=120.0,
    min_edge_speed_kmh=1.0,
    max_workers=4,
    fetch_batch_size=300,
    max_retries=3,
    request_timeout=30,
    check_bbox=True,
    bbox=None,
    map_flush_every_n_flushes=5,
):
    session = _build_session(pool_size=max(max_workers * 2, 16))

    parts_dir = Path(traj_output_path)
    parts_dir.mkdir(parents=True, exist_ok=True)

    for stray_tmp in parts_dir.glob("*.tmp"):
        stray_tmp.unlink()
    stray_map_tmp = Path(f"{map_output_path}.tmp")
    if stray_map_tmp.exists():
        stray_map_tmp.unlink()

    existing_parts = sorted(parts_dir.glob("part_*.parquet"))
    part_counter = 0
    already_done_trip_ids = set()
    if existing_parts:
        for part in existing_parts:
            try:
                num = int(part.stem.split("_")[1])
                part_counter = max(part_counter, num + 1)
            except (IndexError, ValueError):
                pass
            try:
                already_done_trip_ids.update(
                    pd.read_parquet(part, columns=["trip_id"])["trip_id"].unique()
                )
            except Exception:
                pass
        _log(f"Resuming: found {len(already_done_trip_ids):,} existing trip IDs in parts.")
        base_done = {str(tid).split("_sub_")[0] for tid in already_done_trip_ids}
        df_sample = df_sample[~df_sample["TRIP_ID"].astype(str).isin(base_done)]
        _log(f"{len(df_sample):,} trips remaining to process.")

    map_dict = {}
    if os.path.exists(map_output_path):
        try:
            prior_map = pd.read_parquet(map_output_path)
            map_dict = {row["directed_edge_id"]: row.to_dict() for _, row in prior_map.iterrows()}
            del prior_map
            gc.collect()
        except Exception:
            pass

    map_cols = [
        "directed_edge_id", "segmentkey", "direction", "startpoint", "endpoint",
        "u", "v", "category", "segangle", "meters", "segmentid", "segmentgeo", "speedlimit_forward",
    ]
    map_field_types = {
        "directed_edge_id": pa.string(), "segmentkey": pa.string(),
        "direction": pa.string(), "startpoint": pa.int64(), "endpoint": pa.int64(),
        "u": pa.int64(), "v": pa.int64(), "category": pa.string(),
        "segangle": pa.float64(), "meters": pa.float64(), "segmentid": pa.int64(),
        "segmentgeo": pa.binary(), "speedlimit_forward": pa.float64(),
    }
    map_schema = pa.schema([(c, map_field_types[c]) for c in map_cols])

    def _write_map_checkpoint():
        values = list(map_dict.values())
        if not values:
            return
        columns = {c: [v.get(c) for v in values] for c in map_cols}
        table = pa.table(
            {c: pa.array(columns[c], type=map_field_types[c]) for c in map_cols},
            schema=map_schema,
        )
        del values, columns
        tmp_path = f"{map_output_path}.tmp"
        pq.write_table(table, tmp_path, compression="snappy")
        os.replace(tmp_path, map_output_path)
        del table
        gc.collect()

    col_id = []
    col_trip_id = []
    col_directed_edge_id = []
    col_segmentkey = []
    col_direction = []
    col_trip_segmentno = []
    col_seconds = []
    col_timekey = []
    col_datekey = []
    col_edges_in_leg = []

    global_obs_id = 0
    trips_since_flush = 0
    flushes_since_map_write = 0
    n_fetch_failed = 0
    n_trips_processed = 0

    traj_schema = pa.schema([
        ("id", pa.int64()),
        ("trip_id", pa.string()),
        ("directed_edge_id", pa.string()),
        ("segmentkey", pa.string()),
        ("direction", pa.string()),
        ("trip_segmentno", pa.int32()),
        ("seconds", pa.float64()),
        ("timekey", pa.int32()),
        ("datekey", pa.int32()),
        ("edges_in_leg", pa.int32()),
    ])

    def _flush_traj_part():
        nonlocal part_counter
        if not col_id:
            return
        chunk_table = pa.Table.from_arrays(
            [
                pa.array(col_id, type=pa.int64()),
                pa.array(col_trip_id, type=pa.string()),
                pa.array(col_directed_edge_id, type=pa.string()),
                pa.array(col_segmentkey, type=pa.string()),
                pa.array(col_direction, type=pa.string()),
                pa.array(col_trip_segmentno, type=pa.int32()),
                pa.array(col_seconds, type=pa.float64()),
                pa.array(col_timekey, type=pa.int32()),
                pa.array(col_datekey, type=pa.int32()),
                pa.array(col_edges_in_leg, type=pa.int32()),
            ],
            schema=traj_schema,
        )
        final_path = parts_dir / f"part_{part_counter:06d}.parquet"
        tmp_path = parts_dir / f"part_{part_counter:06d}.parquet.tmp"

        pq.write_table(chunk_table, tmp_path, compression="snappy")
        os.replace(tmp_path, final_path)
        part_counter += 1

        col_id.clear()
        col_trip_id.clear()
        col_directed_edge_id.clear()
        col_segmentkey.clear()
        col_direction.clear()
        col_trip_segmentno.clear()
        col_seconds.clear()
        col_timekey.clear()
        col_datekey.clear()
        col_edges_in_leg.clear()
        del chunk_table
        gc.collect()

    n_total = len(df_sample)
    pbar = tqdm(total=n_total, desc="Extracting routes")

    records_iter = list(zip(
        df_sample["TRIP_ID"].astype(str),
        df_sample["POLYLINE"],
        df_sample["TIMESTAMP"].astype(int),
        df_sample.get("MISSING_DATA", pd.Series([False] * n_total)).astype(str),
    ))

    try:
        for batch_start in range(0, n_total, fetch_batch_size):
            batch_slice = records_iter[batch_start: batch_start + fetch_batch_size]

            prepared = []
            for trip_id, raw_poly, start_ts, missing_data in batch_slice:
                if missing_data.strip().lower() == "true":
                    prepared.append(None)
                    pbar.update(1)
                    continue

                coords_list = json_parser.loads(raw_poly) if isinstance(raw_poly, str) else raw_poly
                if not coords_list or len(coords_list) < 2:
                    prepared.append(None)
                    pbar.update(1)
                    continue

                if len(coords_list) > 200:
                    coords_list = coords_list[:200]

                ok, _ = _trip_prefilter_flags(coords_list, bbox=(bbox if check_bbox else None))
                if not ok:
                    prepared.append(None)
                    pbar.update(1)
                    continue

                kept_points = _dedupe_stationary_points(coords_list)
                if len(kept_points) < 2:
                    prepared.append(None)
                    pbar.update(1)
                    continue

                orig_indices = [idx for idx, _ in kept_points]
                dedup_coords = [pt for _, pt in kept_points]

                coords_str = ";".join([f"{lon:.6f},{lat:.6f}" for lon, lat in dedup_coords])
                ts_str = ";".join([str(start_ts + 15 * idx) for idx in orig_indices])
                radiuses_str = ";".join([str(int(match_radius_m))] * len(dedup_coords))

                prepared.append({
                    "trip_id": trip_id,
                    "start_ts": start_ts,
                    "coords_str": coords_str,
                    "ts_str": ts_str,
                    "orig_timestamps": [start_ts + 15 * idx for idx in orig_indices],
                    "radiuses_str": radiuses_str,
                })

            fetch_targets = [(i, p) for i, p in enumerate(prepared) if p is not None]
            results = [None] * len(prepared)

            if fetch_targets:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {
                        executor.submit(
                            _fetch_match, session, base_url,
                            p["coords_str"], p["ts_str"], p["radiuses_str"],
                            request_timeout, max_retries,
                        ): i
                        for i, p in fetch_targets
                    }
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            results[idx] = future.result()
                        except Exception:
                            results[idx] = None
                        pbar.update(1)

            for i, p in enumerate(prepared):
                if p is None:
                    continue

                data = results[i]
                if data is None:
                    n_fetch_failed += 1
                    continue
                if data.get("code") != "Ok" or not data.get("matchings"):
                    continue

                base_trip_id = p["trip_id"]
                tracepoint_ts = p["orig_timestamps"]
                matchings = data.get("matchings", [])
                tracepoints = data.get("tracepoints", [])

                for m_idx, matching in enumerate(matchings):
                    legs = matching.get("legs", [])
                    if not legs:
                        continue

                    matching_geom = matching.get("geometry", {})
                    matching_coords = matching_geom.get("coordinates", [])

                    # Collect all node IDs from all legs
                    flat_nodes = []
                    for l_idx, leg in enumerate(legs):
                        l_nodes = leg.get("annotation", {}).get("nodes", [])
                        if not l_nodes:
                            continue
                        if l_idx == 0:
                            flat_nodes.extend(l_nodes)
                        else:
                            flat_nodes.extend(l_nodes[1:])

                    if len(matching_coords) == len(flat_nodes) and len(flat_nodes) > 0:
                        for n_id, pt in zip(flat_nodes, matching_coords):
                            node_coords[n_id] = (float(pt[0]), float(pt[1]))

                    sub_trip_id = base_trip_id if len(matchings) == 1 else f"{base_trip_id}_sub_{m_idx}"
                    trip_seg_no = 1
                    coord_idx = 0

                    # Map legs to actual GPS tracepoint timestamps
                    # In OSRM, leg k connects matched tracepoint k to matched tracepoint k+1
                    curr_sim_time = float(tracepoint_ts[0]) if tracepoint_ts else float(p["start_ts"])

                    for l_idx, leg in enumerate(legs):
                        ann = leg.get("annotation", {})
                        nodes = ann.get("nodes", [])
                        durations = ann.get("duration", [])
                        distances = ann.get("distance", [])
                        num_edges = len(nodes) - 1
                        if num_edges <= 0:
                            continue

                        # Determine actual elapsed GPS time across this leg
                        # Normally 15.0 seconds per consecutive waypoint pair
                        if l_idx + 1 < len(tracepoint_ts):
                            observed_leg_duration = float(tracepoint_ts[l_idx + 1] - tracepoint_ts[l_idx])
                        else:
                            observed_leg_duration = 15.0
                        if observed_leg_duration <= 0.0:
                            observed_leg_duration = 15.0

                        # Calculate base reference costs (free-flow duration or distance) for proportional allocation
                        ref_edge_dists = []
                        ref_edge_weights = []

                        for k in range(num_edges):
                            u = nodes[k]
                            v = nodes[k + 1]
                            dist_m = float(distances[k]) if k < len(distances) else 0.0
                            if dist_m <= 0.0:
                                p1 = node_coords.get(u)
                                p2 = node_coords.get(v)
                                if p1 and p2:
                                    dist_m = _haversine_m(p1[0], p1[1], p2[0], p2[1])
                                else:
                                    dist_m = 5.0
                            dist_m = max(dist_m, 1.0)
                            ref_edge_dists.append(dist_m)

                            cat = edge_categories.get((u, v)) or edge_categories.get((v, u)) or "residential"
                            sp_lim = edge_maxspeeds.get((u, v)) or edge_maxspeeds.get((v, u)) or DEFAULT_SPEED_BY_CAT.get(cat, 50.0)
                            free_flow_s = dist_m / max(sp_lim / 3.6, 1.0)

                            osrm_dur = float(durations[k]) if k < len(durations) else free_flow_s
                            weight = max(osrm_dur, free_flow_s, 0.1)
                            ref_edge_weights.append(weight)

                        sum_weights = sum(ref_edge_weights)
                        if sum_weights <= 0:
                            sum_weights = 1.0

                        # Allocate actual observed GPS duration proportionally across edges in this leg
                        allocated_durations = [
                            (w / sum_weights) * observed_leg_duration
                            for w in ref_edge_weights
                        ]

                        for k in range(num_edges):
                            u = nodes[k]
                            v = nodes[k + 1]

                            if u == v:
                                coord_idx += 1
                                continue

                            p1 = node_coords.get(u)
                            p2 = node_coords.get(v)

                            if p1 is None and coord_idx < len(matching_coords):
                                p1 = (float(matching_coords[coord_idx][0]), float(matching_coords[coord_idx][1]))
                                node_coords[u] = p1
                            if p2 is None and (coord_idx + 1) < len(matching_coords):
                                p2 = (float(matching_coords[coord_idx + 1][0]), float(matching_coords[coord_idx + 1][1]))
                                node_coords[v] = p2

                            coord_idx += 1

                            if p1 is None and p2 is not None:
                                p1 = p2
                            elif p2 is None and p1 is not None:
                                p2 = p1
                            elif p1 is None and p2 is None:
                                continue

                            dist = ref_edge_dists[k]
                            cat = edge_categories.get((u, v)) or edge_categories.get((v, u)) or "residential"
                            speedlimit = (
                                edge_maxspeeds.get((u, v))
                                or edge_maxspeeds.get((v, u))
                                or DEFAULT_SPEED_BY_CAT.get(cat, 50.0)
                            )

                            dur = allocated_durations[k]

                            # Clamp speeds to physically valid urban bounds (prevents 200 - 2,000 km/h spikes)
                            edge_max_v = min(max_edge_speed_kmh, max(speedlimit * 1.25, 40.0))
                            min_dur = dist / (edge_max_v / 3.6)
                            max_dur = dist / (min_edge_speed_kmh / 3.6)
                            dur = float(np.clip(dur, min_dur, max_dur))
                            dur = max(dur, 0.5)

                            directed_edge_id = f"{u}_{v}"

                            if directed_edge_id not in map_dict:
                                canon_p1 = node_coords[u]
                                canon_p2 = node_coords[v]
                                seg_angle = calculate_bearing(canon_p1[0], canon_p1[1], canon_p2[0], canon_p2[1])

                                if (u, v) in edge_geometries:
                                    line_geom = edge_geometries[(u, v)]
                                elif (v, u) in edge_geometries:
                                    line_geom = LineString(list(edge_geometries[(v, u)].coords)[::-1])
                                else:
                                    # Fallback to straight linestring
                                    line_geom = LineString([canon_p1, canon_p2])

                                geom_wkb = wkb.dumps(line_geom, hex=wkb_hex, srid=4326)
                                if isinstance(geom_wkb, str):
                                    geom_wkb = geom_wkb.encode("utf-8")

                                road_dir = edge_oneways.get(directed_edge_id, "BOTH")

                                map_dict[directed_edge_id] = {
                                    "directed_edge_id": directed_edge_id,
                                    "segmentkey": directed_edge_id,
                                    "direction": road_dir,
                                    "startpoint": u,
                                    "endpoint": v,
                                    "u": u,
                                    "v": v,
                                    "category": cat,
                                    "segangle": seg_angle,
                                    "meters": round(dist, 2),
                                    "segmentid": u,
                                    "segmentgeo": geom_wkb,
                                    "speedlimit_forward": round(speedlimit, 2),
                                }

                            dt = datetime.fromtimestamp(int(curr_sim_time), tz=timezone.utc)
                            datekey = dt.year * 10000 + dt.month * 100 + dt.day
                            timekey = dt.hour * 100 + dt.minute

                            col_id.append(global_obs_id)
                            col_trip_id.append(sub_trip_id)
                            col_directed_edge_id.append(directed_edge_id)
                            col_segmentkey.append(directed_edge_id)
                            col_direction.append("FORWARD")
                            col_trip_segmentno.append(trip_seg_no)
                            col_seconds.append(round(dur, 3))
                            col_timekey.append(timekey)
                            col_datekey.append(datekey)
                            col_edges_in_leg.append(num_edges)

                            global_obs_id += 1
                            trip_seg_no += 1
                            curr_sim_time += dur

                    n_trips_processed += 1
                    trips_since_flush += 1

                    if trips_since_flush >= flush_every_trips:
                        _flush_traj_part()
                        flushes_since_map_write += 1
                        if flushes_since_map_write >= map_flush_every_n_flushes:
                            _write_map_checkpoint()
                            flushes_since_map_write = 0
                        trips_since_flush = 0

            elapsed = pbar.format_dict.get("elapsed", 0) or 1e-9
            done = batch_start + len(batch_slice)
            rate = done / elapsed
            remaining_s = (n_total - done) / rate if rate > 0 else float("inf")
            _log(
                f"Progress: {done:,}/{n_total:,} trips ({100 * done / n_total:.1f}%) | "
                f"{rate:.1f} trips/s | ETA {remaining_s/60:.1f}m | fetch_failed={n_fetch_failed:,}"
            )

        _flush_traj_part()

    finally:
        pbar.close()
        _flush_traj_part()
        _write_map_checkpoint()

    _log(f"Extraction complete. Successfully processed: {n_trips_processed:,} trips.")
    print(f"Total segment observations: {global_obs_id:,}")
    print(f"Total unique directed road segments in map: {len(map_dict):,}")


# ==========================================
# 4. CONSOLIDATION
# ==========================================
def consolidate_traj_parts(traj_parts_dir, output_single_path, batch_size=1_000_000):
    _log(f"Consolidating {traj_parts_dir} -> {output_single_path}...")
    _t0 = time.time()

    dataset = pa_ds.dataset(traj_parts_dir, format="parquet")
    if dataset.count_rows() == 0:
        return

    tmp_path = f"{output_single_path}.tmp"
    writer = pq.ParquetWriter(tmp_path, dataset.schema, compression="snappy")
    n_written = 0
    try:
        for record_batch in dataset.to_batches(batch_size=batch_size):
            if record_batch.num_rows == 0:
                continue
            writer.write_batch(record_batch)
            n_written += record_batch.num_rows
    finally:
        writer.close()

    os.replace(tmp_path, output_single_path)
    _log(
        f"Consolidated {n_written:,} rows into {output_single_path} "
        f"({time.time() - _t0:.1f}s, {os.path.getsize(output_single_path) / 1e9:.2f} GB)"
    )


# ==========================================
# 5. EXECUTION ENTRY POINT
# ==========================================
traj_extracted_path = os.path.join(RAW_DATA_DIR, "porto_traj_extracted_parts")
traj_final_path = os.path.join(RAW_DATA_DIR, "porto_traj_extracted.parquet")


def _is_valid_polyline(p):
    if isinstance(p, str):
        return p.count("[") >= 3
    return len(p) >= 2 if p else False


valid_mask = [
    _is_valid_polyline(p)
    for p in tqdm(df["POLYLINE"], total=len(df), desc="Validating polylines")
]
valid_trips = df[valid_mask]
del df, valid_mask
gc.collect()

_log("Starting directed map-matching extraction...")
process_porto_to_pipeline(
    df_sample=valid_trips,
    traj_output_path=traj_extracted_path,
    map_output_path=map_extracted_path,
    wkb_hex=True,
    flush_every_trips=10000,
    match_radius_m=25.0,
    max_workers=4,
    fetch_batch_size=300,
    max_retries=3,
    request_timeout=30,
    check_bbox=True,
    bbox=GRAPH_BBOX,
)

consolidate_traj_parts(
    traj_parts_dir=traj_extracted_path,
    output_single_path=traj_final_path,
)
