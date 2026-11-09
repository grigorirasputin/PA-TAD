"""
Generate reproducible, profile-calibrated synthetic trajectory anomalies.

The generator samples route lengths from the matching real trajectory dataset,
uses explicit and validated spatial anomaly operators, batches posterior
inference, and writes output incrementally so country-scale runs do not retain
all generated records in memory.
"""

import argparse
import json
import pickle
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from shapely import wkb
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
PRIOR_MODELING_DIR = BASE_DIR / "src" / "02_prior_modeling"
sys.path.append(str(PRIOR_MODELING_DIR))

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True, help="dataset pair label, e.g. '2014', '2016_itsp', or 'porto'")
parser.add_argument("--seed", type=int, default=42, help="random seed for a fully reproducible run")
parser.add_argument("--regions", nargs="+", default=None,
                    help="regions to generate, e.g. 'porto' or 'aalborg copenhagen denmark'")
parser.add_argument("--counts", nargs="+", type=int, default=None,
                    help="number of trajectories per region (matching order of --regions)")
parser.add_argument("--chunk-size", type=int, default=1_000, help="completed trips held before each write")
args = parser.parse_args()

# Determine active regions default based on tag
if args.regions is None:
    if "porto" in str(args.tag).lower():
        args.regions = ["porto"]
    else:
        args.regions = ["aalborg", "copenhagen", "denmark"]

original_argv = sys.argv
sys.argv = [sys.argv[0], "--tag", args.tag]
try:
    from _05_train_prior import BayesianTrafficGAT, TEMPORAL_FEATURE_COLUMNS, evaluate_global_weekly_prior
finally:
    sys.argv = original_argv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = BASE_DIR / "models" / "prior_weights" / args.tag
PROCESSED_DIR = BASE_DIR / "data" / "03_processed"
SYNTH_DIR = BASE_DIR / "data" / "05_synthetic" / args.tag
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

CLASS_RATIOS = {"normal": 0.90, "temporal": 0.04, "spatial": 0.04, "spatio_temporal": 0.02}

# Bounding boxes (min_lon, min_lat, max_lon, max_lat)
# Aalborg: ~10x10 km urban core centered at (9.925 E, 57.045 N) to support 100x100 grid cells
# Porto: ~10x10 km urban core
REGION_BOUNDS = {
    "aalborg": (9.8425, 57.0000, 10.0075, 57.0900),
    "copenhagen": (12.3000, 55.5600, 12.6800, 55.8500),
    "denmark": (8.00, 54.50, 12.70, 57.80),
    "porto": (-8.665000, 41.104955, -8.545306, 41.195045),
}

MAX_ROUTE_ATTEMPTS = 50
MAX_ANOMALY_ATTEMPTS = 30
MIN_CHANGED_EDGES = 3
MAX_SPATIAL_JACCARD = 0.85
MIN_COST_RATIO = 1.05
INFERENCE_BATCH_SIZE = 32_768


def get_min_route_hops(region: str) -> int:
    return 6 if region.lower() in ("porto", "aalborg") else 10


def temporal_features(timestamps):
    timestamps = pd.to_datetime(timestamps)
    hour_of_week = timestamps.dayofweek.to_numpy() * 24 + timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    return np.column_stack((
        np.sin(2 * np.pi * hour_of_week / 168.0),
        np.cos(2 * np.pi * hour_of_week / 168.0),
        np.sin(2 * np.pi * hour_of_week / 24.0),
        np.cos(2 * np.pi * hour_of_week / 24.0),
        np.sin(2 * np.pi * hour_of_week / 12.0),
        np.cos(2 * np.pi * hour_of_week / 12.0),
    )).astype(np.float32)


def build_profile():
    path = PROCESSED_DIR / f"traj_mapped_{args.tag}.parquet"
    print(f"Reading real trajectory profile from: {path}")
    df = pd.read_parquet(path, columns=["trip_id", "trip_segmentno", "datetime"])
    
    # Sort chronologically to preserve sequence integrity
    df = df.sort_values(["trip_id", "trip_segmentno"])

    is_porto = "porto" in str(args.tag).lower() or "porto" in args.regions
    if is_porto:
        print("Porto dataset detected: aggregating split trip segments (e.g. '_p6') to reconstruct true route lengths.")
        df["profile_trip_id"] = df["trip_id"].astype(str).str.replace(r"_p\d+$", "", regex=True)
    else:
        df["profile_trip_id"] = df["trip_id"]

    grouped = df.groupby("profile_trip_id", sort=False)
    hops = grouped.size().to_numpy(dtype=np.int32)
    departures = pd.to_datetime(grouped["datetime"].first())

    valid_mask = hops >= 2
    hops = hops[valid_mask]
    departures = departures[valid_mask]

    profile = {
        "source": str(path),
        "trip_count": int(len(hops)),
        "hop_quantiles": {str(q): float(np.quantile(hops, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "sampled_hops": hops.tolist(),
        "departure_dates": departures.dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
    }
    print(f"Real-data profile ({'Porto-aggregated' if is_porto else args.tag}): "
          f"median={np.median(hops):.0f} hops, mean={np.mean(hops):.1f} hops, "
          f"min={np.min(hops)}, max={np.max(hops)}.")
    return profile


def directed_edges_in_region_wkb(map_df, bounds):
    """
    Filters directed edges strictly intersecting the regional bounding box
    using segmentgeo directly from 03_processed.
    """
    min_x, min_y, max_x, max_y = bounds

    def intersects_bounds(value):
        if value is None or pd.isna(value):
            return False
        try:
            geometry = wkb.loads(bytes.fromhex(value)) if isinstance(value, str) else wkb.loads(value)
        except Exception:
            return False
        if geometry is None or geometry.is_empty:
            return False
        geom_min_x, geom_min_y, geom_max_x, geom_max_y = geometry.bounds
        return not (geom_max_x < min_x or geom_min_x > max_x or geom_max_y < min_y or geom_min_y > max_y)

    mask = map_df["segmentgeo"].map(intersects_bounds)
    selected = map_df[mask]
    return set(zip(selected["u"].astype(np.int64), selected["v"].astype(np.int64)))


def build_regional_graphs(master_map):
    """
    Constructs graph and slices subgraphs per region directly using segmentgeo
    and segangle from 03_processed/map_directed_{tag}.parquet.
    """
    map_path = PROCESSED_DIR / f"map_directed_{args.tag}.parquet"
    print(f"Loading map geometries directly from: {map_path}")
    
    # Read available geometry and angle columns
    available_cols = pq.ParquetFile(map_path).schema.names
    cols_to_read = ["u", "v"]
    if "segmentgeo" in available_cols:
        cols_to_read.append("segmentgeo")
    if "segangle" in available_cols:
        cols_to_read.append("segangle")
    
    raw_map = pd.read_parquet(map_path, columns=cols_to_read)
    raw_map["u"] = raw_map["u"].astype(np.int64)
    raw_map["v"] = raw_map["v"].astype(np.int64)
    raw_map = raw_map.drop_duplicates(subset=["u", "v"], keep="first")

    master_map["u"] = master_map["u"].astype(np.int64)
    master_map["v"] = master_map["v"].astype(np.int64)

    # Master graph with empirical/posterior edge weights
    full_graph = nx.DiGraph()
    for row in master_map.itertuples(index=False):
        edge_weight = float(np.exp(row.log_base_time + getattr(row, "Bayesian_L2_Shift", 0.0)))
        full_graph.add_edge(
            int(row.u),
            int(row.v),
            edge_id=int(row.edge_id),
            weight=max(0.1, edge_weight),
            l2_shift=float(getattr(row, "Bayesian_L2_Shift", 0.0)),
            obs_count=int(getattr(row, "Obs_Count", 1)),
        )

    graphs, maps = {}, {}
    for region in args.regions:
        valid_edge_set = directed_edges_in_region_wkb(raw_map, REGION_BOUNDS[region])
        graph = full_graph.edge_subgraph(valid_edge_set).copy()
        if not graph or graph.number_of_edges() == 0:
            raise ValueError(f"{region}: no map edges were found within bounds {REGION_BOUNDS[region]}.")
        
        # Retain largest strongly connected component to avoid dead-ends during pathfinding
        sccs = list(nx.strongly_connected_components(graph))
        if sccs:
            component = max(sccs, key=len)
            graph = graph.subgraph(component).copy()
        
        graph_edges = set(graph.edges())
        graphs[region] = graph
        
        # Slice master map dataframe for this region strictly matching graph edges
        edge_tuples = set(zip(master_map["u"], master_map["v"]))
        matched_edges = edge_tuples.intersection(graph_edges)
        sliced_map = master_map[master_map[["u", "v"]].apply(tuple, axis=1).isin(matched_edges)].copy()

        # Merge segmentgeo and segangle from raw_map into sliced_map if not already present
        merge_cols = ["u", "v"]
        for col in ["segmentgeo", "segangle"]:
            if col in raw_map.columns and col not in sliced_map.columns:
                merge_cols.append(col)

        if len(merge_cols) > 2:
            sliced_map = sliced_map.merge(raw_map[merge_cols], on=["u", "v"], how="left")

        maps[region] = sliced_map
        print(f"Region '{region}': {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} directed edges.")
        print(f"Region '{region}' map columns: {list(maps[region].columns)}")
    return graphs, maps


class RouteSampler:
    def __init__(self, graph, map_df, profile, rng, min_route_hops=6):
        self.graph = graph
        self.rng = rng
        self.min_route_hops = min_route_hops
        self.hop_samples = np.asarray(profile["sampled_hops"], dtype=np.int32)
        origins = map_df["u"].to_numpy(dtype=np.int64)
        weights = map_df["Obs_Count"].to_numpy(dtype=np.float64) + 1.0 if "Obs_Count" in map_df.columns else np.ones(len(map_df))
        self.origins = origins
        self.origin_probabilities = weights / weights.sum()

    def reference_route(self):
        for _ in range(MAX_ROUTE_ATTEMPTS):
            target_hops = int(self.rng.choice(self.hop_samples))
            target_hops = max(self.min_route_hops, target_hops)
            tolerance = max(2, int(target_hops * 0.25))
            origin = int(self.rng.choice(self.origins, p=self.origin_probabilities))
            
            try:
                distances = nx.single_source_shortest_path_length(
                    self.graph, origin, cutoff=target_hops + tolerance
                )
            except nx.NodeNotFound:
                continue

            candidates = [
                node for node, hops in distances.items()
                if abs(hops - target_hops) <= tolerance and node != origin
            ]
            if not candidates:
                continue
            destination = int(self.rng.choice(candidates))
            try:
                path = nx.shortest_path(self.graph, origin, destination, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            if len(path) - 1 >= self.min_route_hops:
                return path, target_hops
        return None, None


def edge_pairs(path):
    return list(zip(path[:-1], path[1:]))


def route_cost(graph, path):
    return sum(graph[u][v]["weight"] for u, v in edge_pairs(path))


def valid_spatial_anomaly(graph, reference, candidate):
    reference_edges, candidate_edges = set(edge_pairs(reference)), set(edge_pairs(candidate))
    changed = len(candidate_edges - reference_edges)
    union = len(reference_edges | candidate_edges)
    jaccard = len(reference_edges & candidate_edges) / union if union else 1.0
    reference_cost = route_cost(graph, reference)
    cost_ratio = route_cost(graph, candidate) / reference_cost if reference_cost else 1.0
    return changed >= MIN_CHANGED_EDGES and jaccard <= MAX_SPATIAL_JACCARD and cost_ratio >= MIN_COST_RATIO, {
        "changed_edges": changed, "jaccard": jaccard, "cost_ratio": cost_ratio,
    }


def random_walk(graph, reference, rng, max_walk=12):
    reference_nodes = set(reference)
    for _ in range(MAX_ANOMALY_ATTEMPTS):
        if len(reference) <= 5:
            continue
        start_index = int(rng.integers(1, len(reference) - 3))
        rejoin_index = int(rng.integers(start_index + 2, len(reference)))
        current = reference[start_index]
        excursion = []
        for _ in range(int(rng.integers(3, max_walk + 1))):
            choices = [node for node in graph.successors(current) if node not in reference_nodes]
            if not choices:
                break
            current = int(rng.choice(choices))
            excursion.append(current)
        if len(excursion) < MIN_CHANGED_EDGES:
            continue
        try:
            recovery = nx.shortest_path(graph, current, reference[rejoin_index], weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        candidate = reference[:start_index + 1] + excursion + recovery[1:] + reference[rejoin_index + 1:]
        valid, metadata = valid_spatial_anomaly(graph, reference, candidate)
        if valid:
            metadata.update({"operator": "SPRW", "start_index": start_index, "rejoin_index": rejoin_index})
            return candidate, metadata
    return None, None


def detour_anomaly(graph, reference, rng):
    reference_nodes = set(reference)
    for _ in range(MAX_ANOMALY_ATTEMPTS):
        if len(reference) <= 4:
            continue
        start_index = int(rng.integers(1, len(reference) - 2))
        start = reference[start_index]
        current = start
        detour = []
        for _ in range(int(rng.integers(3, 8))):
            choices = [node for node in graph.successors(current) if node not in reference_nodes]
            if not choices:
                break
            current = int(rng.choice(choices))
            detour.append(current)
        if len(detour) < MIN_CHANGED_EDGES:
            continue
        try:
            return_path = nx.shortest_path(graph, current, start, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        candidate = reference[:start_index + 1] + detour + return_path[1:] + reference[start_index + 1:]
        valid, metadata = valid_spatial_anomaly(graph, reference, candidate)
        if valid:
            metadata.update({"operator": "Detour", "start_index": start_index})
            return candidate, metadata
    return None, None


def overshoot_anomaly(graph, reference, rng):
    destination = reference[-1]
    reference_nodes = set(reference)
    for _ in range(MAX_ANOMALY_ATTEMPTS):
        current, overshoot = destination, []
        for _ in range(int(rng.integers(3, 8))):
            choices = [node for node in graph.successors(current) if node not in reference_nodes]
            if not choices:
                break
            current = int(rng.choice(choices))
            overshoot.append(current)
        if len(overshoot) < MIN_CHANGED_EDGES:
            continue
        try:
            return_path = nx.shortest_path(graph, current, destination, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        candidate = reference + overshoot + return_path[1:]
        valid, metadata = valid_spatial_anomaly(graph, reference, candidate)
        if valid:
            metadata.update({"operator": "OVERSHOOT", "start_index": len(reference) - 1})
            return candidate, metadata
    return None, None


def spatial_anomaly(graph, reference, rng):
    operators = [random_walk, detour_anomaly, overshoot_anomaly]
    rng.shuffle(operators)
    for operator in operators:
        candidate, metadata = operator(graph, reference, rng)
        if candidate:
            return candidate, metadata
    return None, None


class PosteriorSampler:
    def __init__(self):
        with open(MODELS_DIR / "graph_artifacts.pkl", "rb") as file:
            artifacts = pickle.load(file)
        self.graph_data = artifacts["graph_data"].to(DEVICE)
        self.log_base_time = artifacts["log_base_time"].to(DEVICE)
        with open(MODELS_DIR / "global_weekly_prior.json", encoding="utf-8") as file:
            self.global_weekly_prior = json.load(file)
        self.model = BayesianTrafficGAT(
            static_in_channels=self.graph_data.x.shape[1], time_channels=len(TEMPORAL_FEATURE_COLUMNS)
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(MODELS_DIR / "GAT_prior_weights.pth", map_location=DEVICE, weights_only=True))
        self.model.eval()
        with torch.no_grad():
            self.encoded_graph = self.model.encode(self.graph_data.x, self.graph_data.edge_index)

    def sample(self, edge_ids, timestamps, shifts, temporal_intervals, rng):
        features = temporal_features(pd.DatetimeIndex(timestamps))
        timestamp_index = pd.DatetimeIndex(timestamps)
        hour_of_week = (
            timestamp_index.dayofweek.to_numpy() * 24
            + timestamp_index.hour.to_numpy()
            + timestamp_index.minute.to_numpy() / 60.0
        )
        global_delay = evaluate_global_weekly_prior(hour_of_week, self.global_weekly_prior)
        edge_ids = np.asarray(edge_ids, dtype=np.int64)
        means, variances = np.empty(len(edge_ids)), np.empty(len(edge_ids))
        with torch.no_grad():
            for start in range(0, len(edge_ids), INFERENCE_BATCH_SIZE):
                stop = min(start + INFERENCE_BATCH_SIZE, len(edge_ids))
                edges = torch.as_tensor(edge_ids[start:stop], dtype=torch.long, device=DEVICE)
                times = torch.as_tensor(features[start:stop], dtype=torch.float32, device=DEVICE)
                batch_global_delay = torch.as_tensor(global_delay[start:stop], dtype=torch.float32, device=DEVICE)
                mu, log_var = self.model.predict(
                    self.encoded_graph, self.graph_data.edge_index, edges, times, self.log_base_time, batch_global_delay
                )
                means[start:stop] = mu.cpu().numpy()
                variances[start:stop] = np.exp(log_var.cpu().numpy())
        z = np.empty(len(edge_ids))
        z[0] = rng.normal()
        for index in range(1, len(z)):
            z[index] = 0.35 * z[index - 1] + np.sqrt(1 - 0.35 ** 2) * rng.normal()
        flags = np.zeros(len(edge_ids), dtype=np.int8)
        for start, stop in temporal_intervals:
            severity = rng.uniform(2.5, 4.5)
            z[start:stop] += severity
            flags[start:stop] = 1
        seconds = np.exp(means + np.asarray(shifts) + np.sqrt(variances) * z)
        return np.clip(seconds, 0.5, 7_200.0), flags


def classify_edges(reference, actual, operator):
    reference_edges = set(edge_pairs(reference))
    flags = np.array([int(edge not in reference_edges) for edge in edge_pairs(actual)], dtype=np.int8)
    if operator == "OVERSHOOT":
        first_destination = actual.index(reference[-1])
        flags[first_destination:] = 1
    return flags


def generate_region(region, count, graph, map_df, profile, posterior, rng, py_rng):
    min_route_hops = get_min_route_hops(region)
    sampler = RouteSampler(graph, map_df, profile, rng, min_route_hops=min_route_hops)
    class_labels = np.array(list(CLASS_RATIOS))
    class_counts = np.floor(np.array(list(CLASS_RATIOS.values())) * count).astype(int)
    class_counts[0] += count - class_counts.sum()
    requested = np.repeat(class_labels, class_counts)
    rng.shuffle(requested)
    edge_metadata = {(u, v): data for u, v, data in graph.edges(data=True)}
    output_path = SYNTH_DIR / f"synthetic_trajectories_{region}.parquet"
    manifest_path = SYNTH_DIR / f"synthetic_trajectories_{region}_manifest.json"
    writer, records, stats = None, [], Counter()
    stats["requested"] = int(count)

    try:
        for generated, anomaly_class in enumerate(tqdm(requested, desc=f"Generating {region}"), start=1):
            while True:
                reference, target_hops = sampler.reference_route()
                if reference is None:
                    stats["route_sampling_failures"] += 1
                    continue
                actual, anomaly_metadata = reference, {"operator": "None", "changed_edges": 0, "jaccard": 1.0, "cost_ratio": 1.0}
                if anomaly_class in ("spatial", "spatio_temporal"):
                    actual, anomaly_metadata = spatial_anomaly(graph, reference, rng)
                    if actual is None:
                        stats["spatial_rejections"] += 1
                        continue
                break

            edge_data = [edge_metadata[edge] for edge in edge_pairs(actual)]
            departure = pd.Timestamp(py_rng.choice(profile["departure_dates"]))
            temporal_intervals = []
            if anomaly_class in ("temporal", "spatio_temporal"):
                duration = int(rng.integers(4, min(8, len(edge_data))))
                start = int(rng.integers(1, len(edge_data) - duration + 1))
                temporal_intervals.append((start, start + duration))
            timestamps = []
            expected_time = departure
            for edge in edge_data:
                timestamps.append(expected_time)
                expected_time += timedelta(seconds=float(edge["weight"]))
            seconds, temporal_flags = posterior.sample(
                [edge["edge_id"] for edge in edge_data], timestamps,
                [edge["l2_shift"] for edge in edge_data], temporal_intervals, rng,
            )
            spatial_flags = classify_edges(reference, actual, anomaly_metadata["operator"])
            current_time = departure
            trip_id = f"synth_{args.tag}_{region}_{generated - 1:08d}"
            for index, ((u, v), edge, duration) in enumerate(zip(edge_pairs(actual), edge_data, seconds)):
                records.append({
                    "trip_id": trip_id, "u": int(u), "v": int(v), "edge_id": edge["edge_id"], "datetime": current_time,
                    "seconds": float(duration), "anomaly_class": anomaly_class,
                    "spatial_anomaly": int(spatial_flags[index]), "temporal_anomaly": int(temporal_flags[index]),
                    "spatial_subtype": anomaly_metadata["operator"] if spatial_flags[index] else "None",
                    "reference_hops": len(reference) - 1, "target_hops": target_hops,
                    "spatial_jaccard": anomaly_metadata["jaccard"], "spatial_cost_ratio": anomaly_metadata["cost_ratio"],
                })
                current_time += timedelta(seconds=float(duration))
            stats[f"class_{anomaly_class}"] += 1
            stats["generated_hops_total"] += len(edge_data)
            if len(records) >= args.chunk_size * 20:
                table = pa.Table.from_pylist(records)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
                writer.write_table(table)
                records.clear()
        if records:
            table = pa.Table.from_pylist(records)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    map_df.to_parquet(SYNTH_DIR / f"map_df_{region}.parquet", index=False)
    with open(SYNTH_DIR / f"graph_{region}.pkl", "wb") as file:
        pickle.dump(graph, file)
    stats["mean_generated_hops"] = stats["generated_hops_total"] / count
    manifest = {
        "tag": args.tag, "region": region, "seed": args.seed, "configuration": {
            "class_ratios": CLASS_RATIOS, "min_route_hops": min_route_hops,
            "min_changed_edges": MIN_CHANGED_EDGES, "max_spatial_jaccard": MAX_SPATIAL_JACCARD,
            "min_cost_ratio": MIN_COST_RATIO, "inference_batch_size": INFERENCE_BATCH_SIZE,
        },
        "real_data_profile": {key: value for key, value in profile.items() if key not in ("sampled_hops", "departure_dates")},
        "generation_stats": dict(stats),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {count:,} trips to {output_path}; mean generated hops={stats['mean_generated_hops']:.1f}.")


def main():
    if set(args.regions) - set(REGION_BOUNDS):
        raise ValueError(f"Unknown region(s): {sorted(set(args.regions) - set(REGION_BOUNDS))}")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive.")
    
    # Establish regional counts mapping
    if args.counts is not None:
        if len(args.counts) != len(args.regions):
            raise ValueError(f"--counts length ({len(args.counts)}) must match --regions length ({len(args.regions)})")
        counts = dict(zip(args.regions, args.counts))
    else:
        default_map = {"aalborg": 50_000, "copenhagen": 100_000, "denmark": 200_000, "porto": 50_000}  # paper, Table II
        counts = {r: default_map.get(r, 10_000) for r in args.regions}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng, py_rng = np.random.default_rng(args.seed), random.Random(args.seed)

    profile = build_profile()
    master_map = pd.read_parquet(SYNTH_DIR / "master_posterior_map.parquet")
    graphs, regional_maps = build_regional_graphs(master_map)
    posterior = PosteriorSampler()

    for region in args.regions:
        generate_region(region, counts[region], graphs[region], regional_maps[region], profile, posterior, rng, py_rng)


if __name__ == "__main__":
    main()
