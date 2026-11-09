"""
02_build_porto_minimal_basis.py

Transforms Porto's micro-segment network into a canonical Minimal Topological Basis:
1. Contracts chains of degree-2 curve nodes into single macro-edges between real junctions.
2. Extracts constituent micro-edges and their individual metric lengths (micro_meters).
3. Computes accurate compass bearing (segangle), metric length, and tortuosity in EPSG:32629.
4. Serializes segmentgeo as EWKB Hex string, matching the Danish map standard.
"""

import math
import os
import pickle
from pathlib import Path
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from shapely.geometry import LineString, Point, box
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "01_raw"
INTERIM_DATA_DIR = BASE_DIR / "data" / "02_interim"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"

INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

out_map_interim = INTERIM_DATA_DIR / "porto_minimal_map.parquet"
out_map_processed = PROCESSED_DATA_DIR / "map_directed_porto.parquet"
out_lookup_path = INTERIM_DATA_DIR / "porto_micro_to_macro_lookup.parquet"
graph_cache_path = RAW_DATA_DIR / "porto_osmnx_unsimplified_graph.pkl"

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

def calculate_bearing(lon1, lat1, lon2, lat2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    return round((math.degrees(math.atan2(y, x)) + 360) % 360, 2)

# ---------------------------------------------------------
# 1. Load Unsimplified Graph & Coordinate Dictionary
# ---------------------------------------------------------
if graph_cache_path.exists():
    print(f"Loading unsimplified graph from {graph_cache_path}...")
    with open(graph_cache_path, "rb") as f:
        G = pickle.load(f)
else:
    print("Downloading Porto unsimplified drive network...")
    porto_box = box(-8.7300, 41.0900, -8.5100, 41.2700)
    G = ox.graph_from_polygon(porto_box, network_type="drive", simplify=False, retain_all=True)
    with open(graph_cache_path, "wb") as f:
        pickle.dump(G, f)

print(f"Base graph: {len(G.nodes):,} nodes, {len(G.edges):,} micro-edges.")

coord_to_node = {
    (round(float(d["x"]), 6), round(float(d["y"]), 6)): int(n)
    for n, d in G.nodes(data=True)
}

# ---------------------------------------------------------
# 2. Contract Degree-2 Chains into Minimal Topological Basis
# ---------------------------------------------------------
print("Contracting intermediate curve nodes into intersection-to-intersection edges...")
G_simple = ox.simplify_graph(G)
print(f"Contracted minimal graph: {len(G_simple.nodes):,} intersections, {len(G_simple.edges):,} macro-edges.")

macro_records = []
micro_to_macro = []

# Metric distance helper using equirectangular projection approximation
def _quick_dist_m(p1, p2):
    dx = (p2[0] - p1[0]) * math.cos(math.radians(p1[1])) * 111320.0
    dy = (p2[1] - p1[1]) * 110540.0
    return max(1.0, math.sqrt(dx * dx + dy * dy))

for u, v, k, d in tqdm(G_simple.edges(keys=True, data=True), desc="Building canonical macro-edges"):
    u_int = int(u)
    v_int = int(v)
    macro_id = f"{u_int}_{v_int}"

    if "geometry" in d and isinstance(d["geometry"], LineString):
        geom = d["geometry"]
    else:
        p1 = (float(G_simple.nodes[u]["x"]), float(G_simple.nodes[u]["y"]))
        p2 = (float(G_simple.nodes[v]["x"]), float(G_simple.nodes[v]["y"]))
        geom = LineString([p1, p2])

    coords = list(geom.coords)
    segangle = calculate_bearing(coords[0][0], coords[0][1], coords[1][0], coords[1][1]) if len(coords) >= 2 else 0.0

    path_nodes = []
    path_coords = []
    for pt in coords:
        key = (round(pt[0], 6), round(pt[1], 6))
        nid = coord_to_node.get(key)
        if nid is not None:
            if not path_nodes or path_nodes[-1] != nid:
                path_nodes.append(nid)
                path_coords.append(pt)

    if len(path_nodes) >= 2:
        for i in range(len(path_nodes) - 1):
            m_len = _quick_dist_m(path_coords[i], path_coords[i + 1])
            micro_to_macro.append({
                "micro_edge_id": f"{path_nodes[i]}_{path_nodes[i+1]}",
                "macro_edge_id": macro_id,
                "macro_u": u_int,
                "macro_v": v_int,
                "micro_meters": round(m_len, 2)
            })
    else:
        m_len = _quick_dist_m(coords[0], coords[-1])
        micro_to_macro.append({
            "micro_edge_id": macro_id,
            "macro_edge_id": macro_id,
            "macro_u": u_int,
            "macro_v": v_int,
            "micro_meters": round(m_len, 2)
        })

    h = d.get("highway", "residential")
    category = h[0] if isinstance(h, list) else str(h)
    speed_val = d.get("maxspeed", 50.0)
    if isinstance(speed_val, list):
        speed_val = speed_val[0]
    try:
        speed_limit = float(str(speed_val).split()[0])
    except Exception:
        speed_limit = DEFAULT_SPEED_BY_CAT.get(category, 50.0)

    macro_records.append({
        "directed_edge_id": macro_id,
        "segmentkey": macro_id,
        "u": u_int,
        "v": v_int,
        "category": category,
        "speed_limit": speed_limit,
        "geometry": geom,
        "traj_direction": "FORWARD",
        "segangle": segangle
    })

# Unsimplified fallback
known_micro = {r["micro_edge_id"] for r in micro_to_macro}
for u, v in G.edges():
    u_int, v_int = int(u), int(v)
    m_id = f"{u_int}_{v_int}"
    if m_id not in known_micro:
        p1 = (float(G.nodes[u]["x"]), float(G.nodes[u]["y"]))
        p2 = (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))
        micro_to_macro.append({
            "micro_edge_id": m_id,
            "macro_edge_id": m_id,
            "macro_u": u_int,
            "macro_v": v_int,
            "micro_meters": round(_quick_dist_m(p1, p2), 2)
        })

# ---------------------------------------------------------
# 3. Metric Projection & Tortuosity (EPSG:32629)
# ---------------------------------------------------------
print("Projecting macro-edges to EPSG:32629 for metric length & tortuosity...")
gdf = gpd.GeoDataFrame(macro_records, geometry="geometry", crs="EPSG:4326")
gdf_metric = gdf.to_crs("EPSG:32629")

gdf["segment_length"] = gdf_metric["geometry"].length.round(2).clip(lower=2.0)

def get_chord_len(g):
    coords = list(g.coords)
    return Point(coords[0]).distance(Point(coords[-1]))

gdf["chord_length"] = gdf_metric["geometry"].apply(get_chord_len)
gdf["tortuosity"] = np.where(
    gdf["chord_length"] > 0,
    np.maximum(1.0, (gdf["segment_length"] / gdf["chord_length"]).round(4)),
    1.0
)

def to_ewkb_hex(geom):
    if not isinstance(geom, LineString) or geom.is_empty:
        return None
    geom_with_srid = shapely.set_srid(geom, 4326)
    return shapely.to_wkb(geom_with_srid, hex=True, include_srid=True, byte_order=1)

gdf["segmentgeo"] = gdf["geometry"].apply(to_ewkb_hex)
gdf["segmentid"] = gdf["u"]

cols = [
    "directed_edge_id", "u", "v", "speed_limit", "segment_length", "category",
    "segmentid", "tortuosity", "segmentkey", "traj_direction", "segmentgeo", "segangle"
]
df_minimal_map = pd.DataFrame(gdf)[cols].drop_duplicates(subset=["directed_edge_id"])
df_minimal_map.to_parquet(out_map_interim, index=False)
df_minimal_map.to_parquet(out_map_processed, index=False)

df_lookup = pd.DataFrame(micro_to_macro).drop_duplicates(subset=["micro_edge_id"])
df_lookup.to_parquet(out_lookup_path, index=False)

print(f"\nMinimal Basis Map built: {out_map_interim} ({len(df_minimal_map):,} macro-edges)")
print(f"Copied map directly to: {out_map_processed}")
print(f"Micro-to-Macro lookup built: {out_lookup_path} ({len(df_lookup):,} mappings)")