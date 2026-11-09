"""
01_Porto_repair_corrupted_map.py

Runs after 01_Porto_extraction.py and before 03_preprocess.py.
Repairs corrupted node coordinates and synthetic straight-line spike geometries
in data/01_raw/porto_map_extracted.parquet using true OpenStreetMap geometries.
"""

import os
import pickle
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import osmnx as ox
import pandas as pd
import requests
import shapely
from shapely import wkb
from shapely.geometry import LineString, box
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "01_raw"

raw_map_path = RAW_DATA_DIR / "porto_map_extracted.parquet"
graph_cache_path = RAW_DATA_DIR / "porto_osmnx_graph.pkl"

if not raw_map_path.exists():
    raise FileNotFoundError(f"Could not find: {raw_map_path}")

print(f"Loading raw extracted map: {raw_map_path} ...")
df_map = pd.read_parquet(raw_map_path)
print(f"Loaded {len(df_map):,} road segments.")

# Ensure u and v are standardized integers
if "u" not in df_map.columns:
    df_map["u"] = df_map["startpoint"]
if "v" not in df_map.columns:
    df_map["v"] = df_map["endpoint"]

df_map["u"] = df_map["u"].astype("int64")
df_map["v"] = df_map["v"].astype("int64")

all_unique_nodes = set(df_map["u"]).union(set(df_map["v"]))
print(f"Found {len(all_unique_nodes):,} unique OSM nodes across map edges.")

# ---------------------------------------------------------
# 1. Load or Generate OSMnx Porto Graph Cache (Version-Safe Polygon)
# ---------------------------------------------------------
node_coords = {}
edge_geometries = {}

if graph_cache_path.exists():
    print(f"Loading cached graph from {graph_cache_path}...")
    with open(graph_cache_path, "rb") as f:
        G = pickle.load(f)
else:
    print("Downloading Porto drive network via OSMnx (single fast query)...")
    # Using explicit shapely box avoids OSMnx bbox coordinate ordering differences
    # min_lon (-8.73), min_lat (41.09), max_lon (-8.51), max_lat (41.27)
    porto_polygon = box(-8.7300, 41.0900, -8.5100, 41.2700)
    G = ox.graph_from_polygon(porto_polygon, network_type="drive", retain_all=True)
    
    with open(graph_cache_path, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved OSMnx graph cache to: {graph_cache_path}")

for n, d in G.nodes(data=True):
    node_coords[int(n)] = (float(d["x"]), float(d["y"]))

for u, v, d in G.edges(data=True):
    u_int, v_int = int(u), int(v)
    if "geometry" in d and isinstance(d["geometry"], LineString):
        edge_geometries[(u_int, v_int)] = d["geometry"]
    elif u_int in node_coords and v_int in node_coords:
        edge_geometries[(u_int, v_int)] = LineString([node_coords[u_int], node_coords[v_int]])

print(f"Indexed {len(node_coords):,} node coordinates and {len(edge_geometries):,} edge geometries.")

# ---------------------------------------------------------
# 2. Fetch Missing High-ID Nodes (>10B) via Official OSM API
# ---------------------------------------------------------
missing_nodes = [n for n in all_unique_nodes if n not in node_coords]
print(f"Nodes missing verified coordinates: {len(missing_nodes):,}")

if missing_nodes:
    print("Fetching verified coordinates for missing nodes via OpenStreetMap API...")
    batch_size = 200
    session = requests.Session()
    session.headers.update({"User-Agent": "PA-TAD-MapRepair/1.0"})

    for i in tqdm(range(0, len(missing_nodes), batch_size), desc="Fetching OSM nodes"):
        batch = missing_nodes[i:i + batch_size]
        nodes_param = ",".join(str(n) for n in batch)
        url = f"https://api.openstreetmap.org/api/0.6/nodes?nodes={nodes_param}"
        
        for attempt in range(3):
            try:
                resp = session.get(url, timeout=20)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for node_elem in root.findall("node"):
                        nid = int(node_elem.get("id"))
                        lat = float(node_elem.get("lat"))
                        lon = float(node_elem.get("lon"))
                        node_coords[nid] = (lon, lat)
                    break
                elif resp.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
            except Exception:
                time.sleep(1.0)

    print(f"Total resolved nodes: {len(node_coords):,}")

# ---------------------------------------------------------
# 3. Reconstruct Geometries (Replacing Drifting Synthetic Spikes)
# ---------------------------------------------------------
print("Reconstructing edge geometries...")

repaired_geoms = []
repaired_count = 0
fallback_count = 0

for _, row in tqdm(df_map.iterrows(), total=len(df_map), desc="Rebuilding edge geometries"):
    u = row["u"]
    v = row["v"]

    # 1. Exact curved geometry from OSMnx
    if (u, v) in edge_geometries:
        repaired_geoms.append(edge_geometries[(u, v)])
    elif (v, u) in edge_geometries:
        repaired_geoms.append(LineString(list(edge_geometries[(v, u)].coords)[::-1]))
    # 2. Straight line between verified true coordinates
    elif u in node_coords and v in node_coords:
        p1 = node_coords[u]
        p2 = node_coords[v]
        repaired_geoms.append(LineString([p1, p2]))
        repaired_count += 1
    # 3. Retain original binary geometry if unresolvable
    else:
        orig_raw = row["segmentgeo"]
        try:
            g = wkb.loads(orig_raw) if isinstance(orig_raw, (bytes, bytearray)) else wkb.loads(orig_raw, hex=True)
            repaired_geoms.append(g)
            fallback_count += 1
        except Exception:
            repaired_geoms.append(None)

df_map["geometry"] = repaired_geoms
df_map = df_map.dropna(subset=["geometry"])

# Format to binary WKB with SRID 4326 for 03_preprocess.py compatibility
def to_binary_wkb(geom):
    geom_with_srid = shapely.set_srid(geom, 4326)
    return shapely.to_wkb(geom_with_srid, hex=False, include_srid=True, byte_order=1)

df_map["segmentgeo"] = df_map["geometry"].apply(to_binary_wkb)
df_map = df_map.drop(columns=["geometry"])

# Overwrite raw map file safely
tmp_out = f"{raw_map_path}.tmp"
df_map.to_parquet(tmp_out, index=False)
os.replace(tmp_out, raw_map_path)

print(f"\nRaw map repair complete!")
print(f"Overwritten: {raw_map_path}")
print(f"Edges matched with verified coordinates: {repaired_count:,}")
print(f"Fallback edges: {fallback_count:,}")