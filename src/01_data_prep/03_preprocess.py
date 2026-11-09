"""
03_preprocess.py

Streams raw trajectory parquet, maps micro-edges to the canonical minimal basis graph,
normalizes fractional boundary traversals (pickups/dropoffs), prunes GPS teleportation outliers,
and ensures 100% topological continuity for GAT models.
"""

import os
import gc
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--traj-file", required=True, help="Trajectory file name or path")
parser.add_argument("--tag", required=True, help="Dataset identifier tag, e.g. 'porto'")
parser.add_argument("--chunk-size", required=False, type=int, default=500_000)
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "01_raw"
INTERIM_DATA_DIR = BASE_DIR / "data" / "02_interim"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "03_processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Resolve Paths
traj_path = RAW_DATA_DIR / args.traj_file
minimal_map_path = INTERIM_DATA_DIR / f"{args.tag}_minimal_map.parquet"
lookup_path = INTERIM_DATA_DIR / f"{args.tag}_micro_to_macro_lookup.parquet"

if not minimal_map_path.exists() or not lookup_path.exists():
    raise FileNotFoundError("Run 02_build_porto_minimal_basis.py first to build the minimal basis map!")

# 2. Save Final Map to Processed
df_map = pd.read_parquet(minimal_map_path)
final_map_path = PROCESSED_DATA_DIR / f"map_directed_{args.tag}.parquet"
df_map.to_parquet(final_map_path, index=False)
print(f"Minimal basis map copied to {final_map_path}")

# Load lookup table into memory (~12 MB)
df_lookup_pd = pd.read_parquet(lookup_path)
pl_lookup = pl.from_pandas(df_lookup_pd).lazy()

# Prepare map lengths lookup for speed checks
df_map_meta = df_map[["directed_edge_id", "segment_length", "speed_limit"]].copy()
pl_map_meta = pl.from_pandas(df_map_meta).lazy()

# 3. Stream and Contract Trajectories
parquet_file = pq.ParquetFile(traj_path)
total_rows = parquet_file.metadata.num_rows

traj_out = PROCESSED_DATA_DIR / f"traj_mapped_{args.tag}.parquet"
traj_out_tmp = PROCESSED_DATA_DIR / f"traj_mapped_{args.tag}.parquet.tmp"
if traj_out_tmp.exists():
    traj_out_tmp.unlink()

writer = None
carry_df = None
total_written_rows = 0

pbar = tqdm(total=total_rows, desc="Streaming, Contracting & Normalizing", unit="rows")

try:
    for batch in parquet_file.iter_batches(batch_size=args.chunk_size):
        chunk_pl = pl.from_arrow(batch)
        pbar.update(len(chunk_pl))

        c_names = chunk_pl.columns
        id_col = "trip_id" if "trip_id" in c_names else "traj_id"
        sec_col = "seconds" if "seconds" in c_names else ("duration" if "duration" in c_names else None)

        if "datetime" in c_names:
            if chunk_pl.schema["datetime"] != pl.Datetime:
                dt_expr = pl.col("datetime").str.to_datetime(strict=False).alias("datetime")
            else:
                dt_expr = pl.col("datetime").alias("datetime")
        elif "timestamp" in c_names:
            if chunk_pl.schema["timestamp"] in [pl.Int64, pl.Int32, pl.Float64]:
                dt_expr = pl.from_epoch(pl.col("timestamp").cast(pl.Int64), time_unit="s").alias("datetime")
            else:
                dt_expr = pl.col("timestamp").str.to_datetime(strict=False).alias("datetime")
        elif "datekey" in c_names and "timekey" in c_names:
            date_str = pl.col("datekey").cast(pl.String)
            time_str = pl.col("timekey").cast(pl.String).str.pad_start(4, "0")
            dt_expr = (date_str + time_str).str.to_datetime(format="%Y%m%d%H%M", strict=False).alias("datetime")
        else:
            dt_expr = pl.lit(None, dtype=pl.Datetime).alias("datetime")

        exprs = [
            pl.col(id_col).cast(pl.String).alias("trip_id"),
            pl.col("directed_edge_id").cast(pl.String).alias("micro_edge_id"),
            (pl.col(sec_col).cast(pl.Float64) if sec_col else pl.lit(1.0)).alias("seconds"),
            dt_expr,
        ]
        chunk_clean = chunk_pl.select(exprs)

        if carry_df is not None:
            chunk_clean = pl.concat([carry_df, chunk_clean])
            carry_df = None

        if len(chunk_clean) == 0:
            continue

        last_id = chunk_clean["trip_id"][-1]
        mask_ready = chunk_clean["trip_id"] != last_id
        if not mask_ready.any():
            carry_df = chunk_clean
            continue

        df_ready = chunk_clean.filter(mask_ready).with_columns(
            pl.int_range(0, pl.len(), dtype=pl.Int64).alias("row_idx")
        )
        carry_df = chunk_clean.filter(~mask_ready)

        # Map micro-edges to parent canonical macro-edges
        df_mapped = (
            df_ready.lazy()
            .join(pl_lookup, on="micro_edge_id", how="inner")
            .join(pl_map_meta, left_on="macro_edge_id", right_on="directed_edge_id", how="inner")
            .collect()
            .sort("row_idx")
        )

        if len(df_mapped) == 0:
            continue

        # Run-length deduplication: collapse micro-steps into macro-edge traversals
        is_new_macro_edge = (
            (pl.col("macro_edge_id") != pl.col("macro_edge_id").shift(1)) |
            (pl.col("trip_id") != pl.col("trip_id").shift(1)) |
            pl.col("macro_edge_id").shift(1).is_null()
        )

        df_contracted = (
            df_mapped
            .with_columns(is_new_macro_edge.cum_sum().alias("step_group"))
            .group_by("step_group", maintain_order=True)
            .agg([
                pl.col("trip_id").first(),
                pl.col("macro_edge_id").first().alias("directed_edge_id"),
                pl.col("macro_u").first().alias("u"),
                pl.col("macro_v").first().alias("v"),
                pl.col("datetime").first(),
                pl.col("seconds").sum().alias("raw_seconds"),
                pl.col("micro_meters").sum().alias("traversed_micro_m"),
                pl.col("segment_length").first().alias("seg_len"),
                pl.col("speed_limit").first().alias("spd_lim"),
            ])
            # Proportional Duration Normalization:
            # Scales partial boardings/dropoffs to full segment length, preserving measured vehicle velocity
            .with_columns([
                pl.when(pl.col("traversed_micro_m") < (pl.col("seg_len") * 0.90))
                .then(
                    pl.col("raw_seconds") * (pl.col("seg_len") / pl.max_horizontal(pl.col("traversed_micro_m"), pl.lit(2.0)))
                )
                .otherwise(pl.col("raw_seconds"))
                .alias("seconds")
            ])
            .with_columns(
                pl.max_horizontal(pl.col("seconds"), pl.col("seg_len") / (130.0 / 3.6), pl.lit(0.5)).alias("seconds")
            )
        )

        # -----------------------------------------------------
        # Kinematic & Teleportation Filter (Discards >125 km/h trips)
        # -----------------------------------------------------
        plausible_trips = (
            df_contracted
            .group_by("trip_id")
            .agg([
                (pl.col("seg_len").sum() / pl.col("seconds").sum() * 3.6).alias("mean_v"),
                pl.len().alias("n_edges")
            ])
            .filter((pl.col("mean_v") <= 125.0) & (pl.col("n_edges") >= 3))
            .select("trip_id")
        )

        df_filtered = df_contracted.join(plausible_trips, on="trip_id", how="inner")

        # -----------------------------------------------------
        # Continuity Splitter & Boundary Pickup Normalization
        # -----------------------------------------------------
        is_gap = (
            (pl.col("v").shift(1).over("trip_id") != pl.col("u")) &
            pl.col("v").shift(1).over("trip_id").is_not_null()
        )

        df_sequenced = (
            df_filtered
            .with_columns(is_gap.cum_sum().over("trip_id").alias("gap_split"))
            .with_columns(
                pl.when(pl.col("gap_split") == 0)
                .then(pl.col("trip_id"))
                .otherwise(pl.col("trip_id") + "_p" + pl.col("gap_split").cast(pl.String))
                .alias("trip_id")
            )
            .with_columns([
                pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over("trip_id").alias("trip_segmentno"),
                pl.len().over("trip_id").alias("trip_total_steps"),
                pl.col("directed_edge_id").alias("segmentkey"),
                pl.lit("FORWARD").alias("traj_direction")
            ])
            # If the very first step or very last step was a single-point boarding spike, bound to speed limit
            .with_columns([
                pl.when(
                    (pl.col("trip_segmentno") == 1) | (pl.col("trip_segmentno") == pl.col("trip_total_steps"))
                )
                .then(
                    pl.max_horizontal(
                        pl.col("seconds"),
                        pl.col("seg_len") / (pl.max_horizontal(pl.col("spd_lim") * 1.15, pl.lit(45.0)) / 3.6)
                    )
                )
                .otherwise(pl.col("seconds"))
                .alias("seconds")
            ])
            .select([
                "trip_id", "directed_edge_id", "u", "v", "segmentkey",
                "traj_direction", "datetime", "trip_segmentno", "seconds"
            ])
        )

        arrow_table = df_sequenced.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(str(traj_out_tmp), arrow_table.schema, compression="snappy")
        writer.write_table(arrow_table)
        total_written_rows += len(df_sequenced)

        del chunk_pl, chunk_clean, df_ready, df_mapped, df_contracted, df_filtered, df_sequenced, arrow_table

finally:
    pbar.close()
    if writer is not None:
        writer.close()
    if traj_out_tmp.exists():
        os.replace(traj_out_tmp, traj_out)

print(f"\nProcessing Complete!")
print(f"Final Contracted Trajectory Rows: {total_written_rows:,} -> {traj_out}")

# 4. Topological Continuity Verification
print("Verifying continuity across 1,000 trips...")
sample_trips = pl.scan_parquet(traj_out).select("trip_id").unique().limit(1000).collect()["trip_id"]

cont_df = (
    pl.scan_parquet(traj_out)
    .filter(pl.col("trip_id").is_in(sample_trips))
    .sort(["trip_id", "trip_segmentno"])
    .with_columns([
        pl.col("v").alias("curr_v"),
        pl.col("u").shift(-1).over("trip_id").alias("next_u")
    ])
    .filter(pl.col("next_u").is_not_null())
    .select([
        pl.len().alias("total"),
        (pl.col("curr_v") == pl.col("next_u")).sum().alias("continuous")
    ])
    .collect()
)

n_tot, n_cont = cont_df["total"][0], cont_df["continuous"][0]
pct = (n_cont / n_tot * 100.0) if n_tot > 0 else 0.0
print(f"Topological Continuity (v_i == u_{{i+1}}): {n_cont:,} / {n_tot:,} ({pct:.2f}%)")