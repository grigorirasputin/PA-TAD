"""
run_multiseed_getad.py

Runs 3-seed evaluation (seeds 42, 43, 44) for GETAD by executing
GETADRunner.run() end-to-end per seed so ExecutionTracker computational
time, VRAM, and power metrics remain 100% unmodified.
"""

import argparse
import gc
import os
import random
import shutil
import sys
import traceback
from pathlib import Path

# Force unbuffered stdout/stderr so tmux and `tee` display every line immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
# Automatically resolve project root whether in src/04_baselines or src/04_baselines/GETAD
BASE_DIR = SCRIPT_DIR.parents[1] if (SCRIPT_DIR.parents[1] / "data").exists() else SCRIPT_DIR.parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / "getad"
RESULTS_DIR = BASE_DIR / "results" / "anomaly_detection"

sys.path.insert(0, str(BASE_DIR / "src" / "04_baselines"))
sys.path.insert(0, str(SCRIPT_DIR))

from baseline_helpers import (
    evaluate_anomaly_scores,
    evaluate_global_benchmark,
    evaluate_spatial_benchmark,
)
import GETAD as getad_mod

# Ensure GETAD module uses the exact resolved paths
getad_mod.BASE_DIR = BASE_DIR
getad_mod.MODEL_READY_DIR = BASE_DIR / "data" / "06_model_ready"
getad_mod.OUTPUT_DIR = OUTPUT_DIR
getad_mod.CHECKPOINT_DIR = CHECKPOINT_DIR
getad_mod.RESULTS_DIR = RESULTS_DIR


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate_from_npz(npz_path: Path, method_name: str, region: str):
    """Computes standardized Table 1, Spatial, and Tier 3 Global metrics from a saved .npz."""
    data = np.load(npz_path, allow_pickle=True)
    s_scores = data["scores_spatial"]
    y_spatial = data["y_spatial"]

    spatial_bench = evaluate_spatial_benchmark(
        labels=npz_path,
        y_score=s_scores,
        method_name=method_name,
        region=region,
        print_table=False,
    )
    global_bench = evaluate_global_benchmark(
        labels=npz_path,
        method_name=method_name,
        region=region,
        print_results=False,
    )
    sp_all = evaluate_anomaly_scores(
        y_spatial, s_scores, method_name=f"{method_name} (Spatial)", region=region, print_results=False
    )

    return {
        # Table 1 Isolated Spatial
        "detour_auroc": spatial_bench.get("detour_auroc", 0.0),
        "detour_f1": spatial_bench.get("detour_f1", 0.0),
        "detour_auprc": spatial_bench.get("detour_auprc", 0.0),
        "sprw_auroc": spatial_bench.get("sprw_auroc", 0.0),
        "sprw_f1": spatial_bench.get("sprw_f1", 0.0),
        "sprw_auprc": spatial_bench.get("sprw_auprc", 0.0),
        "overshoot_auroc": spatial_bench.get("overshoot_auroc", 0.0),
        "overshoot_f1": spatial_bench.get("overshoot_f1", 0.0),
        "overshoot_auprc": spatial_bench.get("overshoot_auprc", 0.0),
        "spatial_pure_auroc": spatial_bench.get("overall_auroc", 0.0),
        "spatial_pure_f1": spatial_bench.get("overall_f1", 0.0),
        "spatial_pure_auprc": spatial_bench.get("overall_auprc", 0.0),
        # All Spatial
        "spatial_all_auroc": sp_all["auroc"],
        "spatial_all_auprc": sp_all["auprc"],
        "spatial_all_f1": sp_all["f1"],
        # Tier 3 Global
        "global_auroc": global_bench["auroc"],
        "global_auprc": global_bench["auprc"],
        "global_precision": global_bench["precision"],
        "global_recall": global_bench["recall"],
        "global_f1": global_bench["f1"],
    }


def print_multiseed_summary(df_runs: pd.DataFrame, model_name: str, region: str, seeds: list):
    print("\n" + "=" * 95, flush=True)
    print(f"MULTI-SEED SUMMARY ({len(df_runs)} RUNS: {seeds}) - {model_name.upper()} [{region.upper()}]", flush=True)
    print("=" * 95, flush=True)

    def fmt_pct(col):
        m = df_runs[col].mean() * 100.0
        s = df_runs[col].std(ddof=1) * 100.0 if len(df_runs) > 1 else 0.0
        return f"{m:.1f}±{s:.1f}"

    def fmt_dec(col):
        m = df_runs[col].mean()
        s = df_runs[col].std(ddof=1) if len(df_runs) > 1 else 0.0
        return f"{m:.4f} ± {s:.4f}"

    print("\n--- TABLE 1: ISOLATED PURE SPATIAL (AUROC% / F1%) ---", flush=True)
    print(f"  Detour          : {fmt_pct('detour_auroc')} / {fmt_pct('detour_f1')}", flush=True)
    print(f"  SPRW            : {fmt_pct('sprw_auroc')} / {fmt_pct('sprw_f1')}", flush=True)
    print(f"  Overshoot       : {fmt_pct('overshoot_auroc')} / {fmt_pct('overshoot_f1')}", flush=True)
    print(f"  Overall Spatial : {fmt_pct('spatial_pure_auroc')} / {fmt_pct('spatial_pure_f1')}", flush=True)

    print("\n--- TASK-SPECIFIC & TIER 3 GLOBAL METRICS (Mean ± Std) ---", flush=True)
    print(
        f"  Spatial (All)   - AUROC: {fmt_dec('spatial_all_auroc')} | "
        f"AUPRC: {fmt_dec('spatial_all_auprc')} | F1: {fmt_dec('spatial_all_f1')}",
        flush=True,
    )
    print(
        f"  Tier 3 Global   - AUROC: {fmt_dec('global_auroc')} | "
        f"AUPRC: {fmt_dec('global_auprc')} | F1: {fmt_dec('global_f1')}",
        flush=True,
    )
    print(
        f"  Tier 3 Prec/Rec - Prec : {fmt_dec('global_precision')} | Rec  : {fmt_dec('global_recall')}",
        flush=True,
    )
    print("=" * 95 + "\n", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="3-Seed Evaluator for GETAD")
    parser.add_argument("--tag", type=str, required=True, help="dataset tag, e.g. '2016_itsp' or 'porto'")
    parser.add_argument("--regions", "--region", dest="regions", nargs="+", default=None, help="regions to run (defaults to aalborg or porto)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="random seeds to run")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--lr", "--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=5e-4)
    parser.add_argument("--train-epochs", "--train_epochs", "--epochs", dest="train_epochs", type=int, default=150, help="max training epochs (default: 150)")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience (default: 10)")
    parser.add_argument("--early_stopping_delta", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--d_max", type=int, default=None)
    parser.add_argument("--scoring_mode", type=str, default="mean", choices=["mean", "sum", "max"])
    parser.add_argument("--use_cwnll", action="store_true", default=True)
    parser.add_argument("--no_cwnll", dest="use_cwnll", action="store_false")
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--lambda_link", type=float, default=0.01)
    parser.add_argument("--monitor", type=str, default="ce", choices=["ce", "total"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Default only to non-OOM regions (aalborg / porto) unless explicitly overridden
    if args.regions is None:
        active_regions = ["porto"] if "porto" in str(args.tag).lower() else ["aalborg"]
    else:
        active_regions = [r.lower() for r in args.regions]

    display_name = "GETAD"
    file_prefix = "getad"

    print(f"Resolved BASE_DIR: {BASE_DIR}", flush=True)
    print(
        f"Running {display_name} on tag='{args.tag}', regions={active_regions}, "
        f"seeds={args.seeds}, epochs={args.train_epochs}, patience={args.patience}",
        flush=True,
    )

    all_summary_rows = []

    for region in active_regions:
        seed_records = []
        for idx, seed in enumerate(args.seeds):
            print(
                f"\n>>> Running {display_name} | Region: {region.upper()} | Seed: {seed} ({idx + 1}/{len(args.seeds)}) <<<",
                flush=True,
            )
            set_all_seeds(seed)

            run_args = argparse.Namespace(**vars(args))
            run_args.region = region

            try:
                runner = getad_mod.GETADRunner(run_args)
                res = runner.run()
                if res is None:
                    print(f"[{display_name}] Run returned None (OOM/aborted) on region {region}.", flush=True)
                    break
            except Exception as e:
                print(f"Error executing {display_name} on region {region} (seed {seed}): {e}", flush=True)
                traceback.print_exc()
                continue
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Archive per-seed .npz scores
            scores_dir = OUTPUT_DIR / "anomaly_scores"
            default_npz = scores_dir / f"{file_prefix}_{region}_{args.tag}_scores.npz"
            seed_npz = scores_dir / f"{file_prefix}_{region}_{args.tag}_seed{seed}_scores.npz"

            # Archive per-seed computational_time parquet
            times_dir = OUTPUT_DIR / "computational_time"
            default_time_pq = times_dir / f"{file_prefix}_{region}_times.parquet"
            seed_time_pq = times_dir / f"{file_prefix}_{region}_{args.tag}_seed{seed}_times.parquet"
            if default_time_pq.exists():
                shutil.copy2(default_time_pq, seed_time_pq)

            if default_npz.exists():
                shutil.copy2(default_npz, seed_npz)
                metrics = evaluate_from_npz(seed_npz, display_name, region)
                metrics["model"] = display_name
                metrics["region"] = region
                metrics["tag"] = args.tag
                metrics["seed"] = seed
                seed_records.append(metrics)
                all_summary_rows.append(metrics)
            else:
                print(f"WARNING: Expected score file not found at {default_npz}", flush=True)

        if seed_records:
            df_region = pd.DataFrame(seed_records)
            print_multiseed_summary(df_region, display_name, region, [r["seed"] for r in seed_records])

    if all_summary_rows:
        df_all = pd.DataFrame(all_summary_rows)
        raw_out = RESULTS_DIR / f"multiseed_getad_runs_{args.tag}.parquet"
        csv_out = RESULTS_DIR / f"multiseed_getad_summary_{args.tag}.csv"
        df_all.to_parquet(raw_out, index=False)

        metric_cols = [c for c in df_all.columns if c not in ("model", "region", "tag", "seed")]
        df_agg = df_all.groupby(["model", "region", "tag"], as_index=False)[metric_cols].agg(["mean", "std"])
        df_agg.columns = ["_".join(col).strip("_") for col in df_agg.columns.values]
        df_agg.to_csv(csv_out, index=False)
        print(f"Saved per-seed metrics to: {raw_out}", flush=True)
        print(f"Saved aggregated Mean/Std summary to: {csv_out}", flush=True)