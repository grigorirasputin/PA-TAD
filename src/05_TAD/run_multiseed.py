"""
run_multiseed.py

Runs 3-seed evaluation (seeds 42, 43, 44) for PA-TAD-Lite and PA-TAD,
archives per-seed .npz score files and .parquet timing files,
and outputs Mean ± Std tables for anomaly detection and scalability.
"""

import argparse
import importlib.util
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results" / "anomaly_detection"
TIMES_DIR = OUTPUT_DIR / "computational_time"
SCORES_DIR = OUTPUT_DIR / "anomaly_scores"

sys.path.append(str(BASE_DIR / "src" / "04_baselines"))
sys.path.append(str(SCRIPT_DIR))

from PA_TAD_core.tools import set_all_seeds
from baseline_helpers import (
    evaluate_anomaly_scores,
    evaluate_global_benchmark,
    evaluate_spatial_benchmark,
)


def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




def evaluate_from_npz(npz_path: Path, method_name: str, region: str):
    data = np.load(npz_path, allow_pickle=True)
    s_scores = data["scores_spatial"]
    t_scores = data["scores_temporal"]
    y_spatial = data["y_spatial"]
    y_temporal = data["y_temporal"]

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
    tp_all = evaluate_anomaly_scores(
        y_temporal, t_scores, method_name=f"{method_name} (Temporal)", region=region, print_results=False
    )
    cls = np.array([str(c).strip().lower() for c in data["anomaly_class"]])
    tp_mask = np.isin(cls, ["normal", "temporal"])
    tp_pure = evaluate_anomaly_scores(
        (cls[tp_mask] == "temporal").astype(int), t_scores[tp_mask],
        method_name=f"{method_name} (Temporal, pure)", region=region, print_results=False,
    )
    cp_mask = np.isin(cls, ["normal", "spatio_temporal"])
    cp_pure = evaluate_anomaly_scores(
        (cls[cp_mask] == "spatio_temporal").astype(int), np.maximum(s_scores, t_scores)[cp_mask],
        method_name=f"{method_name} (Compound)", region=region, print_results=False,
    )

    return {
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
        "spatial_all_auroc": sp_all["auroc"],
        "spatial_all_auprc": sp_all["auprc"],
        "spatial_all_f1": sp_all["f1"],
        "temporal_all_auroc": tp_all["auroc"],
        "temporal_all_auprc": tp_all["auprc"],
        "temporal_all_f1": tp_all["f1"],
        "temporal_pure_auroc": tp_pure["auroc"],
        "temporal_pure_auprc": tp_pure["auprc"],
        "temporal_pure_f1": tp_pure["f1"],
        "compound_pure_auroc": cp_pure["auroc"],
        "compound_pure_auprc": cp_pure["auprc"],
        "compound_pure_f1": cp_pure["f1"],
        "global_auroc": global_bench["auroc"],
        "global_auprc": global_bench["auprc"],
        "global_precision": global_bench["precision"],
        "global_recall": global_bench["recall"],
        "global_f1": global_bench["f1"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="3-Seed Evaluator for PA-TAD-Lite and PA-TAD")
    parser.add_argument("--tag", required=True, help="dataset tag, e.g. '2016_itsp' or 'porto'")
    parser.add_argument("--regions", "--region", dest="regions", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=["pa_tad_lite", "pa_tad"],
                        choices=["pa_tad_lite", "pa_tad", "ct_tad", "pact_tad"],
                        help="models to run (ct_tad / pact_tad are accepted as legacy aliases)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="random seeds to run")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--prior-epochs", dest="prior_epochs", type=int, default=200)
    parser.add_argument("--force-retrain-prior", dest="force_retrain_prior", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early_stopping_delta", type=float, default=0.001)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    TIMES_DIR.mkdir(parents=True, exist_ok=True)

    if args.regions is None:
        active_regions = ["porto"] if "porto" in str(args.tag).lower() else ["aalborg", "copenhagen", "denmark"]
    else:
        active_regions = [r.lower() for r in args.regions]

    legacy_alias = {"ct_tad": "pa_tad_lite", "pact_tad": "pa_tad"}
    args.models = list(dict.fromkeys(legacy_alias.get(m, m) for m in args.models))
    lite_mod = load_module_from_path("pa_tad_lite_mod", SCRIPT_DIR / "PA-TAD-Lite.py") if "pa_tad_lite" in args.models else None
    full_mod = load_module_from_path("pa_tad_mod", SCRIPT_DIR / "PA-TAD.py") if "pa_tad" in args.models else None

    all_summary_rows = []

    for model_key in args.models:
        is_full = model_key == "pa_tad"
        display_name = "PA-TAD" if is_full else "PA-TAD-Lite"
        score_prefix = "pa_tad" if is_full else "pa_tad_lite"
        time_prefixes = ["pa-tad"] if is_full else ["pa-tad-lite"]

        for region in active_regions:
            seed_records = []
            for idx, seed in enumerate(args.seeds):
                print(f"\n>>> Running {display_name} | Region: {region.upper()} | Seed: {seed} ({idx + 1}/{len(args.seeds)}) <<<")
                set_all_seeds(seed)

                run_args = argparse.Namespace(**vars(args))
                # Only retrain prior on the very first seed if requested; reuse cached weights thereafter
                if is_full and idx > 0:
                    run_args.force_retrain_prior = False

                runner = full_mod.PATADRunner(run_args, region) if is_full else lite_mod.PATADLiteRunner(run_args, region)
                runner.run()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 1. Archive Anomaly Scores
                default_npz = SCORES_DIR / f"{score_prefix}_{region}_{args.tag}_scores_test.npz"
                seed_npz = SCORES_DIR / f"{score_prefix}_{region}_{args.tag}_seed{seed}_scores_test.npz"
                if default_npz.exists():
                    shutil.copy2(default_npz, seed_npz)
                    metrics = evaluate_from_npz(seed_npz, display_name, region)
                    metrics["model"] = display_name
                    metrics["region"] = region
                    metrics["tag"] = args.tag
                    metrics["seed"] = seed
                    seed_records.append(metrics)
                    all_summary_rows.append(metrics)

                # 2. Archive Computational Times (check both hyphen and underscore variants)
                for tp in time_prefixes:
                    default_time = TIMES_DIR / f"{tp}_{region}_times.parquet"
                    if default_time.exists():
                        seed_time = TIMES_DIR / f"{tp}_{region}_{args.tag}_seed{seed}_times.parquet"
                        shutil.copy2(default_time, seed_time)
                        print(f"Archived timing parquet: {seed_time.name}")
                        break

            if seed_records:
                df_region = pd.DataFrame(seed_records)
                print(f"\n=== MULTI-SEED ANOMALY DETECTION SUMMARY: {display_name} [{region.upper()}] ===")
                for col in ["spatial_pure_auroc", "spatial_pure_f1", "temporal_pure_auroc", "temporal_pure_f1", "compound_pure_auroc", "compound_pure_f1"]:
                    m = df_region[col].mean() * 100.0
                    s = df_region[col].std(ddof=1) * 100.0 if len(df_region) > 1 else 0.0
                    print(f"  {col:<22}: {m:.1f} ± {s:.1f}")

    if all_summary_rows:
        df_all = pd.DataFrame(all_summary_rows)
        raw_out = RESULTS_DIR / f"multiseed_tad_runs_{args.tag}.parquet"
        csv_out = RESULTS_DIR / f"multiseed_tad_summary_{args.tag}.csv"
        df_all.to_parquet(raw_out, index=False)
        print(f"\nSaved raw multi-seed runs to: {raw_out}")
        metric_cols = [c for c in df_all.columns if c not in ("model", "region", "tag", "seed")]
        summary = df_all.groupby(["model", "region"])[metric_cols].agg(["mean", "std"])
        summary.to_csv(csv_out)
        print(f"Saved mean/std summary to: {csv_out}")