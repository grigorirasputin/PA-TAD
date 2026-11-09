#!/usr/bin/env python3
"""
ablation_multiseed.py

Multi-seed component and feature ablation study on PA-TAD-Lite (seeds 42-46, as in the paper).
Evaluates 7 variants across Porto, Aalborg, and Copenhagen:
  1. base           : Full PA-TAD-Lite with Gated Ego-Residual GAT
  2. no_gat         : w/o Topological GAT (Pure Ego-MLP: in -> 128 -> ELU -> 64)
  3. no_tau         : w/o tau (masks the 6-D harmonic time-of-week features)
  4. no_latlon      : w/o lat,lon (masks centroid_lon, centroid_lat, bearing_sin, bearing_cos)
  5. no_road_cat    : w/o Road Category (masks one-hot functional road hierarchy columns)
  6. no_kinematics  : w/o Kinematics (masks segment_length and speed_limit columns)
  7. uni_gru        : w/ Uni-GRU (replaces Bidirectional GRU with forward-only GRU)
"""

import argparse
import random
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[1]
MODEL_READY_DIR = BASE_DIR / "data" / "06_model_ready"
SYNTH_DIR = BASE_DIR / "data" / "05_synthetic"
OUTPUT_DIR = BASE_DIR / "outputs"
ABLATION_SCORES_DIR = OUTPUT_DIR / "anomaly_scores" / "ablation"
MAIN_SCORES_DIR = OUTPUT_DIR / "anomaly_scores"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / "pa_tad_lite_ablation"
RESULTS_DIR = BASE_DIR / "results" / "anomaly_detection"
FIGURES_DIR = RESULTS_DIR / "figures"

sys.path.append(str(BASE_DIR / "src" / "04_baselines"))
sys.path.append(str(SCRIPT_DIR))

from baseline_helpers import (
    evaluate_anomaly_scores,
    evaluate_spatial_benchmark,
)
from PA_TAD_core import (
    EarlyStopping,
    MultiTaskLossWrapper,
    TrajectoryDataset,
    build_graph_and_mapping,
    collate_fn,
    evaluate_model,
    extract_spatial_subtype,
)
from PA_TAD_core.tools import set_all_seeds, setup_logger

# Canonical Ablation Registry
ABLATION_VARIANTS = OrderedDict([
    ("base", "PA-TAD-Lite"),
    ("no_gat", "w/o Topological GAT"),
    ("no_tau", r"w/o $\tau$"),
    ("no_latlon", "w/o lat,lon"),
    ("no_road_cat", "w/o Road Category"),
    ("no_kinematics", "w/o kinematics"),
    ("uni_gru", "w Uni-GRU"),
])


def main_base_archive(region: str, tag: str, seed: int) -> Path:
    """Score archive of the main PA-TAD-Lite run (falls back to the pre-rename ct_tad_beta_* name)."""
    new = MAIN_SCORES_DIR / f"pa_tad_lite_{region}_{tag}_seed{seed}_scores_test.npz"
    legacy = MAIN_SCORES_DIR / f"ct_tad_beta_{region}_{tag}_seed{seed}_scores_test.npz"
    return new if new.exists() or not legacy.exists() else legacy


# =====================================================================
# 1. MODULAR ABLATION MODEL ARCHITECTURE
# =====================================================================
class AblatedPATADLite(nn.Module):
    """
    PA-TAD-Lite with Gated Ego-Residual GAT and clean toggles for each ablation variant.
    Feature layout in real_feats (from build_graph_and_mapping):
      - Col 0: agg_length     } Kinematics (0:2)
      - Col 1: agg_speed      }
      - Col 2: centroid_lon   }
      - Col 3: centroid_lat   } Coordinate & Directional Geometry (2:6)
      - Col 4: bearing_sin    }
      - Col 5: bearing_cos    }
      - Col 6+: one-hot road category (6:)
    """
    def __init__(self, num_nodes, real_feats, cfg, variant="base"):
        super().__init__()
        self.variant = variant

        # Clone features and apply feature-level masks without altering tensor dimensions
        mod_feats = real_feats.clone().detach()
        if variant == "no_kinematics":
            mod_feats[:, 0:2] = 0.0
        elif variant == "no_latlon":
            mod_feats[:, 2:6] = 0.0
        elif variant == "no_road_cat":
            mod_feats[:, 6:] = 0.0

        self.real_feats = nn.Parameter(mod_feats, requires_grad=False)
        in_dim = mod_feats.shape[1]
        emb_dim = cfg["node_emb_dim"]  # 64

        # 1. Ego-Node Feature Projection (shared across all variants)
        self.ego_mlp1 = nn.Linear(in_dim, emb_dim * 2)
        self.ego_mlp2 = nn.Linear(emb_dim * 2, emb_dim)

        # 2. Topological GAT Context (disabled only in 'no_gat')
        if variant != "no_gat":
            self.gat1 = GATConv(in_dim, emb_dim, heads=2, concat=True)
            self.gat2 = GATConv(emb_dim * 2, emb_dim, heads=1, concat=False)
            self.topo_gate = nn.Parameter(torch.full((1, emb_dim), 0.05))

        # Sequence Backbone: Bi-GRU vs Uni-GRU
        is_bidirectional = (variant != "uni_gru")
        gru_in_dim = emb_dim + 1 + cfg["time_dim"]
        self.gru = nn.GRU(
            gru_in_dim,
            cfg["hidden_dim"],
            cfg["lstm_layers"],
            batch_first=True,
            dropout=0.2 if cfg["lstm_layers"] > 1 else 0.0,
            bidirectional=is_bidirectional,
        )

        gru_out_dim = cfg["hidden_dim"] * (2 if is_bidirectional else 1)
        adapter_dim = cfg["adapter_dim"]

        # Disentangled Projection Adapters
        self.spatial_adapter = nn.Sequential(
            nn.Linear(gru_out_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.GELU(),
        )
        self.temporal_adapter = nn.Sequential(
            nn.Linear(gru_out_dim, adapter_dim),
            nn.LayerNorm(adapter_dim),
            nn.GELU(),
        )

        # Classification Heads
        self.s_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(adapter_dim, 1),
        )
        self.t_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(adapter_dim, 1),
        )

    def forward(self, nodes, secs, time_feats, edge_idx, lengths):
        # 1. Node Embedding (Ego + Gated GAT vs Pure Ego-MLP)
        ego_embs = self.ego_mlp2(F.elu(self.ego_mlp1(self.real_feats)))
        if self.variant == "no_gat":
            n_embs = ego_embs[nodes]
        else:
            topo_embs = self.gat2(F.elu(self.gat1(self.real_feats, edge_idx)), edge_idx)
            n_embs = (ego_embs + self.topo_gate * topo_embs)[nodes]

        # 2. Temporal Context Toggle (w/o tau)
        if self.variant == "no_tau":
            time_in = torch.zeros_like(time_feats)
        else:
            time_in = time_feats

        # 3. Sequence Modeling
        packed, _ = self.gru(
            nn.utils.rnn.pack_padded_sequence(
                torch.cat([n_embs, secs, time_in], dim=-1),
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
        )
        out, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=nodes.size(1))

        # 4. Disentangled Heads
        s_feat = self.spatial_adapter(out)
        t_feat = self.temporal_adapter(out)
        s_logits = self.s_head(s_feat).squeeze(-1)
        t_logits = self.t_head(t_feat).squeeze(-1)
        return s_logits, t_logits


# =====================================================================
# 2. DATA RESOLUTION & EVALUATION HELPERS
# =====================================================================
def resolve_region_paths(region: str, tag: str):
    data_dir = MODEL_READY_DIR / tag
    synth_dir = SYNTH_DIR / tag

    g_candidates = [
        data_dir / f"model_ready_map_df_{region}.parquet",
        synth_dir / f"map_df_{region}.parquet",
        BASE_DIR / "data" / "03_processed" / f"map_directed_{tag}.parquet",
    ]
    t_candidates = [
        data_dir / f"model_ready_trajectories_{region}.parquet",
        synth_dir / f"synthetic_trajectories_{region}.parquet",
    ]

    g_path = next((p for p in g_candidates if p.exists()), None)
    t_path = next((p for p in t_candidates if p.exists()), None)
    if g_path is None or t_path is None:
        raise FileNotFoundError(f"Could not find map or trajectory parquet for region={region}, tag={tag}")
    return g_path, t_path


def build_test_metadata_lookup(t_path: Path, test_ds):
    traj_df = pd.read_parquet(t_path)
    if "split" in traj_df.columns:
        test_df = traj_df[traj_df["split"] == "test"]
    else:
        test_trip_set = set(test_ds.trips[i]["trip_id"] for i in range(len(test_ds)))
        test_df = traj_df[traj_df["trip_id"].isin(test_trip_set)]

    test_meta_lookup = {}
    for tid, grp in test_df.groupby("trip_id", sort=False):
        a_cls = grp["anomaly_class"].iloc[0] if "anomaly_class" in grp.columns else "normal"
        sp_max = int(grp["spatial_anomaly"].max()) if "spatial_anomaly" in grp.columns else 0
        tp_max = int(grp["temporal_anomaly"].max()) if "temporal_anomaly" in grp.columns else 0
        st_sub = extract_spatial_subtype(grp["spatial_subtype"]) if "spatial_subtype" in grp.columns else "None"
        y_bin = 0 if str(a_cls).lower() == "normal" else 1
        test_meta_lookup[str(tid)] = {
            "anomaly_class": str(a_cls),
            "spatial_anomaly": sp_max,
            "temporal_anomaly": tp_max,
            "spatial_subtype": st_sub,
            "label": y_bin,
        }
    return test_meta_lookup


def evaluate_ablation_npz(npz_path: Path, variant_key: str, region: str):
    """
    Evaluates a saved .npz archive using the exact Table 1 (Pure Spatial Isolation)
    and Table 2 (Temporal) protocols. Also returns raw arrays for PR curve plotting.
    """
    data = np.load(npz_path, allow_pickle=True)
    s_scores = data["scores_spatial"]
    t_scores = data["scores_temporal"]
    cls_lower = np.array([str(x).strip().lower() for x in data["anomaly_class"]])
    tp_mask = np.isin(cls_lower, ["normal", "temporal"])  # temporal-only vs normal (Table VI protocol)
    y_tp_pure = (cls_lower[tp_mask] == "temporal").astype(int)
    t_scores_pure = t_scores[tp_mask]

    # 1. Pure Spatial Evaluation (Table 1 protocol)
    sb = evaluate_spatial_benchmark(
        labels=npz_path,
        y_score=s_scores,
        method_name=variant_key,
        region=region,
        print_table=False,
    )

    # 2. Temporal Evaluation (Table 2 protocol)
    tb = evaluate_anomaly_scores(
        y_true=y_tp_pure,
        y_score=t_scores_pure,
        method_name=variant_key,
        region=region,
        print_results=False,
    )

    # Extract isolated pure spatial mask for exact Spatial PR curve
    a_cls = np.array([str(x).strip().lower() for x in data["anomaly_class"]])
    st_sub = np.array([str(x).strip().lower() for x in data["spatial_subtype"]])
    alias_map = {"over": "overshoot", "none": "normal", "nan": "normal", "": "normal"}
    combined_lbl = []
    for st, ac in zip(st_sub, a_cls):
        st_c = alias_map.get(st, st)
        if st_c == "normal":
            combined_lbl.append(alias_map.get(ac, ac))
        else:
            combined_lbl.append(st_c)
    combined_lbl = np.array(combined_lbl)
    sp_mask = ((combined_lbl == "normal") | np.isin(combined_lbl, ["detour", "sprw", "overshoot"])) & (a_cls != "spatio_temporal")
    y_sp_pure = (combined_lbl[sp_mask] != "normal").astype(int)
    s_scores_pure = s_scores[sp_mask]

    metrics = {
        "region": region,
        "variant": variant_key,
        "display_name": ABLATION_VARIANTS[variant_key],
        "s_auroc": sb.get("overall_auroc", 0.0) * 100.0,
        "s_auprc": sb.get("overall_auprc", 0.0) * 100.0,
        "s_f1": sb.get("overall_f1", 0.0) * 100.0,
        "t_auroc": tb.get("auroc", 0.0) * 100.0,
        "t_auprc": tb.get("auprc", 0.0) * 100.0,
        "t_f1": tb.get("f1", 0.0) * 100.0,
    }
    curves_data = {
        "y_sp_pure": y_sp_pure,
        "s_scores_pure": s_scores_pure,
        "y_tp": y_tp_pure,
        "t_scores": t_scores_pure,
    }
    return metrics, curves_data


# =====================================================================
# 3. PLOTTING: 7-CURVE PRECISION-RECALL (AUPRC) FIGURES
# =====================================================================
def plot_region_pr_curves(region: str, curve_store: dict):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), dpi=300)

    style_map = {
        "base":          {"color": "#111111", "ls": "-",  "lw": 2.4, "zorder": 10},
        "no_gat":        {"color": "#1f77b4", "ls": "--", "lw": 1.8, "zorder": 5},
        "no_tau":        {"color": "#ff7f0e", "ls": "-.", "lw": 1.8, "zorder": 6},
        "no_latlon":     {"color": "#2ca02c", "ls": ":",  "lw": 1.9, "zorder": 5},
        "no_road_cat":   {"color": "#d62728", "ls": "--", "lw": 1.9, "zorder": 7},
        "no_kinematics": {"color": "#9467bd", "ls": "-.", "lw": 2.0, "zorder": 8},
        "uni_gru":       {"color": "#8c564b", "ls": ":",  "lw": 1.8, "zorder": 4},
    }

    for v_key, disp_name in ABLATION_VARIANTS.items():
        if v_key not in curve_store or not curve_store[v_key]:
            continue

        runs = curve_store[v_key]
        y_sp_all = np.concatenate([r["y_sp_pure"] for r in runs])
        s_sc_all = np.concatenate([r["s_scores_pure"] for r in runs])
        y_tp_all = np.concatenate([r["y_tp"] for r in runs])
        t_sc_all = np.concatenate([r["t_scores"] for r in runs])

        mean_s_auprc = np.mean([average_precision_score(r["y_sp_pure"], r["s_scores_pure"]) for r in runs]) * 100.0
        mean_t_auprc = np.mean([average_precision_score(r["y_tp"], r["t_scores"]) for r in runs]) * 100.0

        st = style_map.get(v_key, {"color": "gray", "ls": "-", "lw": 1.5, "zorder": 3})
        clean_label = disp_name.replace(r"$\tau$", "\u03c4")

        prec_s, rec_s, _ = precision_recall_curve(y_sp_all, s_sc_all)
        axes[0].plot(
            rec_s, prec_s,
            label=f"{clean_label} ({mean_s_auprc:.1f}%)",
            color=st["color"], linestyle=st["ls"], linewidth=st["lw"], zorder=st["zorder"]
        )

        prec_t, rec_t, _ = precision_recall_curve(y_tp_all, t_sc_all)
        axes[1].plot(
            rec_t, prec_t,
            label=f"{clean_label} ({mean_t_auprc:.1f}%)",
            color=st["color"], linestyle=st["ls"], linewidth=st["lw"], zorder=st["zorder"]
        )

    axes[0].set_title(f"{region.capitalize()} — Spatial Anomaly PR Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Recall", fontsize=11)
    axes[0].set_ylabel("Precision", fontsize=11)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.02])
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="lower left", fontsize=8.5, frameon=True)

    axes[1].set_title(f"{region.capitalize()} — Temporal Anomaly PR Curve", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.02])
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="lower left", fontsize=8.5, frameon=True)

    plt.tight_layout()
    pdf_out = FIGURES_DIR / f"ablation_pr_curves_{region}.pdf"
    png_out = FIGURES_DIR / f"ablation_pr_curves_{region}.png"
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.savefig(png_out, bbox_inches="tight")
    plt.close(fig)
    print(f"[{region.upper()}] Saved 7-curve AUPRC plots to: {pdf_out} and {png_out}")


# =====================================================================
# 4. LATEX TABLE GENERATOR
# =====================================================================
def generate_latex_ablation_table(df_all: pd.DataFrame, regions: list):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Component and Feature Ablation Study on PA-TAD-Lite Across Regions}")
    lines.append(r"\label{tab:component_ablation_dynamic}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Region} & \textbf{Architecture} & \textbf{S-AUROC} & \textbf{S-F1} & \textbf{T-AUROC} & \textbf{T-F1} \\")
    lines.append(r"\midrule")

    for r_idx, reg in enumerate(regions):
        sub = df_all[df_all["region"] == reg]
        if sub.empty:
            continue

        agg = sub.groupby("variant", sort=False)[["s_auroc", "s_f1", "t_auroc", "t_f1"]].mean()
        best_vals = {col: agg[col].max() for col in ["s_auroc", "s_f1", "t_auroc", "t_f1"]}

        lines.append(f"\\textbf{{{reg.capitalize()}}} ")
        for v_key, disp_name in ABLATION_VARIANTS.items():
            if v_key not in agg.index:
                lines.append(f"        & {disp_name:<20} & -- & -- & -- & -- \\\\")
                continue

            row = agg.loc[v_key]
            cells = []
            for col in ["s_auroc", "s_f1", "t_auroc", "t_f1"]:
                val = row[col]
                val_str = f"{val:.1f}"
                if abs(val - best_vals[col]) < 1e-4:
                    cells.append(f"\\textbf{{{val_str}}}")
                else:
                    cells.append(val_str)

            lines.append(f"        & {disp_name:<20} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\\\")

        if r_idx < len(regions) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =====================================================================
# 5. MAIN EXECUTION LOOP
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PA-TAD-Lite multi-seed ablation study runner")
    parser.add_argument("--regions", nargs="+", default=["porto", "aalborg", "copenhagen"], help="Regions to evaluate")
    parser.add_argument("--variants", nargs="+", default=list(ABLATION_VARIANTS.keys()), choices=list(ABLATION_VARIANTS.keys()), help="Ablation variants to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="Random seeds (paper: 42-46)")
    parser.add_argument("--rerun-base", action="store_true", help="Force re-training base PA-TAD-Lite even if .npz files exist")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only evaluate existing ablation .npz files + plot PR curves")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early_stopping_delta", type=float, default=0.001)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    ABLATION_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model_cfg = {
        "node_emb_dim": 64,
        "time_dim": 6,  # must match the 6-D harmonic features built by TrajectoryDataset
        "hidden_dim": 128,
        "adapter_dim": 64,
        "lstm_layers": 2,
    }

    all_metrics = []

    for region in [r.lower() for r in args.regions]:
        tag = "porto" if region == "porto" else "2016_itsp"
        logger = setup_logger(f"ablation_{region}", tag, log_dir=OUTPUT_DIR / "logs" / "pa_tad_lite_ablation")
        region_curves = {v: [] for v in ABLATION_VARIANTS.keys()}

        need_training = False
        if not args.eval_only:
            for v_key in args.variants:
                for seed in args.seeds:
                    if v_key == "base" and not args.rerun_base:
                        main_npz = main_base_archive(region, tag, seed)
                        abl_npz = ABLATION_SCORES_DIR / f"ablation_base_{region}_{tag}_seed{seed}_scores_test.npz"
                        if abl_npz.exists() or main_npz.exists():
                            continue
                    need_training = True
                    break

        if need_training:
            g_path, t_path = resolve_region_paths(region, tag)
            logger.info(f"[{region.upper()}] Loading graph ({g_path.name}) and trajectories ({t_path.name}) once...")
            edge_index, num_nodes, node2idx, real_features = build_graph_and_mapping(g_path)
            edge_index = edge_index.to(device)
            real_features = real_features.to(device)

            train_ds = TrajectoryDataset(t_path, node2idx, "train")
            val_ds = TrajectoryDataset(t_path, node2idx, "val")
            test_ds = TrajectoryDataset(t_path, node2idx, "test")
            test_meta_lookup = build_test_metadata_lookup(t_path, test_ds)

            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
            )

        for v_key in args.variants:
            disp_name = ABLATION_VARIANTS[v_key]
            for seed in args.seeds:
                abl_npz = ABLATION_SCORES_DIR / f"ablation_{v_key}_{region}_{tag}_seed{seed}_scores_test.npz"
                main_base_npz = main_base_archive(region, tag, seed)

                # Reuse existing base run if available (prefer abl_npz if trained with --rerun-base)
                if v_key == "base" and not args.rerun_base and (abl_npz.exists() or main_base_npz.exists()):
                    target_npz = abl_npz if abl_npz.exists() else main_base_npz
                    logger.info(f"[{region.upper()} | {disp_name} | Seed {seed}] Reusing archive: {target_npz.name}")
                    m, c_data = evaluate_ablation_npz(target_npz, v_key, region)
                    m["seed"] = seed
                    all_metrics.append(m)
                    region_curves[v_key].append(c_data)
                    continue

                if args.eval_only:
                    target_npz = abl_npz if abl_npz.exists() else (main_base_npz if v_key == "base" and main_base_npz.exists() else None)
                    if target_npz is not None:
                        m, c_data = evaluate_ablation_npz(target_npz, v_key, region)
                        m["seed"] = seed
                        all_metrics.append(m)
                        region_curves[v_key].append(c_data)
                    continue

                # Train Ablation Variant
                logger.info(f"\n>>> Training Ablation: {disp_name} ({v_key}) | Region: {region.upper()} | Seed: {seed} <<<")
                set_all_seeds(seed)

                train_loader = DataLoader(
                    train_ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                )

                model = AblatedPATADLite(num_nodes, real_features, model_cfg, variant=v_key).to(device)
                loss_wrapper = MultiTaskLossWrapper().to(device)
                trainable_params = [p for p in model.parameters() if p.requires_grad] + list(loss_wrapper.parameters())
                opt = torch.optim.Adam(trainable_params, lr=args.learning_rate)

                early_stopping = EarlyStopping(
                    logger=logger, patience=args.patience, delta=args.early_stopping_delta, mode="max"
                )
                ckpt_filename = f"ablation_{v_key}_{region}_{tag}_seed{seed}_best.pt"

                for ep in range(1, args.epochs + 1):
                    model.train()
                    loss_wrapper.train()
                    for b in train_loader:
                        nodes, secs, time_f, s_l, t_l, sub, mask, lens, _, _ = b
                        nodes = nodes.to(device)
                        secs = secs.to(device)
                        time_f = time_f.to(device)
                        s_l = s_l.to(device)
                        t_l = t_l.to(device)
                        mask = mask.to(device)

                        opt.zero_grad(set_to_none=True)
                        s_log, t_log = model(nodes, secs, time_f, edge_index, lens)
                        loss, _, _ = loss_wrapper(s_log, t_log, s_l, t_l, mask)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
                        opt.step()

                    vm = evaluate_model(model, val_loader, edge_index, device)
                    joint_f1 = (vm["s_f1_pure"] + vm["t_f1_pure"]) / 2.0

                    early_stopping(
                        joint_f1, model, CHECKPOINT_DIR / tag, ckpt_filename,
                        extra_state={"s_th": vm["s_th"], "t_th": vm["t_th"]},
                    )
                    if early_stopping.early_stop:
                        logger.info(f"[{v_key} | Seed {seed}] Early stopping at epoch {ep} (Best Val Joint-F1: {early_stopping.best_score:.4f})")
                        break

                # Restore Best Checkpoint & Evaluate on Test Split
                best_ckpt = torch.load(early_stopping.best_checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(best_ckpt["state"])
                tm = evaluate_model(
                    model, test_loader, edge_index, device,
                    th_s=best_ckpt.get("s_th", None), th_t=best_ckpt.get("t_th", None),
                )

                test_trip_ids = tm.get("trip_ids", [t["trip_id"] for t in test_ds.trips])
                s_scores = tm.get("s_scores", np.zeros(len(test_trip_ids), dtype=np.float32))
                t_scores = tm.get("t_scores", np.zeros(len(test_trip_ids), dtype=np.float32))
                global_scores = np.maximum(s_scores, t_scores)

                y_true = np.array([test_meta_lookup.get(tid, {}).get("label", 0) for tid in test_trip_ids], dtype=np.int32)
                y_spatial = np.array([test_meta_lookup.get(tid, {}).get("spatial_anomaly", 0) for tid in test_trip_ids], dtype=np.int32)
                y_temporal = np.array([test_meta_lookup.get(tid, {}).get("temporal_anomaly", 0) for tid in test_trip_ids], dtype=np.int32)
                anomaly_classes = np.array([test_meta_lookup.get(tid, {}).get("anomaly_class", "normal") for tid in test_trip_ids], dtype=object)
                spatial_subtypes = np.array([test_meta_lookup.get(tid, {}).get("spatial_subtype", "None") for tid in test_trip_ids], dtype=object)

                np.savez_compressed(
                    abl_npz,
                    trip_id=np.array(test_trip_ids, dtype=str),
                    scores=global_scores,
                    scores_spatial=s_scores,
                    scores_temporal=t_scores,
                    y_true=y_true,
                    y_spatial=y_spatial,
                    y_temporal=y_temporal,
                    anomaly_class=anomaly_classes,
                    spatial_subtype=spatial_subtypes,
                )

                m, c_data = evaluate_ablation_npz(abl_npz, v_key, region)
                m["seed"] = seed
                all_metrics.append(m)
                region_curves[v_key].append(c_data)

                logger.info(
                    f"Result [{region.upper()} | {disp_name} | Seed {seed}] -> "
                    f"S-AUROC: {m['s_auroc']:.1f} | S-F1: {m['s_f1']:.1f} | "
                    f"T-AUROC: {m['t_auroc']:.1f} | T-F1: {m['t_f1']:.1f}"
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Plot 7-curve AUPRC figure for this region
        if any(len(v) > 0 for v in region_curves.values()):
            plot_region_pr_curves(region, region_curves)

    # Save & Print Final Summary + LaTeX Table
    if all_metrics:
        df_all = pd.DataFrame(all_metrics)
        csv_path = RESULTS_DIR / "ablation_study_results.csv"
        df_all.to_csv(csv_path, index=False)

        print("\n" + "=" * 100)
        print("3-SEED ABLATION STUDY SUMMARY (MEAN ± STD)")
        print("=" * 100)
        for reg in [r.lower() for r in args.regions]:
            sub = df_all[df_all["region"] == reg]
            if sub.empty:
                continue
            print(f"\n--- {reg.upper()} ---")
            for v_key, disp_name in ABLATION_VARIANTS.items():
                v_df = sub[sub["variant"] == v_key]
                if v_df.empty:
                    continue
                n = len(v_df)
                sa_m, sa_s = v_df["s_auroc"].mean(), v_df["s_auroc"].std(ddof=1) if n > 1 else 0.0
                sf_m, sf_s = v_df["s_f1"].mean(), v_df["s_f1"].std(ddof=1) if n > 1 else 0.0
                ta_m, ta_s = v_df["t_auroc"].mean(), v_df["t_auroc"].std(ddof=1) if n > 1 else 0.0
                tf_m, tf_s = v_df["t_f1"].mean(), v_df["t_f1"].std(ddof=1) if n > 1 else 0.0
                print(
                    f"  {disp_name:<22} (n={n}) | "
                    f"S-AUROC: {sa_m:5.1f}±{sa_s:4.1f} | S-F1: {sf_m:5.1f}±{sf_s:4.1f} | "
                    f"T-AUROC: {ta_m:5.1f}±{ta_s:4.1f} | T-F1: {tf_m:5.1f}±{tf_s:4.1f}"
                )

        latex_str = generate_latex_ablation_table(df_all, [r.lower() for r in args.regions])
        tex_path = RESULTS_DIR / "table_ablation_verified.tex"
        tex_path.write_text(latex_str)
        print("\n" + "%" * 60)
        print("% PASTE-READY LATEX ABLATION TABLE")
        print("%" * 60)
        print(latex_str)
        print(f"\nSaved raw ablation CSV to: {csv_path}")
        print(f"Saved LaTeX table to     : {tex_path}")


if __name__ == "__main__":
    main()