"""
ablation_pure_eval.py

Recomputes the ablation study (Table IX) and the precision-recall figure (Fig. 6) from saved
score archives, with the paper's evaluation definitions:
  spatial  : spatial-only trajectories vs normal trajectories (compound excluded)
  temporal : temporal-only trajectories vs normal trajectories
F1 is the best F1 over thresholds per seed; values are averaged over seeds. For the base
PA-TAD-Lite, archives from the ablation folder are used when present, otherwise the main
multi-seed run (pa_tad_lite_* or legacy ct_tad_beta_*).

Usage (from the repository root):
    python src/05_TAD/ablation_pure_eval.py
    python src/05_TAD/ablation_pure_eval.py --plot-region porto --seeds 42 43 44 45 46
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

BASE_DIR = Path(__file__).resolve().parents[2]
VARIANTS = OrderedDict([
    ("base", "PA-TAD-Lite"), ("no_gat", "w/o Topological GAT"), ("no_tau", "w/o $\\tau$"),
    ("no_latlon", "w/o lat,lon"), ("no_road_cat", "w/o Road Category"),
    ("no_kinematics", "w/o Kinematics"), ("uni_gru", "w/ Uni-GRU"),
])
LEGEND = {"base": "PA-TAD-Lite", "no_gat": "w/o Topo. GAT", "no_tau": "w/o τ", "no_latlon": "w/o lat,lon",
          "no_road_cat": "w/o Road Cat.", "no_kinematics": "w/o kinematics", "uni_gru": "w/ Uni-GRU"}
STYLE = {
    "base": ("#111111", "-", 2.2), "no_gat": ("#1f77b4", "--", 1.4), "no_tau": ("#ff7f0e", "-.", 1.4),
    "no_latlon": ("#2ca02c", ":", 1.6), "no_road_cat": ("#d62728", "--", 1.4),
    "no_kinematics": ("#9467bd", "-.", 1.5), "uni_gru": ("#8c564b", ":", 1.6),
}
TAGS = {"porto": "porto", "aalborg": "2016_itsp", "copenhagen": "2016_itsp"}


def best_f1(y, s):
    p, r, _ = precision_recall_curve(y, s)
    d = p + r
    return float(np.max(np.divide(2 * p * r, d, out=np.zeros_like(d), where=d != 0)))


def archive_path(scores_dir, variant, region, tag, seed):
    abl = scores_dir / "ablation" / f"ablation_{variant}_{region}_{tag}_seed{seed}_scores_test.npz"
    if abl.exists() or variant != "base":
        return abl
    for prefix in ("pa_tad_lite", "ct_tad_beta"):
        main = scores_dir / f"{prefix}_{region}_{tag}_seed{seed}_scores_test.npz"
        if main.exists():
            return main
    return abl


def pure_arrays(path):
    d = np.load(path, allow_pickle=True)
    cls = np.array([str(c).strip().lower() for c in d["anomaly_class"]])
    s_mask = np.isin(cls, ["normal", "spatial"])
    t_mask = np.isin(cls, ["normal", "temporal"])
    return ((cls[s_mask] == "spatial").astype(int), d["scores_spatial"][s_mask].astype(float),
            (cls[t_mask] == "temporal").astype(int), d["scores_temporal"][t_mask].astype(float))


def main():
    ap = argparse.ArgumentParser(description="Ablation (Table IX) and PR figure (Fig. 6) with pure definitions")
    ap.add_argument("--scores-dir", type=Path, default=BASE_DIR / "outputs" / "anomaly_scores")
    ap.add_argument("--regions", nargs="+", default=["porto", "aalborg", "copenhagen"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--plot-region", default="porto")
    ap.add_argument("--out-dir", type=Path, default=BASE_DIR / "results" / "anomaly_detection")
    args = ap.parse_args()

    rows, curves = [], {}
    for region in args.regions:
        tag = TAGS.get(region, "2016_itsp")
        for v in VARIANTS:
            for seed in args.seeds:
                path = archive_path(args.scores_dir, v, region, tag, seed)
                if not path.exists():
                    print(f"[missing] {path.relative_to(args.scores_dir)}")
                    continue
                ys, ss, yt, st = pure_arrays(path)
                rows.append({"region": region, "variant": v, "seed": seed, "source": path.name,
                             "s_f1": best_f1(ys, ss) * 100, "t_f1": best_f1(yt, st) * 100,
                             "s_ap": average_precision_score(ys, ss) * 100, "t_ap": average_precision_score(yt, st) * 100})
                if region == args.plot_region:
                    curves.setdefault(v, []).append((ys, ss, yt, st))

    if not rows:
        print("No archives found.")
        return
    df = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "ablation_pure_runs.csv", index=False)
    summ = df.groupby(["region", "variant"], sort=False).agg(
        seeds=("seed", "count"), s_f1=("s_f1", "mean"), t_f1=("t_f1", "mean"),
        s_ap=("s_ap", "mean"), t_ap=("t_ap", "mean")).round(1).reset_index()
    summ.to_csv(args.out_dir / "ablation_pure_summary.csv", index=False)
    print("\n=== TABLE IX (pure definitions; F1 and AP in %, mean over seeds) ===")
    table = summ.pivot(index="variant", columns="region", values=["s_f1", "t_f1"])
    table = table.reindex(list(VARIANTS))
    table.index = [VARIANTS[v] for v in table.index]
    print(table.to_string())
    print("\nSeeds per cell:\n" + summ.pivot(index="variant", columns="region", values="seeds").reindex(list(VARIANTS)).to_string())

    plot_ap = summ[summ["region"] == args.plot_region].set_index("variant").reindex(list(VARIANTS))
    if not plot_ap.empty:
        print(f"\n=== AP for Fig. 6 ({args.plot_region}; mean over seeds, %) ===")
        print(plot_ap[["s_ap", "t_ap"]].rename(index=lambda v: LEGEND[v]).to_string())

    if curves:
        # Sized for one IEEE column (3.5 in) so fonts print at 5-7 pt without scaling
        plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.5})
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95), sharey=True)
        for v in VARIANTS:
            if v not in curves:
                continue
            runs = curves[v]
            color, ls, lw = STYLE[v]
            for ax, yi, si in [(axes[0], 0, 1), (axes[1], 2, 3)]:
                y_all = np.concatenate([r[yi] for r in runs]); s_all = np.concatenate([r[si] for r in runs])
                p, r_, _ = precision_recall_curve(y_all, s_all)
                ap_mean = np.mean([average_precision_score(r[yi], r[si]) for r in runs]) * 100
                ax.plot(r_, p, color=color, linestyle=ls, linewidth=lw * 0.6, label=f"{LEGEND[v]} ({ap_mean:.1f}%)")
        for ax, title in zip(axes, ["Spatial PR", "Temporal PR"]):
            ax.set_title(f"{args.plot_region.capitalize()} \u2014 {title}", fontsize=6.5, fontweight="bold", pad=2)
            ax.set_xlabel("Recall", fontsize=6, labelpad=1); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.tick_params(labelsize=5.5, width=0.5, length=2, pad=1); ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
            ax.legend(loc="lower left", fontsize=4.6, frameon=True, handlelength=2.2, borderpad=0.3,
                      labelspacing=0.2, framealpha=0.85)
        axes[0].set_ylabel("Precision", fontsize=6, labelpad=1)
        fig.tight_layout(pad=0.3, w_pad=0.6)
        stem = args.out_dir / "figures" / f"ablation_pr_curves_{args.plot_region}_pure"
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        print(f"\nSaved figure to {stem}.png / .pdf")
    print(f"Saved {args.out_dir / 'ablation_pure_summary.csv'} and ablation_pure_runs.csv")


if __name__ == "__main__":
    main()