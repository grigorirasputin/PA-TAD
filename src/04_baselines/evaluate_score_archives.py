"""
evaluate_score_archives.py

Recomputes trajectory-level detection metrics from the saved per-seed score archives in
outputs/anomaly_scores/*.npz, under explicit label definitions:

Spatial (Table V), per subtype (Detour, SPRW, Overshoot) and overall:
  with_compound : positives = spatial AND compound trajectories with that deviation;
                  negatives = normal trajectories (what baseline_helpers.evaluate_spatial_benchmark does)
  pure          : positives = spatial-only trajectories; negatives = normal trajectories

Temporal (Table VI):
  all  : positives = temporal AND compound trajectories; negatives = all other trajectories
  pure : positives = temporal-only trajectories; negatives = normal trajectories

Compound (Table VII):
  pure : positives = compound trajectories; negatives = normal trajectories
  all  : positives = compound trajectories; negatives = all other trajectories

Spatial metrics use `scores_spatial` if present, otherwise `scores` (as evaluate_spatial_benchmark).
Temporal metrics use `scores_temporal` when it is non-constant, otherwise `scores`.
Compound metrics use `scores` and, for dual-head archives, each head separately.
AUROC and AUPRC are threshold-free; F1 is the best F1 over all thresholds, as in
baseline_helpers.evaluate_anomaly_scores. Results are averaged over seeds.

Usage (from the repository root):
    python src/04_baselines/evaluate_score_archives.py
    python src/04_baselines/evaluate_score_archives.py --tasks spatial
    python src/04_baselines/evaluate_score_archives.py --methods fotraj --keep-unseeded
    python src/04_baselines/evaluate_score_archives.py --scores-dir outputs/anomaly_scores/ablation --tasks spatial
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[2]
NAME_RE = re.compile(
    r"^(?P<method>.+?)_(?P<region>porto|aalborg|copenhagen|denmark)_(?P<tag>.+?)"
    r"(?:_seed(?P<seed>\d+))?_scores(?:_test)?$"
)
DISPLAY = {"pa_tad": "PA-TAD", "pact_tad": "PA-TAD", "pa_tad_lite": "PA-TAD-Lite", "ct_tad_beta": "PA-TAD-Lite"}
SUBTYPES = ("detour", "sprw", "overshoot")
SUBTYPE_ALIAS = {"over": "overshoot", "loop": "detour"}
NONE_LABELS = {"none", "nan", "", "normal"}


def metrics(y_true, y_score):
    """AUROC, AUPRC, best-threshold F1 (all in %), and positive prevalence (%)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    prevalence = float(y_true.mean() * 100.0) if len(y_true) else np.nan
    if len(np.unique(y_true)) < 2:
        return {"auroc": np.nan, "auprc": np.nan, "f1": np.nan, "prevalence": prevalence}
    prec, rec, th = precision_recall_curve(y_true, y_score)
    denom = prec + rec
    f1_curve = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom != 0)
    best = int(np.argmax(f1_curve))
    best_th = th[best] if best < len(th) else float(y_score.mean())
    f1 = f1_score(y_true, (y_score >= best_th).astype(int), zero_division=0)
    return {
        "auroc": roc_auc_score(y_true, y_score) * 100.0,
        "auprc": average_precision_score(y_true, y_score) * 100.0,
        "f1": f1 * 100.0,
        "prevalence": prevalence,
    }


def is_informative(arr):
    return arr is not None and len(arr) > 0 and not np.allclose(arr, arr[0])


def spatial_rows(cls, subtype, spatial_score):
    rows = []
    is_normal = cls == "normal"
    has_dev = np.isin(subtype, SUBTYPES)
    definitions = {
        "with_compound": has_dev & np.isin(cls, ["spatial", "spatio_temporal"]),
        "pure": has_dev & (cls == "spatial"),
    }
    for definition, positives in definitions.items():
        for st in SUBTYPES + ("overall",):
            pos = positives if st == "overall" else positives & (subtype == st)
            mask = is_normal | pos
            rows.append(("spatial", definition, st, metrics(pos[mask], spatial_score[mask])))
    return rows


def evaluate_archive(path, tasks):
    data = np.load(path, allow_pickle=True)
    cls = np.array([str(c).strip().lower() for c in data["anomaly_class"]])
    scores = data["scores"].astype(float)
    s_scores = data["scores_spatial"].astype(float) if "scores_spatial" in data.files else None
    t_scores = data["scores_temporal"].astype(float) if "scores_temporal" in data.files else None
    rows = []

    if "spatial" in tasks and "spatial_subtype" in data.files:
        sub = np.array([SUBTYPE_ALIAS.get(str(s).strip().lower(), str(s).strip().lower()) for s in data["spatial_subtype"]])
        sub = np.where(np.isin(sub, list(NONE_LABELS)), "none", sub)
        rows += spatial_rows(cls, sub, s_scores if s_scores is not None else scores)

    if "temporal" in tasks:
        temporal_score = t_scores if is_informative(t_scores) else scores
        y_temporal = (data["y_temporal"].astype(int) if "y_temporal" in data.files
                      else np.isin(cls, ["temporal", "spatio_temporal"]).astype(int))
        pure_t = np.isin(cls, ["normal", "temporal"])
        rows.append(("temporal", "pure", "-", metrics(cls[pure_t] == "temporal", temporal_score[pure_t])))
        rows.append(("temporal", "all", "-", metrics(y_temporal, temporal_score)))

    if "compound" in tasks:
        dual_head = is_informative(t_scores) and is_informative(s_scores) and not np.allclose(s_scores, scores)
        variants = [("scores", scores)] + ([("spatial", s_scores), ("temporal", t_scores)] if dual_head else [])
        pure_c = np.isin(cls, ["normal", "spatio_temporal"])
        for name, sc in variants:
            rows.append(("compound", "pure", name, metrics(cls[pure_c] == "spatio_temporal", sc[pure_c])))
            rows.append(("compound", "all", name, metrics(cls == "spatio_temporal", sc)))
    return rows


def print_spatial_tables(summary):
    sp = summary[summary["task"] == "spatial"].copy()
    if sp.empty:
        return
    sp["cell"] = sp["auroc"].map("{:.1f}".format) + " / " + sp["f1"].map("{:.1f}".format)
    for definition in ("with_compound", "pure"):
        block = sp[sp["definition"] == definition]
        if block.empty:
            continue
        prev = block[block["variant"] == "overall"]["prevalence"].mean()
        print(f"\n=== SPATIAL (Table V layout, AUROC / F1) | {definition} | overall positive prevalence ~{prev:.1f}% ===")
        table = block.pivot_table(index=["region", "tag", "method"], columns="variant", values="cell", aggfunc="first")
        table = table.reindex(columns=[c for c in SUBTYPES + ("overall",) if c in table.columns])
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(table.to_string())


def main():
    parser = argparse.ArgumentParser(description="Recompute detection tables from saved score archives")
    parser.add_argument("--scores-dir", type=Path, default=BASE_DIR / "outputs" / "anomaly_scores")
    parser.add_argument("--regions", nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="only these archive prefixes, e.g. fotraj (default: all)")
    parser.add_argument("--keep-unseeded", action="store_true",
                        help="also count archives without a _seedNN suffix as a run (e.g. an early FoTraj run saved without its seed)")
    parser.add_argument("--tasks", nargs="+", default=["spatial", "temporal", "compound"],
                        choices=["spatial", "temporal", "compound"])
    parser.add_argument("--out", type=Path, default=BASE_DIR / "results" / "anomaly_detection" / "score_archive_check.csv")
    args = parser.parse_args()

    found = []
    for path in sorted(args.scores_dir.glob("*.npz")):
        m = NAME_RE.match(path.stem)
        if not m:
            print(f"[skip] unrecognised name: {path.name}")
            continue
        if args.regions and m["region"] not in args.regions:
            continue
        if args.methods and m["method"] not in args.methods:
            continue
        found.append((path, m["method"], m["region"], m["tag"], int(m["seed"]) if m["seed"] else None))

    # Drop unseeded "latest run" copies when seeded archives exist for the same method/region/tag
    seeded = {(meth, reg, tag) for _, meth, reg, tag, seed in found if seed is not None}
    if not args.keep_unseeded:
        found = [f for f in found if f[4] is not None or (f[1], f[2], f[3]) not in seeded]
    for f in found:
        print(f"[use] {f[0].name}")

    records = []
    for path, method, region, tag, seed in found:
        try:
            for task, definition, variant, res in evaluate_archive(path, args.tasks):
                records.append({"method": DISPLAY.get(method, method), "region": region, "tag": tag, "seed": seed, "source": path.name,
                                "task": task, "definition": definition, "variant": variant, **res})
        except KeyError as exc:
            print(f"[skip] {path.name}: missing array {exc}")

    if not records:
        print(f"No usable archives in {args.scores_dir}")
        return
    df = pd.DataFrame(records)
    summary = (df.groupby(["task", "definition", "region", "tag", "method", "variant"], sort=True, dropna=False)
                 .agg(seeds=("source", "count"), prevalence=("prevalence", "mean"),
                      auroc=("auroc", "mean"), auprc=("auprc", "mean"), f1=("f1", "mean"), f1_std=("f1", "std"))
                 .round(1).reset_index())

    print_spatial_tables(summary)
    with pd.option_context("display.max_rows", None, "display.width", 160):
        for (task, definition), block in summary[summary["task"] != "spatial"].groupby(["task", "definition"], sort=False):
            print(f"\n=== {task.upper()} | {definition} ===")
            print(block.drop(columns=["task", "definition"]).to_string(index=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    runs_out = args.out.with_name(args.out.stem + "_runs.csv")
    df.to_csv(runs_out, index=False)
    print(f"\nSaved summary to {args.out} and per-seed rows to {runs_out}")


if __name__ == "__main__":
    main()
