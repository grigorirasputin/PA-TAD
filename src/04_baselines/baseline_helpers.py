import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class ExecutionTracker:
    def __init__(self, method_name, region, output_dir):
        self.method_name = method_name
        self.region = region
        self.output_dir = Path(output_dir) / "computational_time"
        self.metrics = {
            "method": method_name,
            "region": region,
            "timestamp": datetime.now().isoformat()
        }
        
        # Initialize NVML for power tracking (Nvidia GPUs only)
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Default to GPU 0
            except Exception:
                self.nvml_handle = None
        else:
            self.nvml_handle = None

    def track(self, label):
        """Returns a context manager to track time, peak VRAM, and power draw."""
        class Profiler:
            def __init__(self, outer, label):
                self.outer = outer
                self.label = label
                
            def __enter__(self):
                self.start_time = time.perf_counter()
                
                # Reset PyTorch peak memory counter for this specific block
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                # Snapshot initial power draw
                self.start_power = None
                if self.outer.nvml_handle:
                    try:
                        self.start_power = pynvml.nvmlDeviceGetPowerUsage(self.outer.nvml_handle) / 1000.0  # mW to W
                    except Exception:
                        pass
                
                return self
                
            def __exit__(self, *args):
                elapsed = time.perf_counter() - self.start_time
                self.outer.metrics[self.label] = elapsed
                
                # Capture Peak VRAM during this block
                if torch.cuda.is_available():
                    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
                    self.outer.metrics[f"{self.label}_peak_vram_mb"] = peak_vram
                
                # Capture Ending Power and calculate a rough average
                if self.outer.nvml_handle:
                    try:
                        end_power = pynvml.nvmlDeviceGetPowerUsage(self.outer.nvml_handle) / 1000.0
                        self.outer.metrics[f"{self.label}_end_power_W"] = end_power
                        if self.start_power is not None:
                            self.outer.metrics[f"{self.label}_avg_power_W"] = (self.start_power + end_power) / 2.0
                    except Exception:
                        pass

        return Profiler(self, label)

    def save(self):
        """Saves results named specifically by model and region."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.method_name.lower().replace(' ', '_')}_{self.region}_times.parquet"
        out_path = self.output_dir / filename
        
        # Saves the dictionary as a wide Parquet table automatically
        df = pd.DataFrame([self.metrics])
        df.to_parquet(out_path, index=False)


def evaluate_anomaly_scores(y_true, y_score, method_name="Method", region="Aalborg", print_results=True):
    """
    Compute full anomaly detection metrics and print nicely formatted results.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        return {
            'auroc': 0.0,
            'auprc': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'best_threshold': 0.0
        }

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    
    # Best F1 threshold
    prec, rec, th = precision_recall_curve(y_true, y_score)
    denom = prec + rec
    f1_scores = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom != 0)
    best_idx = np.argmax(f1_scores)
    best_th = th[best_idx] if best_idx < len(th) else float(y_score.mean())
    
    y_pred = (y_score >= best_th).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if print_results:
        print("\n" + "=" * 75)
        print(f"{method_name.upper()} RESULTS - {region.upper()}")
        print("=" * 75)
        print(f"AUROC      : {auroc:.4f}")
        print(f"AUPRC      : {auprc:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1 Score   : {f1:.4f}")
        print(f"Best Thresh: {best_th:.4f}")
        print("=" * 75)
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'best_threshold': best_th
    }


def compute_binary_auprc_f1(y_true_binary, scores):
    """Calculates AUPRC and best-threshold F1 score on 1D binary vectors."""
    y_true = np.asarray(y_true_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0

    auprc = float(average_precision_score(y_true, scores))
    prec, rec, th = precision_recall_curve(y_true, scores)
    denom = prec + rec
    f1_curve = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom != 0)
    best_f1 = float(np.max(f1_curve)) if len(f1_curve) > 0 else 0.0
    return best_f1, auprc


def evaluate_spatial_benchmark(
    labels,
    y_score=None,
    method_name="GM-VSAE",
    region="Copenhagen",
    subtypes=("detour", "sprw", "overshoot"),
    normal_label="normal",
    print_table=True,
    anomaly_class=None,
    include_compound=False,
):
    """
    Evaluates spatial anomaly detection (Table V): spatial-only trajectories of each subtype,
    and all spatial-only trajectories jointly, against normal trajectories. Temporal and
    compound (spatio-temporal) trajectories are excluded; compound trajectories also carry a
    spatial subtype, so they are dropped via `anomaly_class` unless include_compound=True.
    Supports either direct array inputs (pass `anomaly_class` too) or a path to a saved .npz archive.
    """
    if isinstance(labels, (str, Path)):
        npz_path = Path(labels)
        if not npz_path.exists():
            raise FileNotFoundError(f"Score archive file not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        if anomaly_class is None and "anomaly_class" in data:
            anomaly_class = data["anomaly_class"]

        if "scores_spatial" in data:
            y_score = data["scores_spatial"]
        else:
            y_score = data["scores"]

        if "spatial_subtype" in data:
            raw_labels = data["spatial_subtype"]
            if "anomaly_class" in data:
                a_cls = data["anomaly_class"]
                combined = []
                for st, ac in zip(raw_labels, a_cls):
                    st_clean = str(st).strip().lower()
                    if st_clean in ("none", "nan", "", "normal"):
                        combined.append(str(ac).strip().lower())
                    else:
                        combined.append(st_clean)
                raw_labels = np.array(combined)
        elif "anomaly_class" in data:
            raw_labels = data["anomaly_class"]
        else:
            raise KeyError(f"Neither 'spatial_subtype' nor 'anomaly_class' found in {npz_path}")
    else:
        raw_labels = np.asarray(labels)

    if y_score is None:
        raise ValueError("y_score must be provided when labels is not a .npz path.")

    # Normalize labels and scores to numpy arrays
    clean_labels = np.array([str(x).strip().lower() for x in raw_labels.ravel()])
    y_score = np.asarray(y_score, dtype=float).ravel()
    normal_label = str(normal_label).strip().lower()

    # Map aliases and variations of normal/none
    alias_map = {
        "over": "overshoot",
        "none": normal_label,
        "nan": normal_label,
        "": normal_label,
    }
    normalized_labels = np.array([alias_map.get(lbl, lbl) for lbl in clean_labels])
    if anomaly_class is not None and not include_compound:
        cls_arr = np.array([str(x).strip().lower() for x in np.asarray(anomaly_class).ravel()])
        normalized_labels = np.where(cls_arr == "spatio_temporal", "__excluded__", normalized_labels)
    canonical_subtypes = tuple(alias_map.get(str(st).strip().lower(), str(st).strip().lower()) for st in subtypes)

    results = {
        "method": method_name,
        "region": region,
        "table_formatted": {},
    }

    # 1. Per-Subtype Isolated Benchmark
    for st in canonical_subtypes:
        st_mask = (normalized_labels == normal_label) | (normalized_labels == st)
        sub_labels = normalized_labels[st_mask]
        sub_scores = y_score[st_mask]

        has_positives = np.any(sub_labels == st)
        has_negatives = np.any(sub_labels == normal_label)

        if has_positives and has_negatives:
            y_true_st = (sub_labels == st).astype(int)
            st_metrics = evaluate_anomaly_scores(
                y_true_st,
                sub_scores,
                method_name=f"{method_name}-{st}",
                region=region,
                print_results=False,
            )
            auroc = st_metrics["auroc"]
            f1 = st_metrics["f1"]
            auprc = st_metrics["auprc"]
            formatted_str = f"{auroc * 100:.1f} / {f1 * 100:.1f}"
        else:
            auroc = 0.0
            f1 = 0.0
            auprc = 0.0
            st_metrics = {
                "auroc": 0.0,
                "auprc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "best_threshold": 0.0,
            }
            formatted_str = "-- / --"

        results[f"{st}_auroc"] = auroc
        results[f"{st}_f1"] = f1
        results[f"{st}_auprc"] = auprc
        results[f"{st}_metrics"] = st_metrics
        results["table_formatted"][st] = formatted_str

    # 2. Overall Isolated Spatial Benchmark (exclude temporal & compound)
    spatial_mask = (normalized_labels == normal_label) | np.isin(normalized_labels, canonical_subtypes)
    overall_labels = normalized_labels[spatial_mask]
    overall_scores = y_score[spatial_mask]

    has_spatial_pos = np.any(np.isin(overall_labels, canonical_subtypes))
    has_spatial_neg = np.any(overall_labels == normal_label)

    if has_spatial_pos and has_spatial_neg:
        y_true_overall = (overall_labels != normal_label).astype(int)
        overall_metrics = evaluate_anomaly_scores(
            y_true_overall,
            overall_scores,
            method_name=f"{method_name}-overall_spatial",
            region=region,
            print_results=False,
        )
        overall_auroc = overall_metrics["auroc"]
        overall_f1 = overall_metrics["f1"]
        overall_auprc = overall_metrics["auprc"]
        overall_formatted = f"{overall_auroc * 100:.1f} / {overall_f1 * 100:.1f}"
    else:
        overall_auroc = 0.0
        overall_f1 = 0.0
        overall_auprc = 0.0
        overall_metrics = {
            "auroc": 0.0,
            "auprc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "best_threshold": 0.0,
        }
        overall_formatted = "-- / --"

    results["overall_auroc"] = overall_auroc
    results["overall_f1"] = overall_f1
    results["overall_auprc"] = overall_auprc
    results["overall_metrics"] = overall_metrics
    results["table_formatted"]["overall"] = overall_formatted

    if print_table:
        header_cols = [st.capitalize() for st in canonical_subtypes] + ["Overall Spatial"]
        val_cols = [results["table_formatted"].get(st, "-- / --") for st in canonical_subtypes] + [results["table_formatted"]["overall"]]

        col_widths = [max(len(h), len(v), 16) for h, v in zip(header_cols, val_cols)]
        model_col_w = max(len("Model"), len(method_name), 12)
        region_col_w = max(len("Region"), len(region), 12)

        total_w = model_col_w + region_col_w + sum(col_widths) + (3 * (len(col_widths) + 2) - 1)

        print("\n" + "=" * total_w)
        print(f"TABLE 1: ISOLATED PURE SPATIAL BENCHMARK - {method_name.upper()} ({region.upper()})")
        print("=" * total_w)

        header_str = (
            f"{'Model':<{model_col_w}} | "
            f"{'Region':<{region_col_w}} | "
            + " | ".join(f"{h:^{w}}" for h, w in zip(header_cols, col_widths))
        )
        print(header_str)
        print("-" * len(header_str))

        row_str = (
            f"{method_name:<{model_col_w}} | "
            f"{region.capitalize():<{region_col_w}} | "
            + " | ".join(f"{v:^{w}}" for v, w in zip(val_cols, col_widths))
        )
        print(row_str)
        print("=" * total_w + "\n")

    return results


def evaluate_global_benchmark(
    labels,
    y_score=None,
    method_name="Method",
    region="Region",
    normal_label="normal",
    print_results=True,
):
    """
    Evaluates global detection performance (Tier 3), where any anomaly
    (spatial, temporal, or compound) is treated as positive (1) and normal
    trajectories are negative (0).
    Supports either direct array inputs or passing a path to a saved .npz archive.
    """
    if isinstance(labels, (str, Path)):
        npz_path = Path(labels)
        if not npz_path.exists():
            raise FileNotFoundError(f"Score archive file not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)

        if "scores" in data:
            y_score = data["scores"]
        else:
            raise KeyError(f"'scores' array not found in {npz_path}")

        if "anomaly_class" in data:
            raw_labels = data["anomaly_class"]
        elif "y_true" in data:
            return evaluate_anomaly_scores(
                data["y_true"],
                y_score,
                method_name=f"{method_name} (Tier 3 Global)",
                region=region,
                print_results=print_results,
            )
        else:
            raise KeyError(f"Neither 'anomaly_class' nor 'y_true' found in {npz_path}")
    else:
        raw_labels = np.asarray(labels)

    if y_score is None:
        raise ValueError("y_score must be provided when labels is not a .npz path.")

    clean_labels = np.array([str(x).strip().lower() for x in raw_labels.ravel()])
    normal_label = str(normal_label).strip().lower()
    y_score = np.asarray(y_score, dtype=float).ravel()

    y_true = (clean_labels != normal_label).astype(int)
    return evaluate_anomaly_scores(
        y_true,
        y_score,
        method_name=f"{method_name} (Tier 3 Global)",
        region=region,
        print_results=print_results,
    )