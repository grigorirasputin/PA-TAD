"""
GETAD.py
Implemented based on: Graph Enhanced Trajectory Anomaly Detection by Mbuya et al. (SIGSPATIAL '25)
https://arxiv.org/abs/2509.18386

Autoregressive Transformer-Decoder baseline for Trajectory Anomaly Detection (TAD).
Road network topology is captured via GAT embeddings and Graph Positional Embeddings (GPE).
Trajectory anomaly scores are computed using Negative Log-Likelihood (NLL) / CW-NLL.
Aligned with standardized benchmark output format, Tier 3 global benchmark, and pure-spatial evaluation.
"""

import argparse
import gc
import math
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_helpers import (
    ExecutionTracker,
    evaluate_anomaly_scores,
    evaluate_spatial_benchmark,
    evaluate_global_benchmark,
)
from GETAD_core.dataset import GETADCollateFn, TrajectoryDataset, build_graph_and_mapping
from GETAD_core.model import GETAD, PAD_TOKEN, link_prediction_loss
from GETAD_core.tools import EarlyStopping, setup_logger

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_READY_DIR = BASE_DIR / "data" / "06_model_ready"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / "getad"
RESULTS_DIR = BASE_DIR / "results" / "anomaly_detection"


def extract_spatial_subtype(series):
    for val in series:
        if pd.notna(val) and str(val).strip() and str(val).strip().lower() != "none":
            return str(val).strip()
    return "None"


class GETADRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
        self.region = args.region.lower()
        self.tag = args.tag

        # Configure logger
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = OUTPUT_DIR / "logs" / "getad"
        self.logger = setup_logger(args, current_time=curr_time, log_dir=str(log_dir))

        # Default d_max per region if not explicitly specified
        if args.d_max is not None:
            self.d_max = args.d_max
        else:
            self.d_max = {"aalborg": 30, "zealand": 15, "denmark": 10, "porto": 20}.get(self.region, 15)

        self.model_config = {
            "node_in_dim": 32,
            "node_emb_dim": 64,
            "edge_dim": 1,
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 8,
            "d_max": self.d_max,
        }

        self.data_dir = MODEL_READY_DIR / self.tag
        self.g_path = self.data_dir / f"model_ready_map_df_{self.region}.parquet"
        self.t_path = self.data_dir / f"model_ready_trajectories_{self.region}.parquet"

        if not self.g_path.exists() or not self.t_path.exists():
            raise FileNotFoundError(f"Missing model-ready dataset files in {self.data_dir} for region {self.region}")

        self.edge_index = None
        self.edge_attr = None
        self.num_nodes = None
        self.node2idx = None
        self.real_features = None
        self.global_D = None
        self.model = None

    def load_dataset(self, split):
        return TrajectoryDataset(self.t_path, self.node2idx, split=split, logger=self.logger)

    def _compute_batch_loss(self, batch, criterion_ce, compute_link=True):
        tokens, gpe_matrix, mask, lengths, _, _, _, _ = [
            b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch
        ]

        valid_mask = mask.bool()
        logits, node_embs = self.model(tokens, gpe_matrix, self.edge_index, valid_mask, edge_attr=self.edge_attr)

        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_targets = tokens[:, 1:].contiguous()
        shifted_logits = torch.clamp(shifted_logits, min=-30.0, max=30.0)

        loss_ce = criterion_ce(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_targets.view(-1),
        )

        loss_link = torch.tensor(0.0, device=self.device)
        if compute_link and self.args.lambda_link > 0.0:
            try:
                # Retain gradient flow into GAT layers per Section 4.1
                raw_loss_link = link_prediction_loss(self.model.link_predictor, node_embs, self.edge_index)
                loss_link = torch.clamp(raw_loss_link, min=0.0, max=10.0)
            except Exception:
                loss_link = torch.tensor(0.0, device=self.device)

        if not torch.isfinite(loss_ce):
            loss_ce = torch.tensor(0.0, device=self.device, requires_grad=True)
        if not torch.isfinite(loss_link):
            loss_link = torch.tensor(0.0, device=self.device)

        total_loss = (self.args.lambda_ce * loss_ce) + (self.args.lambda_link * loss_link)
        return total_loss, loss_ce.item(), loss_link.item()

    def train_epoch(self, train_loader, optimizer, criterion_ce):
        self.model.train()
        total_losses, ce_losses, link_losses = [], [], []

        for batch in tqdm(train_loader, desc=f"Training [{self.region.upper()}]", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss, ce_val, link_val = self._compute_batch_loss(batch, criterion_ce, compute_link=True)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            valid_gradients = True
            for param in self.model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    valid_gradients = False
                    break

            if valid_gradients:
                optimizer.step()
                total_losses.append(loss.item())
                ce_losses.append(ce_val)
                link_losses.append(link_val)

        return (
            float(np.mean(total_losses)) if total_losses else float("nan"),
            float(np.mean(ce_losses)) if ce_losses else float("nan"),
            float(np.mean(link_losses)) if link_losses else float("nan"),
        )

    def validate_epoch(self, val_loader, criterion_ce):
        self.model.eval()
        ce_losses = []
        total_losses = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                loss, ce_val, _ = self._compute_batch_loss(batch, criterion_ce, compute_link=False)
                if math.isfinite(ce_val):
                    ce_losses.append(ce_val)
                    total_losses.append(loss.item())

        mean_ce = float(np.mean(ce_losses)) if ce_losses else float("nan")
        mean_tot = float(np.mean(total_losses)) if total_losses else float("nan")
        return mean_ce, mean_tot

    def evaluate_anomaly_scores(self, loader):
        """
        Computes negative log-likelihood (NLL / CW-NLL) per trajectory.
        Returns both scores and the exact ordered trip IDs from the batches.
        """
        self.model.eval()
        criterion_nll = nn.CrossEntropyLoss(reduction="none", ignore_index=PAD_TOKEN)
        max_entropy = math.log(max(self.model.vocab_size, 2))

        scores_list = []
        ordered_trip_ids = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Scoring Trajectories", leave=False):
                tokens, gpe_matrix, mask, lengths, _, _, _, batch_tids = [
                    b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch
                ]

                valid_mask = mask.bool()
                logits, _ = self.model(tokens, gpe_matrix, self.edge_index, valid_mask, edge_attr=self.edge_attr)

                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_targets = tokens[:, 1:].contiguous()
                step_valid_mask = valid_mask[:, 1:].float()

                shifted_logits = torch.clamp(shifted_logits, min=-30.0, max=30.0)

                nll = criterion_nll(
                    shifted_logits.view(-1, self.model.vocab_size),
                    shifted_targets.view(-1),
                ).view(tokens.size(0), -1)

                if self.args.use_cwnll:
                    probs = F.softmax(shifted_logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                    confidence = torch.clamp(1.0 - (entropy / max_entropy), min=0.01, max=1.0)
                    step_scores = torch.nan_to_num(nll * confidence * step_valid_mask, nan=0.0)
                else:
                    step_scores = torch.nan_to_num(nll * step_valid_mask, nan=0.0)

                valid_counts = step_valid_mask.sum(dim=1).clamp(min=1.0)
                if self.args.scoring_mode == "mean":
                    batch_scores = step_scores.sum(dim=1) / valid_counts
                elif self.args.scoring_mode == "sum":
                    batch_scores = step_scores.sum(dim=1)
                elif self.args.scoring_mode == "max":
                    masked_for_max = step_scores.clone()
                    masked_for_max[step_valid_mask == 0] = -1e9
                    batch_scores = masked_for_max.max(dim=1)[0].clamp(min=0.0)
                else:
                    batch_scores = step_scores.sum(dim=1) / valid_counts

                scores_list.extend(batch_scores.cpu().numpy().tolist())
                ordered_trip_ids.extend([str(t) for t in batch_tids])

        return np.array(scores_list, dtype=np.float32), ordered_trip_ids

    def load_best_checkpoint(self, checkpoint_path):
        if checkpoint_path and Path(checkpoint_path).exists():
            self.logger.info(f"Loading best checkpoint from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self):
        self.logger.info(f"\n{'='*75}\nSTARTING GETAD PIPELINE - Region: {self.region.upper()} (Tag: {self.tag})\n{'='*75}")
        tracker = ExecutionTracker("GETAD", self.region, OUTPUT_DIR)

        # ----------------------------------------------------
        # 1. Data Loading & Graph Construction
        # ----------------------------------------------------
        try:
            with tracker.track("data_loading"):
                (self.edge_index, self.edge_attr, self.num_nodes, self.node2idx,
                 self.real_features, self.global_D) = build_graph_and_mapping(
                    self.g_path, self.d_max, t_path=self.t_path, logger=self.logger
                )
        except MemoryError:
            self.logger.error(f"FATAL: Insufficient memory for GPE distance matrix in region {self.region}. Aborting cleanly.")
            tracker.metrics["status"] = "OOM"
            tracker.save()
            return None

        self.edge_index = self.edge_index.to(self.device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(self.device)
        self.real_features = self.real_features.to(self.device)

        collate_fn = GETADCollateFn(self.global_D, self.d_max)

        train_ds = self.load_dataset("train")
        val_ds = self.load_dataset("val")
        test_ds = self.load_dataset("test")

        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # Build ground truth metadata lookup dictionary
        traj_df = pd.read_parquet(self.t_path)
        test_df = traj_df[traj_df["split"] == "test"]
        test_grouped = test_df.groupby("trip_id", sort=False)
        test_meta_lookup = {}
        for tid, grp in test_grouped:
            a_cls = grp["anomaly_class"].iloc[0]
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

        # ----------------------------------------------------
        # 2. Model Initialization & Setup
        # ----------------------------------------------------
        self.model = GETAD(self.num_nodes, self.real_features, self.model_config).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion_ce = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

        early_stopping = EarlyStopping(
            logger=self.logger,
            patience=self.args.patience,
            delta=self.args.early_stopping_delta,
            verbose=True,
        )

        ckpt_filename = f"getad_{self.region}_{self.tag}_best.pt"

        # ----------------------------------------------------
        # 3. Training Loop with Early Stopping
        # ----------------------------------------------------
        if self.args.train_epochs > 0:
            with tracker.track("training"):
                self.logger.info(f"\nBeginning training for up to {self.args.train_epochs} epochs...")
                for ep in range(1, self.args.train_epochs + 1):
                    start_time = time.time()
                    train_loss, ce_loss, link_loss = self.train_epoch(train_loader, optimizer, criterion_ce)
                    val_ce_loss, val_tot_loss = self.validate_epoch(val_loader, criterion_ce)
                    epoch_time = time.time() - start_time

                    self.logger.info(
                        f"[{self.region.upper()}] Epoch {ep:02d}/{self.args.train_epochs:02d} ({epoch_time:.1f}s) - "
                        f"Train Total: {train_loss:.4f} (CE: {ce_loss:.4f}, Link: {link_loss:.4f}) | "
                        f"Val CE: {val_ce_loss:.4f}"
                    )

                    monitor_loss = val_ce_loss if self.args.monitor == "ce" else val_tot_loss
                    if not math.isnan(monitor_loss):
                        early_stopping(monitor_loss, self.model, CHECKPOINT_DIR, ckpt_filename)
                    else:
                        self.logger.warning(f"Epoch {ep} produced non-finite validation loss. Skipping checkpoint saving.")

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    if early_stopping.early_stop:
                        self.logger.info(f"Early stopping triggered at epoch {ep}!")
                        break

            if early_stopping.best_checkpoint_path:
                self.load_best_checkpoint(early_stopping.best_checkpoint_path)

        # ----------------------------------------------------
        # 4. Inference & Trajectory Anomaly Scoring
        # ----------------------------------------------------
        self.logger.info("\nRunning inference and trajectory anomaly scoring...")
        with tracker.track("inference"):
            scores, test_trip_ids = self.evaluate_anomaly_scores(test_loader)

        tracker.metrics["avg_inf_per_trip"] = (
            tracker.metrics.get("inference", 0.0) / max(1, len(test_loader.dataset))
        )

        # Strict alignment guaranteed directly from batch iteration
        y_true = np.array([test_meta_lookup.get(tid, {}).get("label", 0) for tid in test_trip_ids], dtype=np.int32)
        y_spatial = np.array([test_meta_lookup.get(tid, {}).get("spatial_anomaly", 0) for tid in test_trip_ids], dtype=np.int32)
        y_temporal = np.array([test_meta_lookup.get(tid, {}).get("temporal_anomaly", 0) for tid in test_trip_ids], dtype=np.int32)
        anomaly_classes = np.array([test_meta_lookup.get(tid, {}).get("anomaly_class", "normal") for tid in test_trip_ids], dtype=object)
        spatial_subtypes = np.array([test_meta_lookup.get(tid, {}).get("spatial_subtype", "None") for tid in test_trip_ids], dtype=object)

        # ----------------------------------------------------
        # 5. Metric Evaluation & Output Artifacts
        # ----------------------------------------------------
        # Tier 3 Global Benchmark Evaluation
        global_results = evaluate_global_benchmark(
            labels=anomaly_classes,
            y_score=scores,
            method_name="GETAD",
            region=self.region,
            normal_label="normal",
            print_results=True,
        )
        tracker.metrics.update(global_results)

        if len(np.unique(y_spatial)) > 1:
            print("\n--- SPATIAL ANOMALY DETECTION (GETAD) ---")
            spatial_results = evaluate_anomaly_scores(
                y_true=y_spatial,
                y_score=scores,
                method_name="GETAD (Spatial)",
                region=self.region,
                print_results=True,
            )
            tracker.metrics.update({f"spatial_{k}": v for k, v in spatial_results.items()})

        # Save scores to outputs/anomaly_scores/getad_{region}_{tag}_scores.npz with standardized schema
        out_scores_dir = OUTPUT_DIR / "anomaly_scores"
        out_scores_dir.mkdir(parents=True, exist_ok=True)
        scores_file = out_scores_dir / f"getad_{self.region}_{self.tag}_scores.npz"
        np.savez_compressed(
            scores_file,
            trip_id=np.array(test_trip_ids, dtype=str),
            scores=scores,
            scores_spatial=scores,
            scores_temporal=np.zeros_like(scores),
            y_true=y_true,
            y_spatial=y_spatial,
            y_temporal=y_temporal,
            anomaly_class=anomaly_classes,
            spatial_subtype=spatial_subtypes,
        )
        self.logger.info(f"Saved inference anomaly scores to: {scores_file}")

        # Table 1: Isolated Pure Spatial Benchmark Evaluation
        spatial_bench = evaluate_spatial_benchmark(
            labels=spatial_subtypes,
            anomaly_class=anomaly_classes,
            y_score=scores,
            method_name="GETAD",
            region=self.region,
            print_table=True,
        )
        if isinstance(spatial_bench, dict):
            for k in ("overall_f1", "overall_auprc", "overall_auroc"):
                if k in spatial_bench:
                    tracker.metrics[f"spatial_{k}"] = spatial_bench[k]

        tracker.save()
        return global_results


def parse_args():
    parser = argparse.ArgumentParser(description="GETAD Benchmark Baseline")
    parser.add_argument("--tag", type=str, required=True, help="dataset pair label, e.g. '2014', 'porto'")
    parser.add_argument("--region", type=str, default="aalborg", help="Region (aalborg, porto, zealand, denmark)")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and inference")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--train_epochs", type=int, default=100, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience epochs")
    parser.add_argument("--early_stopping_delta", type=float, default=0.001, help="Early stopping improvement delta")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--d_max", type=int, default=None, help="Maximum hop-distance cutoff for GPE")
    parser.add_argument(
        "--scoring_mode",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
        help="CW-NLL trajectory score reduction mode",
    )
    parser.add_argument(
        "--use_cwnll",
        action="store_true",
        default=True,
        help="Whether to weight negative log-likelihood by prediction confidence (defaults to True per paper)",
    )
    parser.add_argument(
        "--no_cwnll",
        dest="use_cwnll",
        action="store_false",
        help="Disable CW-NLL confidence weighting and use standard NLL",
    )
    parser.add_argument("--lambda_ce", type=float, default=1.0, help="Weight for sequence cross-entropy loss")
    parser.add_argument("--lambda_link", type=float, default=0.01, help="Weight for link prediction auxiliary loss")
    parser.add_argument(
        "--monitor",
        type=str,
        default="ce",
        choices=["ce", "total"],
        help="Validation metric to monitor for early stopping (default: ce)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    runner = GETADRunner(cli_args)
    runner.run()
