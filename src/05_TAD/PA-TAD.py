"""
PA-TAD.py

Prior-Anchored Trajectory Anomaly Detection (PA-TAD), the full model in the paper.

Per region, the runner:
- trains (or loads) a Bayesian GAT travel-time prior on the normal traversals of the
  synthetic training split and freezes it;
- encodes the road network with the gated ego-MLP + GAT encoder shared with PA-TAD-Lite;
- feeds each step's node embedding, observed log-duration, prior residual, predictive
  log-variance, z-score, and harmonic time features into a 2-layer Bi-GRU;
- scores every edge with disentangled spatial and temporal heads (focal loss with
  homoscedastic multi-task weighting);
- writes test-split trajectory scores to
  outputs/anomaly_scores/pa_tad_{region}_{tag}_scores_test.npz.

Usage:
    python PA-TAD.py --tag porto
    python PA-TAD.py --tag 2016_itsp --regions aalborg copenhagen denmark
For the multi-seed results reported in the paper, use run_multiseed.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GATConv

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_READY_DIR = BASE_DIR / "data" / "06_model_ready"
SYNTH_DIR = BASE_DIR / "data" / "05_synthetic"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / "pa_tad"
RESULTS_DIR = BASE_DIR / "results" / "anomaly_detection"

# Add baseline helpers and PA-TAD core modules
sys.path.append(str(BASE_DIR / "src" / "04_baselines"))
sys.path.append(str(Path(__file__).resolve().parent))

from baseline_helpers import (
    ExecutionTracker,
    evaluate_anomaly_scores,
    evaluate_global_benchmark,
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
from PA_TAD_core.prior_gat import train_or_load_gat_prior
from PA_TAD_core.tools import set_all_seeds, setup_logger


# ==========================================
# 1. PA-TAD MODEL ARCHITECTURE
# ==========================================
class PA_TAD(nn.Module):
    """
    Prior-Anchored Trajectory Anomaly Detection (PA-TAD).
    Shares the exact backbone, disentangled adapters, and heads of PA-TAD-Lite,
    while anchoring temporal transitions with learned Bayesian GAT priors
    (residual, standardized z-score, and predictive log-variance).
    """
    def __init__(self, num_nodes, real_features, cfg, gat_prior, log_base_time, edge_index=None):
        super().__init__()
        self.real_feats = nn.Parameter(real_features, requires_grad=False)
        self.log_base_time = nn.Parameter(log_base_time, requires_grad=False)

        # Frozen GAT Prior Oracle
        self.gat_prior = gat_prior
        for p in self.gat_prior.parameters():
            p.requires_grad = False
        self.gat_prior.eval()

        # Precompute static node embeddings from the frozen GAT Prior once
        with torch.no_grad():
            if edge_index is not None:
                prior_node_embs = self.gat_prior.encode(self.real_feats, edge_index).detach()
            else:
                prior_node_embs = torch.empty(0, device=real_features.device)
        self.register_buffer("prior_node_embs", prior_node_embs, persistent=False)

        # Spatial Transition Graph Encoder (identical to PA-TAD-Lite)
        in_dim = real_features.shape[1]
        emb_dim = cfg["node_emb_dim"]
        self.ego_mlp1 = nn.Linear(in_dim, emb_dim * 2)
        self.ego_mlp2 = nn.Linear(emb_dim * 2, emb_dim)
        self.gat1 = GATConv(in_dim, emb_dim, heads=2, concat=True)
        self.gat2 = GATConv(emb_dim * 2, emb_dim, heads=1, concat=False)
        self.topo_gate = nn.Parameter(torch.full((1, emb_dim), 0.05))

        #self.gat1 = GATConv(real_features.shape[1], cfg["node_emb_dim"], heads=2, concat=True)
        #self.gat2 = GATConv(cfg["node_emb_dim"] * 2, cfg["node_emb_dim"], heads=1, concat=False)

        # Sequence Backbone:
        # node_emb (64) + log_secs (1) + prior_residual (1) + prior_log_var (1) + prior_z (1) + time_dim (5)
        gru_in_dim = cfg["node_emb_dim"] + 4 + cfg["time_dim"]
        self.gru = nn.GRU(
            gru_in_dim,
            cfg["hidden_dim"],
            cfg["lstm_layers"],
            batch_first=True,
            dropout=0.2 if cfg["lstm_layers"] > 1 else 0.0,
            bidirectional=True,
        )

        gru_out_dim = cfg["hidden_dim"] * 2
        adapter_dim = cfg["adapter_dim"]

        # Disentangled Projection Adapters (identical to PA-TAD-Lite)
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

        # Classification Heads (identical to PA-TAD-Lite)
        self.s_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(adapter_dim, 1),
        )
        self.t_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(adapter_dim, 1),
        )
    

    def _query_prior(self, nodes, time_feats, edge_index):
        with torch.no_grad():
            b, l = nodes.shape
            flat_nodes = nodes.reshape(-1)
            flat_time = time_feats.reshape(-1, time_feats.shape[-1])

            if self.prior_node_embs.numel() == 0 or self.prior_node_embs.shape[0] != self.real_feats.shape[0]:
                self.prior_node_embs = self.gat_prior.encode(self.real_feats, edge_index).detach()

            if hasattr(self.gat_prior, "predict"):
                mu, log_var = self.gat_prior.predict(
                    self.prior_node_embs, flat_nodes, flat_time, self.log_base_time
                )
            else:
                mu, log_var = self.gat_prior(
                    self.real_feats, edge_index, flat_nodes, flat_time, self.log_base_time
                )

            mu = mu.reshape(b, l, 1)
            log_var = log_var.reshape(b, l, 1)
            return mu, log_var

    def forward(self, nodes, secs, time_feats, edge_index, lengths):
        # 1. Topological Graph Transition Embeddings
        ego_embs = self.ego_mlp2(F.elu(self.ego_mlp1(self.real_feats)))
        topo_embs = self.gat2(F.elu(self.gat1(self.real_feats, edge_index)), edge_index)
        n_embs = (ego_embs + self.topo_gate * topo_embs)[nodes] 

        # 2. Query Frozen Prior Oracle for residual, uncertainty, and z-score
        mu, log_var = self._query_prior(nodes, time_feats, edge_index)
        residual = secs - mu
        std = torch.exp(0.5 * log_var).clamp(min=1e-3)
        z_score = torch.clamp(residual / std, min=-10.0, max=10.0)

        # 3. Concatenate and process via Bi-GRU Sequence Backbone
        gru_input = torch.cat([n_embs, secs, residual, log_var, z_score, time_feats], dim=-1)
        packed, _ = self.gru(
            nn.utils.rnn.pack_padded_sequence(
                gru_input,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
        )
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed, batch_first=True, total_length=nodes.size(1)
        )

        # 4. Route through disentangled task adapters
        s_feat = self.spatial_adapter(out)
        t_feat = self.temporal_adapter(out)

        s_logits = self.s_head(s_feat).squeeze(-1)
        t_logits = self.t_head(t_feat).squeeze(-1)
        

        return s_logits, t_logits


# ==========================================
# 2. RUNNER IMPLEMENTATION
# ==========================================
class PATADRunner:
    def __init__(self, args, region):
        self.args = args
        self.region = region.lower()
        self.tag = args.tag
        self.device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

        log_dir = OUTPUT_DIR / "logs" / "pa_tad"
        self.logger = setup_logger(self.region, self.tag, log_dir=log_dir)

        self.model_config = {
            "node_emb_dim": 64,
            "time_dim": 6,
            "hidden_dim": 128,
            "adapter_dim": 64,
            "lstm_layers": 2,
        }

        self.data_dir = MODEL_READY_DIR / self.tag
        self.synth_dir = SYNTH_DIR / self.tag

        self.g_path = self._resolve_file([
            self.data_dir / f"model_ready_map_df_{self.region}.parquet",
            self.synth_dir / f"map_df_{self.region}.parquet",
            self.data_dir / "model_ready_map_df_copenhagen.parquet" if self.region == "copenhagen" else None,
            self.synth_dir / "map_df_copenhagen.parquet" if self.region == "copenhagen" else None,
            BASE_DIR / "data" / "03_processed" / f"map_directed_{self.tag}.parquet",
        ])

        self.t_path = self._resolve_file([
            self.data_dir / f"model_ready_trajectories_{self.region}.parquet",
            self.synth_dir / f"synthetic_trajectories_{self.region}.parquet",
        ])

        if self.g_path is None or self.t_path is None:
            raise FileNotFoundError(
                f"Missing map ({self.g_path}) or trajectories ({self.t_path}) for region {self.region}"
            )

        self.edge_index = None
        self.num_nodes = None
        self.node2idx = None
        self.real_features = None
        self.log_base_time = None
        self.gat_prior = None
        self.model = None

    @staticmethod
    def _resolve_file(candidates):
        for c in candidates:
            if c is not None and c.exists():
                return c
        return None

    def _compute_node_base_time(self):
        """
        Derives statutory kinematic free-flow travel time per node in log1p space
        to match TrajectoryDataset's log1p(seconds) encoding.
        """
        map_df = pd.read_parquet(self.g_path)
        length_col = "segment_length" if "segment_length" in map_df.columns else (
            "length" if "length" in map_df.columns else None
        )
        speed_col = "speed_limit" if "speed_limit" in map_df.columns else (
            "maxspeed" if "maxspeed" in map_df.columns else None
        )

        lengths = (
            pd.to_numeric(map_df[length_col], errors="coerce").fillna(50.0).values
            if length_col else np.full(len(map_df), 50.0)
        )
        speeds_kmh = (
            pd.to_numeric(map_df[speed_col], errors="coerce").fillna(50.0).values
            if speed_col else np.full(len(map_df), 50.0)
        )
        speeds_ms = np.clip(speeds_kmh / 3.6, 1.0, 130.0 / 3.6)
        base_seconds = np.clip(lengths / speeds_ms, 1.0, 3600.0)

        acc = np.zeros(self.num_nodes, dtype=np.float32)
        cnt = np.zeros(self.num_nodes, dtype=np.float32)

        node_u = map_df["u"].map(self.node2idx).values
        u_valid = ~pd.isna(node_u)
        np.add.at(acc, node_u[u_valid].astype(np.int64), base_seconds[u_valid])
        np.add.at(cnt, node_u[u_valid].astype(np.int64), 1.0)

        # Fallback to incoming edges for any sink nodes with no outgoing edges
        missing_mask = cnt == 0
        if np.any(missing_mask) and "v" in map_df.columns:
            node_v = map_df["v"].map(self.node2idx).values
            v_valid = ~pd.isna(node_v)
            v_idx = node_v[v_valid].astype(np.int64)
            v_sec = base_seconds[v_valid]
            acc_v = np.zeros(self.num_nodes, dtype=np.float32)
            cnt_v = np.zeros(self.num_nodes, dtype=np.float32)
            np.add.at(acc_v, v_idx, v_sec)
            np.add.at(cnt_v, v_idx, 1.0)
            fill_mask = missing_mask & (cnt_v > 0)
            acc[fill_mask] = acc_v[fill_mask]
            cnt[fill_mask] = cnt_v[fill_mask]

        avg_base_time = np.divide(acc, np.maximum(cnt, 1.0))
        median_base = float(np.median(base_seconds)) if len(base_seconds) > 0 else 5.0
        avg_base_time = np.where(cnt > 0, avg_base_time, median_base)
        avg_base_time = np.clip(avg_base_time, 1.0, 3600.0)

        return torch.tensor(np.log1p(avg_base_time), dtype=torch.float32, device=self.device)

    def run(self):
        self.logger.info(
            f"\n{'='*75}\nSTARTING PA-TAD PIPELINE - Region: {self.region.upper()} (Tag: {self.tag})\n{'='*75}"
        )
        self.logger.info(f"Map file: {self.g_path}")
        self.logger.info(f"Trajectory file: {self.t_path}")

        tracker = ExecutionTracker("PA-TAD", self.region, OUTPUT_DIR)

        # 1. Data Loading & Graph Construction
        with tracker.track("data_loading"):
            self.edge_index, self.num_nodes, self.node2idx, self.real_features = build_graph_and_mapping(self.g_path)

        self.edge_index = self.edge_index.to(self.device)
        self.real_features = self.real_features.to(self.device)
        self.log_base_time = self._compute_node_base_time()

        train_ds = TrajectoryDataset(self.t_path, self.node2idx, "train")
        val_ds = TrajectoryDataset(self.t_path, self.node2idx, "val")
        test_ds = TrajectoryDataset(self.t_path, self.node2idx, "test")

        self.logger.info(f"Dataset split counts: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        if len(train_ds) == 0:
            self.logger.warning(f"No training trajectories for region '{self.region}'. Skipping.")
            return None

        # 2. Zero-Leakage Prior Training / Loading
        prior_ckpt_path = CHECKPOINT_DIR / self.tag / f"gat_prior_{self.region}_{self.tag}.pt"
        # Reuse a prior cached before the PACT-TAD -> PA-TAD rename instead of retraining it
        legacy_prior = OUTPUT_DIR / "checkpoints" / "pact_tad" / self.tag / f"gat_prior_{self.region}_{self.tag}.pt"
        if not prior_ckpt_path.exists() and legacy_prior.exists() and not self.args.force_retrain_prior:
            prior_ckpt_path = legacy_prior
        with tracker.track("prior_training"):
            self.gat_prior = train_or_load_gat_prior(
                train_ds=train_ds,
                val_ds=val_ds,
                real_features=self.real_features,
                edge_index=self.edge_index,
                log_base_time=self.log_base_time,
                checkpoint_path=prior_ckpt_path,
                device=self.device,
                logger=self.logger,
                time_channels=self.model_config["time_dim"],
                epochs=self.args.prior_epochs,
                force_retrain=self.args.force_retrain_prior,
                tag=self.tag,
            )

        train_loader = DataLoader(
            train_ds, batch_size=self.args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=self.args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # Build ground-truth metadata lookup for test split
        traj_df = pd.read_parquet(self.t_path)
        if "split" in traj_df.columns:
            test_df = traj_df[traj_df["split"] == "test"]
        else:
            test_trip_set = set(test_ds.trips[i]["trip_id"] for i in range(len(test_ds)))
            test_df = traj_df[traj_df["trip_id"].isin(test_trip_set)]

        test_grouped = test_df.groupby("trip_id", sort=False)
        test_meta_lookup = {}
        for tid, grp in test_grouped:
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

        # 3. Model & Loss Wrapper Initialization
        self.model = PA_TAD(
            self.num_nodes,
            self.real_features,
            self.model_config,
            self.gat_prior,
            self.log_base_time,
            edge_index=self.edge_index,
        ).to(self.device)
        loss_wrapper = MultiTaskLossWrapper().to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad] + list(loss_wrapper.parameters())
        opt = torch.optim.Adam(trainable_params, lr=self.args.learning_rate)

        early_stopping = EarlyStopping(
            logger=self.logger,
            patience=self.args.patience,
            delta=self.args.early_stopping_delta,
            mode="max",
        )
        ckpt_filename = f"pa_tad_{self.region}_{self.tag}_best.pt"

        # 4. Training Loop with Early Stopping
        with tracker.track("training"):
            for ep in range(1, self.args.epochs + 1):
                self.model.train()
                self.model.gat_prior.eval()
                loss_wrapper.train()
                with tracker.track(f"epoch_{ep}_time"):
                    for b in train_loader:
                        nodes, secs, time_f, s_l, t_l, sub, mask, lens, _, _ = b
                        nodes = nodes.to(self.device)
                        secs = secs.to(self.device)
                        time_f = time_f.to(self.device)
                        s_l = s_l.to(self.device)
                        t_l = t_l.to(self.device)
                        sub = sub.to(self.device)
                        mask = mask.to(self.device)

                        opt.zero_grad(set_to_none=True)
                        
                        s_log, t_log = self.model(nodes, secs, time_f, self.edge_index, lens)
                        loss, _, _ = loss_wrapper(s_log, t_log, s_l, t_l, mask)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
                        opt.step()

                vm = evaluate_model(self.model, val_loader, self.edge_index, self.device)
                joint_f1 = (vm["s_f1_pure"] + vm["t_f1_pure"]) / 2.0

                ep_time = tracker.metrics.get(f"epoch_{ep}_time", 0.0)
                ep_vram = tracker.metrics.get(f"epoch_{ep}_time_peak_vram_mb", 0.0)
                self.logger.info(
                    f"[Ep {ep:02d}/{self.args.epochs:02d}] "
                    f"Time: {ep_time:4.1f}s | VRAM: {ep_vram:4.0f}MB | "
                    f"Val S-F1: {vm['s_f1_pure']:.3f} | Val T-F1: {vm['t_f1_pure']:.3f} | Joint-F1: {joint_f1:.3f}"
                )

                early_stopping(
                    joint_f1,
                    self.model,
                    CHECKPOINT_DIR / self.tag,
                    ckpt_filename,
                    extra_state={"s_th": vm["s_th"], "t_th": vm["t_th"]},
                )
                if early_stopping.early_stop:
                    self.logger.info(f"Early stopping triggered at epoch {ep}!")
                    break

        # 5. Load Best Checkpoint
        best_ckpt = torch.load(early_stopping.best_checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(best_ckpt["state"])
        s_threshold = best_ckpt.get("s_th", None)
        t_threshold = best_ckpt.get("t_th", None)

        # 6. Inference (using standardized PA_TAD_core.evaluate_model)
        with tracker.track("inference"):
            tm = evaluate_model(
                self.model, test_loader, self.edge_index, self.device,
                th_s=s_threshold, th_t=t_threshold,
            )

        tracker.metrics["avg_inf_per_trip"] = (
            tracker.metrics.get("inference", 0.0) / max(len(test_loader.dataset), 1)
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

        # 7. Benchmark Evaluation & Standardized Artifacts
        if len(np.unique(y_spatial)) > 1:
            spatial_results = evaluate_anomaly_scores(
                y_true=y_spatial,
                y_score=s_scores,
                method_name="PA-TAD (SPATIAL)",
                region=self.region,
                print_results=True,
            )
            tracker.metrics.update({f"spatial_{k}": v for k, v in spatial_results.items()})

        if len(np.unique(y_temporal)) > 1:
            temporal_results = evaluate_anomaly_scores(
                y_true=y_temporal,
                y_score=t_scores,
                method_name="PA-TAD (TEMPORAL)",
                region=self.region,
                print_results=True,
            )
            tracker.metrics.update({f"temporal_{k}": v for k, v in temporal_results.items()})

        # Save standardized scores to outputs/anomaly_scores/pa_tad_{region}_{tag}_scores_test.npz
        out_scores_dir = OUTPUT_DIR / "anomaly_scores"
        out_scores_dir.mkdir(parents=True, exist_ok=True)
        scores_file = out_scores_dir / f"pa_tad_{self.region}_{self.tag}_scores_test.npz"
        np.savez_compressed(
            scores_file,
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
        self.logger.info(f"Saved inference anomaly scores to: {scores_file.resolve()}")

        # Table 1: Isolated Pure Spatial Benchmark (passing scores_file prevents temporal contamination)
        spatial_bench = evaluate_spatial_benchmark(
            labels=scores_file,
            y_score=s_scores,
            method_name="PA-TAD",
            region=self.region,
            print_table=True,
        )
        if isinstance(spatial_bench, dict):
            for k in ("overall_f1", "overall_auprc", "overall_auroc"):
                if k in spatial_bench:
                    tracker.metrics[f"spatial_{k}"] = spatial_bench[k]

        # Tier 3 Global Benchmark Evaluation
        global_results = evaluate_global_benchmark(
            labels=anomaly_classes,
            y_score=global_scores,
            method_name="PA-TAD",
            region=self.region,
            normal_label="normal",
            print_results=True,
        )
        tracker.metrics.update(global_results)

        tracker.metrics.update({k: v for k, v in tm.items() if not isinstance(v, (np.ndarray, list))})
        tracker.save()

        inf_vram = tracker.metrics.get("inference_peak_vram_mb", 0.0)
        inf_power = tracker.metrics.get("inference_avg_power_W", 0.0)
        try:
            import pynvml
            has_nvml = True
        except ImportError:
            has_nvml = False

        self.logger.info("\n--- HARDWARE PERFORMANCE ---")
        self.logger.info(f"Peak Inference VRAM : {inf_vram:.2f} MB")
        self.logger.info(f"Average Power Draw  : {inf_power:.2f} W" if has_nvml else "Average Power Draw  : [NVML not installed]")

        return global_results


def parse_args():
    parser = argparse.ArgumentParser(description="PA-TAD Benchmark Runner")
    parser.add_argument("--tag", required=True, help="dataset pair label, e.g. '2014', '2016_itsp', or 'porto'")
    parser.add_argument("--regions", "--region", dest="regions", nargs="+", default=None, help="regions to run, e.g. 'porto' or 'aalborg copenhagen denmark'")
    parser.add_argument("--device", type=str, default="cuda", help="execution device (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=50, help="maximum sequence model training epochs")
    parser.add_argument("--prior-epochs", "--prior_epochs", dest="prior_epochs", type=int, default=200, help="maximum GAT prior pre-training epochs")
    parser.add_argument("--force-retrain-prior", "--force_retrain_prior", dest="force_retrain_prior", action="store_true", help="force retrain GAT prior even if cached")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--early_stopping_delta", type=float, default=0.001, help="early stopping improvement delta")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", "--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="random seed (run_multiseed.py sets its own seeds)")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if cli_args.regions is None:
        if "porto" in str(cli_args.tag).lower():
            active_regions = ["porto"]
        else:
            active_regions = ["aalborg", "copenhagen", "denmark"]
    else:
        active_regions = cli_args.regions

    for r in active_regions:
        try:
            set_all_seeds(cli_args.seed)
            runner = PATADRunner(cli_args, r)
            runner.run()
        except Exception as e:
            print(f"Error executing PA-TAD on region {r}: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()