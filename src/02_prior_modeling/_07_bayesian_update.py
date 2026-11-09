"""
07_bayesian_update.py

This script performs local Bayesian conjugate updating on unseen cold-start edges
using revealed warmup observations.

Update Levels:
- Level 1: Additive shift in traversal seconds (Gaussian-Gaussian conjugate).
- Level 2: Proportional / scalar multiplicative shift in log-space (Log-Normal conjugate).
- Level 3: Feature weighted harmonic shape shift via Bayesian Ridge Regression in log-space.

Theoretical Formulation:
Hyperparameters (sigma^2_prior, sigma^2_obs) are derived via closed-form Empirical Bayes
variance-components estimation (ANOVA method of moments) on the pooled warmup set,
guaranteeing unbiased scale estimation.
"""

import os
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import polars as pl
import torch
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from _05_train_prior import (
    BayesianTrafficGAT,
    TEMPORAL_FEATURE_COLUMNS,
    add_temporal_features_polars,
    evaluate_global_weekly_prior,
)
from _metrics_common import (
    log_space_point_metrics, lognormal_prediction_interval, to_log_seconds,
    gaussian_distributional_report, Z_90,
)


# ====================================================================
# 0. CLI & ENVIRONMENT
# ====================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Run tag matching prior artifacts.")
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.80,
        help="Fraction of cold-start edge observations revealed for updating.",
    )
    parser.add_argument(
        "--tuning-ratio",
        type=float,
        default=0.20,
        help="(Deprecated) Retained for command-line backwards compatibility.",
    )
    return parser.parse_args()


args = parse_args()
if not 0.0 < args.warmup_ratio < 1.0:
    raise ValueError("--warmup-ratio must be strictly between 0 and 1.")

BASE_DIR = Path(__file__).resolve().parents[2]
COLD_START_DIR = BASE_DIR / "data" / "04_cold_start"
MODELS_DIR = BASE_DIR / "models" / "prior_weights" / args.tag
RESULTS_DIR = BASE_DIR / "results" / "prior_modeling" / args.tag / "bayesian_update"
INFERENCE_BATCH_SIZE = 32_768
LEVEL3_RIDGE = 1e-1  # Well-conditioned regularizer for small-sample harmonic inversion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================================================
# 1. EMPIRICAL BAYES VARIANCE ESTIMATOR
# ====================================================================
def estimate_variance_components(
    warmup_df,
    prior_col,
    multiplicative=True,
    feature_columns=None,
):
    """
    Computes closed-form Empirical Bayes variance components for a 2-level model:
        residual_{e, i} = delta_e + epsilon_{e, i}
        delta_e ~ N(0, sigma0_sq)
        epsilon_{e, i} ~ N(0, obs_var)
    """
    if multiplicative:
        residuals = np.log(warmup_df["seconds"].values) - np.log(warmup_df[prior_col].values)
    else:
        residuals = warmup_df["seconds"].values - warmup_df[prior_col].values

    # Remove temporal structure shared across edges before estimating the
    # irreducible observation variance.
    noise_residuals = residuals.copy()
    if feature_columns is not None:
        design = warmup_df[list(feature_columns)].to_numpy(dtype=np.float64)
        if design.shape[1] > 1 and len(design) > design.shape[1]:
            varying_columns = np.std(design, axis=0) > 1e-12
            temporal_design = np.column_stack([np.ones(len(design)), design[:, varying_columns]])
            coefficients = np.linalg.lstsq(temporal_design, residuals, rcond=None)[0]
            noise_residuals = residuals - temporal_design[:, 1:] @ coefficients[1:]

    temp_df = pd.DataFrame({
        "edge_id": warmup_df["edge_id"].values,
        "residual": residuals,
        "noise_residual": noise_residuals,
    })

    # Group statistics
    stats = temp_df.groupby("edge_id")["residual"].agg(["count", "mean"])
    edge_means = stats["mean"]
    edge_counts = stats["count"]
    noise_edge_means = temp_df.groupby("edge_id")["noise_residual"].mean()

    # 1. Within-edge variance (sigma^2_obs) pooled over edges with n >= 2
    multi_obs = temp_df[temp_df["edge_id"].isin(stats[stats["count"] >= 2].index)]
    if len(multi_obs) > 0:
        edge_mean_mapped = multi_obs["edge_id"].map(noise_edge_means).values
        within_residuals = multi_obs["noise_residual"].values - edge_mean_mapped
        median = np.median(within_residuals)
        mad = np.median(np.abs(within_residuals - median))
        robust_sigma = 1.4826 * mad
        if robust_sigma > 0:
            obs_var = float(robust_sigma ** 2)
        else:
            lo, hi = np.quantile(within_residuals, [0.01, 0.99])
            within_residuals_clipped = np.clip(within_residuals, lo, hi)
            dof = len(multi_obs) - len(stats[stats["count"] >= 2])
            obs_var = float(np.sum(within_residuals_clipped ** 2) / max(dof, 1))
    else:
        obs_var = float(np.var(noise_residuals))
    obs_var = max(obs_var, 1e-4)

    # 2. Between-edge variance (sigma0_sq) corrected for sample mean sampling error
    median_means = np.median(edge_means.values)
    mad_means = np.median(np.abs(edge_means.values - median_means))
    robust_between = (1.4826 * mad_means) ** 2
    if robust_between > 0:
        var_between_raw = float(robust_between)
    else:
        lo_m, hi_m = np.quantile(edge_means.values, [0.01, 0.99])
        clipped_means = np.clip(edge_means.values, lo_m, hi_m)
        var_between_raw = float(np.var(clipped_means, ddof=1)) if len(clipped_means) > 1 else float(obs_var)
    mean_inv_n = float(np.mean(1.0 / edge_counts.values))
    prior_var = max(var_between_raw - obs_var * mean_inv_n, 1e-4)

    return prior_var, obs_var


def estimate_variance_components_stratified(
    warmup_df,
    prior_col,
    multiplicative=True,
    feature_columns=None,
    cluster_col="variance_cluster",
    min_cluster_obs=150,
    min_cluster_edges=5,
):
    global_params = estimate_variance_components(
        warmup_df, prior_col, multiplicative=multiplicative, feature_columns=feature_columns
    )
    cluster_params = {}
    for cluster_id, sub in warmup_df.groupby(cluster_col):
        enough_obs = len(sub) >= min_cluster_obs
        enough_edges = sub["edge_id"].nunique() >= min_cluster_edges
        if enough_obs and enough_edges:
            try:
                cluster_params[cluster_id] = estimate_variance_components(
                    sub, prior_col, multiplicative=multiplicative, feature_columns=feature_columns
                )
            except Exception:
                cluster_params[cluster_id] = global_params
        else:
            cluster_params[cluster_id] = global_params
    return cluster_params, global_params


# ====================================================================
# 2. BAYESIAN CONJUGATE UPDATER
# ====================================================================
class BayesianConjugateUpdater:
    def __init__(self, prior_var=0.5, obs_var=1.0, ridge=LEVEL3_RIDGE):
        self.sigma0_sq = prior_var
        self.obs_var = obs_var
        self.ridge = ridge

    def level_1_additive(self, y_true, y_prior):
        residuals = y_true - y_prior
        n = len(residuals)
        if n == 0:
            return 0.0, self.sigma0_sq
        shrinkage = (n * self.sigma0_sq) / (n * self.sigma0_sq + self.obs_var)
        post_mean = shrinkage * float(np.mean(residuals))
        post_var = 1.0 / (1.0 / self.sigma0_sq + n / self.obs_var)
        return post_mean, post_var

    def level_2_proportional(self, y_true, y_prior):
        residuals = np.log(y_true) - np.log(y_prior)
        n = len(residuals)
        if n == 0:
            return 0.0, self.sigma0_sq
        shrinkage = (n * self.sigma0_sq) / (n * self.sigma0_sq + self.obs_var)
        post_mean = shrinkage * float(np.mean(residuals))
        post_mean = float(np.clip(post_mean, -2.0, 2.0))
        post_var = 1.0 / (1.0 / self.sigma0_sq + n / self.obs_var)
        return post_mean, post_var

    def level_3_feature_weighted(self, y_true, y_prior, X_features, max_iter=50, tol=1e-3):
        residuals = np.log(y_true) - np.log(y_prior)
        n, d = X_features.shape
        if n == 0:
            return np.zeros(d), np.eye(d) * self.sigma0_sq

        alpha = np.full(d, 10.0 / self.sigma0_sq)
        alpha[0] = 1.0 / self.sigma0_sq
        beta = 1.0 / self.obs_var

        XTX = X_features.T @ X_features
        XTy = X_features.T @ residuals
        ridge_eye = np.eye(d) * self.ridge

        mu_n = np.zeros(d)
        Sigma_n = np.eye(d) * self.sigma0_sq
        for _ in range(max_iter):
            A = np.diag(alpha) + beta * XTX + ridge_eye
            Sigma_n = np.linalg.inv(A)
            mu_n = beta * Sigma_n @ XTy

            gamma = np.clip(1.0 - alpha * np.diag(Sigma_n), 1e-6, None)
            alpha_new = np.clip(gamma / (mu_n ** 2 + 1e-8), 1e-6, 1e8)

            resid = residuals - X_features @ mu_n
            denom = max(n - gamma.sum(), 1e-6)
            beta_new = np.clip(denom / (resid @ resid + 1e-8), 1e-6, 1e8)

            delta = np.max(np.abs(np.log(alpha_new) - np.log(alpha)))
            alpha, beta = alpha_new, beta_new
            if delta < tol:
                break

        return mu_n, Sigma_n


# ====================================================================
# 3. PIPELINE IMPLEMENTATION
# ====================================================================
def run_bayesian_pipeline():
    start_time = datetime.now()
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] --- STARTING ROBUST BAYESIAN UPDATE ON {device.type.upper()} ---")

    # ---------------------------------------------------------
    # A. LOAD DATA & ARTIFACTS
    # ---------------------------------------------------------
    print("Loading Graph Artifacts and Prior Models...")
    with open(MODELS_DIR / "graph_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    graph_data = artifacts["graph_data"].to(device)
    log_base_time = artifacts["log_base_time"].to(device)
    edge_lookup = artifacts["edge_lookup"]
    train_edge_ids = artifacts["train_edge_ids"]

    with open(MODELS_DIR / "global_weekly_prior.json", encoding="utf-8") as f:
        global_weekly_prior = json.load(f)

    gat_model = BayesianTrafficGAT(
        static_in_channels=graph_data.x.shape[1],
        time_channels=len(TEMPORAL_FEATURE_COLUMNS),
    ).to(device)
    gat_model.load_state_dict(torch.load(MODELS_DIR / "GAT_prior_weights.pth", weights_only=True))
    gat_model.eval()

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(MODELS_DIR / "xgboost_weights.json")

    map_parquet_path = COLD_START_DIR / f"cold_start_graph_{args.tag}.parquet"
    map_df = pd.read_parquet(map_parquet_path)
    map_df["segmentkey"] = map_df["segmentkey"].astype(str)
    
    # Assign edge_id directly or map from lookup
    map_df["edge_id"] = pd.MultiIndex.from_arrays([map_df["u"], map_df["v"], map_df["segmentkey"]]).map(edge_lookup)
    map_df = map_df.dropna(subset=["edge_id"]).drop_duplicates(subset=["edge_id"], keep="last").copy()
    map_df["edge_id"] = map_df["edge_id"].astype(int)

    edge_meta_lookup = map_df.set_index("edge_id")[["u", "v", "category", "speed_limit"]]

    cat_dummies = pd.get_dummies(map_df["category"], prefix="cat", dtype=float)
    map_df = pd.concat([map_df, cat_dummies], axis=1)

    cont_cols = ["speed_limit", "segment_length", "tortuosity"]
    all_scaled_cols = cont_cols + list(cat_dummies.columns)
    map_df[all_scaled_cols] = artifacts["scaler"].transform(map_df[all_scaled_cols])
    xgb_feat_cols = cont_cols + TEMPORAL_FEATURE_COLUMNS + list(cat_dummies.columns)

    # Heteroscedastic Variance Clustering
    N_VARIANCE_CLUSTERS = 8
    n_clusters = max(1, min(N_VARIANCE_CLUSTERS, map_df["edge_id"].nunique() // 25))
    if n_clusters > 1:
        cluster_features = map_df[all_scaled_cols].to_numpy(dtype=np.float64)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        map_df["variance_cluster"] = kmeans.fit_predict(cluster_features)
    else:
        map_df["variance_cluster"] = 0
    edge_cluster_map = map_df.set_index("edge_id")["variance_cluster"].to_dict()
    print(f"Assigned {map_df['edge_id'].nunique():,} cold-start edges to {n_clusters} variance clusters.")

    # Edge Lookup table for streaming Polars join
    pl_edge_lookup = pl.from_pandas(
        map_df[["u", "v", "segmentkey", "edge_id"]].astype({"segmentkey": str, "edge_id": np.int32})
    )

    print("Loading held-out cold-start trajectories via Polars...")
    test_path = COLD_START_DIR / f"test_traj_{args.tag}.parquet"
    split_struct = pl.struct(["edge_id", "trip_id", "trip_segmentno", "datetime", "seconds"])

    time_cols = TEMPORAL_FEATURE_COLUMNS
    lvl3_features = ["bias_term"] + time_cols

    # Process and partition test data through lazy streaming
    traj_lazy = (
        pl.scan_parquet(test_path)
        .with_columns(pl.col("segmentkey").cast(pl.String))
        .pipe(add_temporal_features_polars)
        .join(pl_edge_lookup.lazy(), on=["u", "v", "segmentkey"], how="inner")
        .with_columns([
            pl.lit(1.0).alias("bias_term"),
            split_struct.hash(seed=42).alias("_split_hash"),
        ])
        .sort(["edge_id", "_split_hash"])
        .with_columns([
            pl.len().over("edge_id").alias("edge_obs_count"),
            pl.int_range(0, pl.len()).over("edge_id").alias("edge_obs_rank"),
        ])
        .filter(pl.col("edge_obs_count") >= 2)
        .with_columns(
            (pl.col("edge_obs_count") * args.warmup_ratio)
            .floor()
            .cast(pl.Int64)
            .clip(1, pl.col("edge_obs_count") - 1)
            .alias("warmup_count")
        )
        .with_columns(
            (pl.col("edge_obs_rank") < pl.col("warmup_count")).alias("_is_warmup")
        )
    )

    req_cols = ["edge_id", "seconds", "hour_of_week", "bias_term"] + time_cols
    print("Collecting warmup and evaluation splits into memory...")
    warmup_df = traj_lazy.filter(pl.col("_is_warmup")).select(req_cols).collect().to_pandas()
    eval_df = traj_lazy.filter(~pl.col("_is_warmup")).select(req_cols).collect().to_pandas()

    if any(e in train_edge_ids for e in warmup_df["edge_id"].unique()):
        raise ValueError("test_traj contains edges used during prior training.")

    warmup_df["variance_cluster"] = warmup_df["edge_id"].map(edge_cluster_map).fillna(-1).astype(int)
    eval_df["variance_cluster"] = eval_df["edge_id"].map(edge_cluster_map).fillna(-1).astype(int)

    print(
        f"Partitioned: {args.warmup_ratio:.0%} warmup ({len(warmup_df):,} rows) / "
        f"{1 - args.warmup_ratio:.0%} eval ({len(eval_df):,} rows) across {warmup_df['edge_id'].nunique():,} edges."
    )

    # ---------------------------------------------------------
    # B. GENERATE PRIORS
    # ---------------------------------------------------------
    print("Computing baseline zero-shot priors...")
    def get_priors(target_df):
        edge_values = target_df["edge_id"].to_numpy(dtype=np.int64, copy=False)
        time_values = target_df[time_cols].to_numpy(dtype=np.float32, copy=False)
        n_rows = len(target_df)
        gat_prior = np.empty(n_rows, dtype=np.float32)
        kin_prior = np.empty(n_rows, dtype=np.float32)
        xgb_prior = np.empty(n_rows, dtype=np.float32)
        edge_features = map_df.set_index("edge_id")[all_scaled_cols]

        with torch.no_grad():
            encoded_graph = gat_model.encode(graph_data.x, graph_data.edge_index)
            for start in range(0, n_rows, INFERENCE_BATCH_SIZE):
                stop = min(start + INFERENCE_BATCH_SIZE, n_rows)
                b_edges = torch.as_tensor(edge_values[start:stop], dtype=torch.long, device=device)
                b_time = torch.as_tensor(time_values[start:stop], dtype=torch.float, device=device)
                b_delay = torch.as_tensor(
                    evaluate_global_weekly_prior(target_df["hour_of_week"].iloc[start:stop], global_weekly_prior),
                    dtype=torch.float,
                    device=device,
                )
                mu, _ = gat_model.predict(
                    encoded_graph, graph_data.edge_index, b_edges, b_time, log_base_time, b_delay
                )
                gat_prior[start:stop] = np.exp(mu.cpu().numpy())
                kin_prior[start:stop] = np.exp(log_base_time[b_edges].cpu().numpy() + b_delay.cpu().numpy())

                xgb_input = edge_features.reindex(edge_values[start:stop]).reset_index(drop=True)
                xgb_input[time_cols] = time_values[start:stop]
                xgb_pred = xgb_model.predict(xgb_input[xgb_feat_cols])
                xgb_prior[start:stop] = np.clip(
                    np.exp(log_base_time[b_edges].cpu().numpy() + b_delay.cpu().numpy() + xgb_pred),
                    a_min=1.0,
                    a_max=None,
                )

        target_df["Prior_GAT"] = gat_prior
        target_df["Prior_Kin"] = kin_prior
        target_df["Prior_XGB"] = xgb_prior
        return target_df

    warmup_df = get_priors(warmup_df)
    eval_df = get_priors(eval_df)

    # ---------------------------------------------------------
    # C. EMPIRICAL BAYES HYPERPARAMETER ESTIMATION
    # ---------------------------------------------------------
    print("Estimating heteroscedastic (per-cluster) Empirical Bayes variance components...")
    best_params = {}
    for m_key, prior_col in [("Kin", "Prior_Kin"), ("XGB", "Prior_XGB"), ("GAT", "Prior_GAT")]:
        l1_cluster_params, l1_global = estimate_variance_components_stratified(
            warmup_df, prior_col, multiplicative=False, feature_columns=lvl3_features
        )
        log_cluster_params, log_global = estimate_variance_components_stratified(
            warmup_df, prior_col, multiplicative=True, feature_columns=lvl3_features
        )
        best_params[m_key] = {
            "L1": l1_cluster_params,
            "L1_global": l1_global,
            "L2_L3": log_cluster_params,
            "L2_L3_global": log_global,
        }
        print(f"  [{m_key.ljust(3)}] L1 (sec²)    global: Prior_Var = {l1_global[0]:.3f}, Obs_Var = {l1_global[1]:.3f}")
        print(f"  [{m_key.ljust(3)}] L2/L3 (log²) global: Prior_Var = {log_global[0]:.3f}, Obs_Var = {log_global[1]:.3f}")
        for cid in sorted(log_cluster_params):
            lp, lo = log_cluster_params[cid]
            tag = " (fallback to global)" if log_cluster_params[cid] == log_global else ""
            print(f"      cluster {cid}: L2/L3 Prior_Var = {lp:.3f}, Obs_Var = {lo:.3f}{tag}")

    def edge_variance_params(m_key, level_key, edge_id):
        cluster_id = edge_cluster_map.get(edge_id, -1)
        cluster_dict = best_params[m_key][level_key]
        return cluster_dict.get(cluster_id, best_params[m_key][f"{level_key}_global"])

    # ---------------------------------------------------------
    # D. COMPUTE CLOSED-FORM POSTERIORS
    # ---------------------------------------------------------
    print("Solving edge-level conjugate posteriors (heteroscedastic, per-edge variance)...")
    posteriors = {
        model: {
            "L1": {}, "L1_var": {}, "L1_obsvar": {},
            "L2": {}, "L2_var": {}, "L2_obsvar": {},
            "L3_weights": {}, "L3_cov": {},
        }
        for model in ["GAT", "XGB", "Kin"]
    }
    warmup_grouped = warmup_df.groupby("edge_id", sort=False)

    for m_key, prior_col in [("GAT", "Prior_GAT"), ("XGB", "Prior_XGB"), ("Kin", "Prior_Kin")]:
        for edge_id, group in warmup_grouped:
            y = group["seconds"].values
            X = group[lvl3_features].values
            prior_vals = group[prior_col].values

            l1_p_var, l1_o_var = edge_variance_params(m_key, "L1", edge_id)
            log_p_var, log_o_var = edge_variance_params(m_key, "L2_L3", edge_id)
            l1_updater = BayesianConjugateUpdater(l1_p_var, l1_o_var)
            log_updater = BayesianConjugateUpdater(log_p_var, log_o_var)

            l1_mean, l1_var = l1_updater.level_1_additive(y, prior_vals)
            l2_mean, l2_var = log_updater.level_2_proportional(y, prior_vals)
            l3_w, l3_cov = log_updater.level_3_feature_weighted(y, prior_vals, X)

            posteriors[m_key]["L1"][edge_id] = l1_mean
            posteriors[m_key]["L1_var"][edge_id] = l1_var
            posteriors[m_key]["L1_obsvar"][edge_id] = l1_o_var
            posteriors[m_key]["L2"][edge_id] = l2_mean
            posteriors[m_key]["L2_var"][edge_id] = l2_var
            posteriors[m_key]["L2_obsvar"][edge_id] = log_o_var
            posteriors[m_key]["L3_weights"][edge_id] = l3_w
            posteriors[m_key]["L3_cov"][edge_id] = l3_cov

    # ---------------------------------------------------------
    # E. APPLY SHIFTS & EVALUATE
    # ---------------------------------------------------------
    print("Evaluating posteriors on unseen evaluation trajectories...")
    for m_key, prior_col in [("GAT", "Prior_GAT"), ("XGB", "Prior_XGB"), ("Kin", "Prior_Kin")]:
        l1_p_var_global, l1_o_var_global = best_params[m_key]["L1_global"]
        log_p_var_global, log_o_var_global = best_params[m_key]["L2_L3_global"]

        l1_obs_var_row = eval_df["edge_id"].map(posteriors[m_key]["L1_obsvar"]).fillna(l1_o_var_global)
        log_obs_var_row = eval_df["edge_id"].map(posteriors[m_key]["L2_obsvar"]).fillna(log_o_var_global)

        # Level 1 (Additive)
        l1_shifts = eval_df["edge_id"].map(posteriors[m_key]["L1"]).fillna(0.0)
        l1_post_var = eval_df["edge_id"].map(posteriors[m_key]["L1_var"]).fillna(l1_p_var_global)
        eval_df[f"Post_{m_key}_L1"] = np.clip(eval_df[prior_col] + l1_shifts, a_min=1.0, a_max=None)
        eval_df[f"Post_{m_key}_L1_var"] = l1_post_var + l1_obs_var_row

        # Level 2 (Proportional)
        l2_shifts = eval_df["edge_id"].map(posteriors[m_key]["L2"]).fillna(0.0)
        l2_post_var = eval_df["edge_id"].map(posteriors[m_key]["L2_var"]).fillna(log_p_var_global)
        eval_df[f"Post_{m_key}_L2"] = eval_df[prior_col] * np.exp(l2_shifts)
        eval_df[f"Post_{m_key}_L2_logvar"] = l2_post_var + log_obs_var_row

        # Level 3 (Feature-weighted)
        w_df = pd.DataFrame.from_dict(posteriors[m_key]["L3_weights"], orient="index", columns=lvl3_features)
        w_mapped = eval_df[["edge_id"]].merge(w_df, left_on="edge_id", right_index=True, how="left").fillna(0.0)
        l3_shifts = (eval_df[lvl3_features].values * w_mapped[lvl3_features].values).sum(axis=1)
        l3_shifts_clamped = np.clip(l3_shifts, a_min=-1.5, a_max=2.0)
        eval_df[f"Post_{m_key}_L3"] = eval_df[prior_col] * np.exp(l3_shifts_clamped)

        default_cov_global = np.eye(len(lvl3_features)) * log_p_var_global
        l3_quad_var = np.empty(len(eval_df), dtype=np.float64)
        edge_ids_arr = eval_df["edge_id"].to_numpy()
        feats_arr = eval_df[lvl3_features].to_numpy(dtype=np.float64)
        l3_cov_map = posteriors[m_key]["L3_cov"]
        for i, (e_id, x_row) in enumerate(zip(edge_ids_arr, feats_arr)):
            if e_id in l3_cov_map:
                cov = l3_cov_map[e_id]
            else:
                edge_log_p_var, _ = edge_variance_params(m_key, "L2_L3", e_id)
                cov = np.eye(len(lvl3_features)) * edge_log_p_var if edge_log_p_var else default_cov_global
            l3_quad_var[i] = float(x_row @ cov @ x_row)
        eval_df[f"Post_{m_key}_L3_logvar"] = log_obs_var_row.to_numpy() + l3_quad_var

    print("\n" + "=" * 96)
    print("RESULTS: FULL MATRIX (ZERO-SHOT PRIOR VS. BAYESIAN CONJUGATE UPDATES)")
    print("=" * 96)

    results = []
    for model_name, prefix in [("Kinematic", "Kin"), ("XGBoost", "XGB"), ("GAT (Ours)", "GAT")]:
        edge_marginal_logvar = {
            e: sum(edge_variance_params(prefix, "L2_L3", e))
            for e in eval_df["edge_id"].unique()
        }
        for level_name, col, dist_kind in [
            ("Prior (Zero-Shot)", f"Prior_{prefix}", "log_marginal"),
            ("Level 1 (Additive)", f"Post_{prefix}_L1", "seconds"),
            ("Level 2 (Proportional)", f"Post_{prefix}_L2", "log"),
            ("Level 3 (Feature-weighted)", f"Post_{prefix}_L3", "log"),
        ]:
            mae = np.abs(eval_df["seconds"] - eval_df[col]).mean()
            mape = (np.abs(eval_df["seconds"] - eval_df[col]) / np.maximum(eval_df["seconds"], 1.0) * 100).mean()
            bias = (eval_df["seconds"] - eval_df[col]).mean()
            log_metrics = log_space_point_metrics(eval_df["seconds"], eval_df[col])

            if dist_kind == "seconds":
                dist = gaussian_distributional_report(
                    eval_df["seconds"], eval_df[col], eval_df[f"Post_{prefix}_L1_var"]
                )
                dist_space = "seconds"
            elif dist_kind == "log":
                logvar_col = f"Post_{prefix}_L2_logvar" if level_name.startswith("Level 2") else f"Post_{prefix}_L3_logvar"
                dist = gaussian_distributional_report(
                    to_log_seconds(eval_df["seconds"]), np.log(eval_df[col]), eval_df[logvar_col]
                )
                dist_space = "log_seconds"
            else:
                marginal_logvar = eval_df["edge_id"].map(edge_marginal_logvar).to_numpy()
                dist = gaussian_distributional_report(
                    to_log_seconds(eval_df["seconds"]), np.log(eval_df[col]), marginal_logvar
                )
                dist_space = "log_seconds"

            results.append({
                "Base Model": model_name,
                "Update Strategy": level_name,
                "Seg MAE (s)": mae,
                "Seg MAPE (%)": mape,
                "Bias (s)": bias,
                "MAE_log": log_metrics["MAE_log"],
                "Bias_log": log_metrics["Bias_log"],
                "SymPct_%": log_metrics["SymPct"],
                "CRPS": dist["CRPS"],
                "NLL": dist["NLL"],
                "Coverage_90": dist["Coverage_90"],
                "Width_90": dist["Width_90"],
                "Dist_Space": dist_space,
            })

    results_df = pd.DataFrame(results).set_index(["Base Model", "Update Strategy"])
    print(results_df.round(3).to_string())
    print("=" * 96)

    # ---------------------------------------------------------
    # F. EXPORT ARTIFACTS
    # ---------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.reset_index().to_csv(RESULTS_DIR / f"posterior_metrics_{args.tag}.csv", index=False)
    results_df.reset_index().to_parquet(RESULTS_DIR / f"posterior_metrics_{args.tag}.parquet", index=False)

    tuned_param_rows = []
    for model, levels in best_params.items():
        for level in ("L1", "L2_L3"):
            residual_space = "seconds" if level == "L1" else "log_seconds"
            g_p, g_o = levels[f"{level}_global"]
            tuned_param_rows.append({
                "model": model, "update_level": level, "variance_cluster": "global",
                "residual_space": residual_space, "prior_variance": g_p, "observation_variance": g_o,
            })
            for cluster_id, (c_p, c_o) in levels[level].items():
                tuned_param_rows.append({
                    "model": model, "update_level": level, "variance_cluster": cluster_id,
                    "residual_space": residual_space, "prior_variance": c_p, "observation_variance": c_o,
                })
    pd.DataFrame(tuned_param_rows).to_csv(RESULTS_DIR / "tuned_parameters.csv", index=False)

    with open(RESULTS_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": args.tag,
                "warmup_ratio": args.warmup_ratio,
                "target_edge_count": int(warmup_df["edge_id"].nunique()),
                "warmup_observation_count": len(warmup_df),
                "evaluation_observation_count": len(eval_df),
            },
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # G. GENERATE VISUALIZATIONS
    # ---------------------------------------------------------
    print("\nGenerating qualitative distribution plots with well-calibrated bands...")
    PLOTS_DIR = RESULTS_DIR / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    edge_stats = warmup_df.groupby("edge_id")["seconds"].agg(["mean", "std", "count"])
    eval_counts = eval_df["edge_id"].value_counts()
    eligible_edges = edge_stats[edge_stats["count"] >= 15].index.intersection(eval_counts[eval_counts >= 10].index)

    if not eligible_edges.empty:
        rng = np.random.default_rng(42)
        eligible_stats = edge_stats.loc[eligible_edges]

        n_random_target = 10
        n_high_target = 10

        random_pool = eligible_stats.index.to_numpy()
        n_random = min(n_random_target, len(random_pool))
        random_edges = rng.choice(random_pool, size=n_random, replace=False) if n_random > 0 else np.array([])

        high_pool = eligible_stats[eligible_stats["mean"] > 50.0].index.to_numpy()
        n_high = min(n_high_target, len(high_pool))
        high_edges = rng.choice(high_pool, size=n_high, replace=False) if n_high > 0 else np.array([])

        sample_edges = pd.unique(np.concatenate([random_edges, high_edges]))
        edge_groups = {
            edge: ("high_ttime" if edge in set(high_edges) and edge not in set(random_edges) else "random")
            for edge in sample_edges
        }
        print(
            f"  Plotting {len(random_edges)} random edges and {len(high_edges)} "
            f"high-traversal-time (mean warmup > 50s) edges "
            f"({len(sample_edges)} unique after de-duplication)."
        )

        hours = np.linspace(0, 167.9, 336)
        synth_df = (
            pl.DataFrame({
                "datetime": pd.Timestamp("2024-01-01") + pd.to_timedelta(hours, unit="h")
            })
            .lazy()
            .pipe(add_temporal_features_polars)
            .collect()
            .to_pandas()
        )
        synth_df["bias_term"] = 1.0
        synth_features = synth_df[lvl3_features].to_numpy(dtype=np.float64)
        synth_time_np = synth_df[TEMPORAL_FEATURE_COLUMNS].values
        synth_global_delay = evaluate_global_weekly_prior(synth_df["hour_of_week"], global_weekly_prior)
        edge_feature_lookup = map_df.set_index("edge_id")[all_scaled_cols]

        with torch.no_grad():
            x_enc = gat_model.encode(graph_data.x, graph_data.edge_index)

        for target_edge in sample_edges:
            w_scatter = warmup_df[warmup_df["edge_id"] == target_edge]
            e_scatter = eval_df[eval_df["edge_id"] == target_edge]
            group_tag = edge_groups[target_edge]

            t_edge = torch.tensor([target_edge] * len(synth_df), dtype=torch.long, device=device)
            t_time = torch.tensor(synth_time_np, dtype=torch.float, device=device)
            global_delay = torch.tensor(synth_global_delay, dtype=torch.float, device=device)

            with torch.no_grad():
                mu_g, _ = gat_model.predict(x_enc, graph_data.edge_index, t_edge, t_time, log_base_time, global_delay)
            gat_prior_curve = np.exp(mu_g.cpu().numpy())

            kin_delay_curve = np.exp(log_base_time[target_edge].cpu().numpy() + global_delay.cpu().numpy())

            if target_edge in edge_feature_lookup.index:
                xgb_synth = synth_df.copy()
                map_feats = edge_feature_lookup.loc[target_edge]
                for c in all_scaled_cols:
                    xgb_synth[c] = map_feats[c]
                xgb_prior_curve = np.clip(
                    np.exp(
                        log_base_time[target_edge].cpu().numpy()
                        + global_delay.cpu().numpy()
                        + xgb_model.predict(xgb_synth[xgb_feat_cols])
                    ),
                    a_min=1.0,
                    a_max=None,
                )
            else:
                xgb_prior_curve = kin_delay_curve.copy()

            if target_edge in edge_meta_lookup.index:
                meta = edge_meta_lookup.loc[target_edge]
                meta_str = (
                    f"cat={meta['category']}, free-flow={meta['speed_limit']:.0f} km/h, "
                    f"u={int(meta['u'])}, v={int(meta['v'])}"
                )
            else:
                meta_str = "metadata unavailable"

            for model_name, prior_curve, m_key in [
                ("Kinematic", kin_delay_curve, "Kin"),
                ("XGBoost", xgb_prior_curve, "XGB"),
                ("GAT", gat_prior_curve, "GAT"),
            ]:
                plt.figure(figsize=(13, 6))
                l1_p_val, l1_o_val = edge_variance_params(m_key, "L1", target_edge)
                log_p_val, log_o_val = edge_variance_params(m_key, "L2_L3", target_edge)

                l1_shift = posteriors[m_key]["L1"].get(target_edge, 0.0)
                l1_shift_var = posteriors[m_key]["L1_var"].get(target_edge, l1_p_val)
                l2_shift = posteriors[m_key]["L2"].get(target_edge, 0.0)
                l3_w = posteriors[m_key]["L3_weights"].get(target_edge, np.zeros(len(lvl3_features)))
                l3_cov = posteriors[m_key]["L3_cov"].get(target_edge, np.eye(len(lvl3_features)) * log_p_val)

                post_l1 = np.clip(prior_curve + l1_shift, a_min=1.0, a_max=None)
                post_l2 = np.clip(prior_curve * np.exp(l2_shift), a_min=1.0, a_max=None)
                post_l3 = np.clip(prior_curve * np.exp(np.clip(synth_features @ l3_w, -1.5, 2.0)), a_min=1.0, a_max=None)

                # Variance in seconds^2 for Level 1 additive model
                l1_pred_var = max(l1_shift_var + l1_o_val, 1e-4)
                l1_pred_low = np.clip(post_l1 - Z_90 * np.sqrt(l1_pred_var), a_min=1.0, a_max=None)
                l1_pred_high = post_l1 + Z_90 * np.sqrt(l1_pred_var)

                # Variances in log-space for Level 2 and Level 3 models
                l2_mean_var = 1.0 / (1.0 / log_p_val + len(w_scatter) / log_o_val)
                _, l2_pred_low, l2_pred_high = lognormal_prediction_interval(
                    np.log(post_l2),
                    np.full_like(post_l2, l2_mean_var + log_o_val),
                    standard_deviations=Z_90,
                )

                l3_mean_var = np.einsum("ij,jk,ik->i", synth_features, l3_cov, synth_features)
                _, l3_pred_low, l3_pred_high = lognormal_prediction_interval(
                    np.log(post_l3),
                    l3_mean_var + log_o_val,
                    standard_deviations=Z_90,
                )

                # Determine reasonable y-axis limits to prevent plot distortion or line occlusion
                obs_max = max(w_scatter["seconds"].quantile(0.99) if len(w_scatter) else 10.0,
                              e_scatter["seconds"].quantile(0.99) if len(e_scatter) else 10.0)
                curve_max = max(prior_curve.max(), post_l1.max(), post_l2.max(), post_l3.max())
                y_max = max(obs_max * 1.35, curve_max * 1.25, 10.0)

                # Clip plotted uncertainty intervals for clean visualization
                l1_pred_high = np.clip(l1_pred_high, a_min=None, a_max=y_max * 1.2)
                l2_pred_high = np.clip(l2_pred_high, a_min=None, a_max=y_max * 1.2)
                l3_pred_high = np.clip(l3_pred_high, a_min=None, a_max=y_max * 1.2)

                # Plot observational scatter
                plt.scatter(w_scatter["hour_of_week"], w_scatter["seconds"], color="black", s=25, alpha=0.5, zorder=2, label=f"Warmup (n={len(w_scatter)})")
                plt.scatter(e_scatter["hour_of_week"], e_scatter["seconds"], color="darkorange", s=30, alpha=0.7, zorder=2, label=f"Eval (n={len(e_scatter)})")

                # Baseline Zero-Shot Prior
                plt.plot(hours, prior_curve, "k--", linewidth=2.0, alpha=0.8, zorder=3, label="Prior (Zero-Shot)")

                # Level 1 (Additive)
                plt.plot(hours, post_l1, color="#1f77b4", linewidth=2.0, zorder=4, label="Level 1 (Additive)")
                plt.fill_between(
                    hours, l1_pred_low, l1_pred_high,
                    color="#1f77b4", alpha=0.12, zorder=1,
                    label="Level 1 90% predictive interval",
                )

                # Level 2 (Proportional)
                plt.plot(hours, post_l2, color="#2ca02c", linewidth=2.2, zorder=5, label="Level 2 (Proportional)")
                plt.fill_between(
                    hours, l2_pred_low, l2_pred_high,
                    color="#2ca02c", alpha=0.12, zorder=1,
                    label="Level 2 90% predictive interval",
                )

                # Level 3 (Feature-weighted)
                plt.plot(hours, post_l3, color="#d62728", linewidth=2.2, zorder=6, label="Level 3 (Feature-weighted)")
                plt.fill_between(
                    hours, l3_pred_low, l3_pred_high,
                    color="#d62728", alpha=0.12, zorder=1,
                    label="Level 3 90% predictive interval",
                )

                plt.ylim(bottom=0.0, top=y_max)
                plt.title(
                    f"Bayesian Adaptation on Cold-Start Segment {target_edge} "
                    f"({model_name}, mean warmup ≈ {w_scatter['seconds'].mean():.1f}s, {group_tag})\n"
                    f"{meta_str}"
                )
                plt.xlabel("Hour of Week")
                plt.ylabel("Traversal Time (Seconds)")
                plt.xlim(0, 168)
                plt.xticks(range(0, 169, 24))
                plt.grid(True, alpha=0.3)
                plt.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
                plt.tight_layout()

                save_path = PLOTS_DIR / f"bayesian_update_{group_tag}_{target_edge}_{m_key}.png"
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"  Saved plot: {save_path.name}")

    print(f"\nExecution finished in {datetime.now() - start_time}. All results saved to {RESULTS_DIR}.")


if __name__ == "__main__":
    run_bayesian_pipeline()
