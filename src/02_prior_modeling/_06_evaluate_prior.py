"""
06_evaluate_prior.py

This script evaluates the three continuous-time travel priors (Kinematic, 
XGBoost, and the Bayesian GAT) across the full map-matched trajectory dataset. 

Key Operations:
- Reproduces exact feature scaling and categorical encoding from training.
- Boundary traversals (first/last segment per trip) were already removed
  upstream in 04, before the cold-start edge split, so this script does not
  repeat that filter.
- Computes Segment-Level metrics (MAE, MAPE).
- Computes Trip-Level metrics, stratified by "Cold Start Cohorts" to 
  demonstrate how each model gracefully degrades as spatial sparsity increases.
- Generates 24-hour temporal distribution plots to visually validate the 
  dynamic variance and temporal mapping of the GAT versus the baselines.
""" 

import os
import pandas as pd
import numpy as np
import polars as pl
import torch
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb
import argparse
import json
import gc

from _05_train_prior import (
    BayesianTrafficGAT, TEMPORAL_FEATURE_COLUMNS, add_temporal_features_polars, evaluate_global_weekly_prior,
)
from _metrics_common import (
    fit_duan_smearing, backtransform_seconds, lognormal_mean_backtransform,
    fit_variance_scale, gaussian_quantiles, ResidualQuantileCalibrator, full_distributional_report,
    to_log_seconds, log_space_point_metrics_from_log_prediction, mape_percent,
)

parser = argparse.ArgumentParser()
parser.add_argument("--tag", required=True)
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parents[2]
COLD_START_DIR = BASE_DIR / 'data' / '04_cold_start'
MODELS_DIR = BASE_DIR / 'models' / 'prior_weights' / args.tag
RESULTS_DIR = BASE_DIR / 'results' / 'prior_modeling' / args.tag / 'prior_evaluation'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def verify_model():
    print(f"--- VERIFYING MODELS ON {device.type.upper()} ---")

    # 1. Load Artifacts, GAT, & XGBoost
    with open(os.path.join(MODELS_DIR, "graph_artifacts.pkl"), 'rb') as f:
        artifacts = pickle.load(f)

    graph_data = artifacts['graph_data'].to(device)
    log_base_time = artifacts['log_base_time'].to(device)
    train_edge_ids = artifacts['train_edge_ids']
    
    with open(MODELS_DIR / "global_weekly_prior.json", encoding="utf-8") as file:
        global_weekly_prior = json.load(file)

    model = BayesianTrafficGAT(
        static_in_channels=graph_data.x.shape[1],
        time_channels=len(TEMPORAL_FEATURE_COLUMNS)
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "GAT_prior_weights.pth"), weights_only=True))
    model.eval()

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(MODELS_DIR, "xgboost_weights.json"))

    # 2. Build Scaled Map Features using Pandas for Scikit-Learn compatibility
    map_parquet_path = COLD_START_DIR / f"cold_start_graph_{args.tag}.parquet"
    map_df = pd.read_parquet(map_parquet_path)

    cat_dummies = pd.get_dummies(map_df['category'], prefix='cat', dtype=float)
    map_df = pd.concat([map_df, cat_dummies], axis=1)

    cont_cols = ['speed_limit', 'segment_length', 'tortuosity']
    cat_cols = list(cat_dummies.columns)
    all_scaled_cols = cont_cols + cat_cols
    map_df[all_scaled_cols] = artifacts['scaler'].transform(map_df[all_scaled_cols])
    
    map_df['edge_id'] = np.arange(len(map_df), dtype=np.int32)
    xgb_feat_cols = cont_cols + TEMPORAL_FEATURE_COLUMNS + cat_cols

    # Convert map features to Polars for instant joins
    pl_map_features = pl.from_pandas(map_df[['u', 'v', 'segmentkey', 'edge_id'] + cont_cols + cat_cols])

    # 3. Model Inference Helper
    EVAL_BATCH_SIZE = 131072

    def run_inference(edge_ids_np, time_feats_np, hour_of_week_np, xgb_matrix_np):
        n = len(edge_ids_np)
        edge_ids_t = torch.from_numpy(edge_ids_np.astype(np.int64)).to(device)
        time_t = torch.from_numpy(time_feats_np.astype(np.float32)).to(device)
        
        global_delay_np = evaluate_global_weekly_prior(hour_of_week_np, global_weekly_prior)
        global_delay_t = torch.from_numpy(global_delay_np.astype(np.float32)).to(device)

        mu_chunks, logvar_chunks = [], []
        with torch.no_grad():
            x_enc = model.encode(graph_data.x, graph_data.edge_index)
            for i in range(0, n, EVAL_BATCH_SIZE):
                b_edges = edge_ids_t[i:i + EVAL_BATCH_SIZE]
                b_time = time_t[i:i + EVAL_BATCH_SIZE]
                b_delay = global_delay_t[i:i + EVAL_BATCH_SIZE]
                b_mu, b_logvar = model.predict(x_enc, graph_data.edge_index, b_edges, b_time, log_base_time, b_delay)
                mu_chunks.append(b_mu.cpu())
                logvar_chunks.append(b_logvar.cpu())
            del x_enc

        mu_gat = torch.cat(mu_chunks).numpy()
        sigma2_gat = np.exp(torch.cat(logvar_chunks).numpy())

        log_base_np = log_base_time[edge_ids_t].cpu().numpy()
        log_pred_kin = log_base_np
        log_pred_global = log_base_np + global_delay_np

        xgb_residual = xgb_model.predict(xgb_matrix_np)
        log_pred_xgb = log_base_np + global_delay_np + xgb_residual

        return {
            'mu_gat': mu_gat, 'sigma2_gat': sigma2_gat,
            'log_pred_kin': log_pred_kin, 'log_pred_global': log_pred_global, 'log_pred_xgb': log_pred_xgb,
        }

    # 4. Calibration on Validation Split
    print("Fitting calibration on held-out validation split...")
    cal_df = artifacts['calibration_df'].copy()
    cal_df = cal_df.merge(map_df[['edge_id'] + cont_cols + cat_cols], on='edge_id', how='inner')

    cal_preds = run_inference(
        cal_df['edge_id'].values,
        cal_df[TEMPORAL_FEATURE_COLUMNS].values,
        cal_df['hour_of_week'].values,
        cal_df[xgb_feat_cols].values
    )
    log_true_cal = to_log_seconds(cal_df['seconds'])

    gat_var_scale = fit_variance_scale(log_true_cal, cal_preds['mu_gat'], cal_preds['sigma2_gat'])
    print(f"  GAT variance-scale factor (s^2={gat_var_scale:.3f})")

    smearing = {
        'Kin': fit_duan_smearing(log_true_cal, cal_preds['log_pred_kin']),
        'Global': fit_duan_smearing(log_true_cal, cal_preds['log_pred_global']),
        'XGB': fit_duan_smearing(log_true_cal, cal_preds['log_pred_xgb']),
    }
    conformal = {
        'Kin': ResidualQuantileCalibrator().fit(log_true_cal, cal_preds['log_pred_kin']),
        'Global': ResidualQuantileCalibrator().fit(log_true_cal, cal_preds['log_pred_global']),
        'XGB': ResidualQuantileCalibrator().fit(log_true_cal, cal_preds['log_pred_xgb']),
    }

    del cal_df, cal_preds
    gc.collect()

    # 5. Load and Process Test Trajectories with Polars
    print("\nLoading held-out cold-start test trajectories via Polars...")
    test_path = COLD_START_DIR / f"test_traj_{args.tag}.parquet"

    test_pl = (
        pl.scan_parquet(test_path)
        .pipe(add_temporal_features_polars)
        .join(pl_map_features.lazy(), on=['u', 'v', 'segmentkey'], how='inner')
        .with_columns(
            pl.col('edge_id').is_in(list(train_edge_ids)).not_().alias('is_cold_start')
        )
        .collect()
    )
    print(f"Test trajectory rows loaded: {len(test_pl):,} observations")

    # Extract NumPy matrices for zero-copy inference
    edge_ids_np = test_pl['edge_id'].to_numpy()
    time_feats_np = test_pl.select(TEMPORAL_FEATURE_COLUMNS).to_numpy()
    how_np = test_pl['hour_of_week'].to_numpy()
    xgb_mat_np = test_pl.select(xgb_feat_cols).to_numpy()
    seconds_np = test_pl['seconds'].to_numpy()

    print("Running Inference on test set...")
    test_preds = run_inference(edge_ids_np, time_feats_np, how_np, xgb_mat_np)

    sigma2_gat_test = test_preds['sigma2_gat'] * gat_var_scale
    pred_sec_gat = lognormal_mean_backtransform(test_preds['mu_gat'], sigma2_gat_test)
    pred_sec_kin = backtransform_seconds(test_preds['log_pred_kin'], smearing['Kin'])
    pred_sec_global = backtransform_seconds(test_preds['log_pred_global'], smearing['Global'])
    pred_sec_xgb = backtransform_seconds(test_preds['log_pred_xgb'], smearing['XGB'])

    # Append predictions back to Polars for cohort analysis
    test_pl = test_pl.with_columns([
        pl.Series('Pred_Sec_GAT', pred_sec_gat),
        pl.Series('Pred_Sec_Kin', pred_sec_kin),
        pl.Series('Pred_Sec_Global_Weekly', pred_sec_global),
        pl.Series('Pred_Sec_XGB', pred_sec_xgb),
    ])

    # 6. Segment-Level Evaluation
    print("\n" + "=" * 100)
    print("SEGMENT-LEVEL EVALUATION -- LOG-SPACE (PRIMARY) + SECONDS-SPACE (SECONDARY)")
    print("=" * 100)

    model_specs = [
        ('1. Kinematic', pred_sec_kin, test_preds['log_pred_kin'], None, None),
        ('2. Global Weekly', pred_sec_global, test_preds['log_pred_global'], None, None),
        ('3. XGBoost', pred_sec_xgb, test_preds['log_pred_xgb'], None, None),
        ('4. GAT (Ours)', pred_sec_gat, test_preds['mu_gat'], test_preds['mu_gat'], sigma2_gat_test),
    ]
    conformal_key = {'1. Kinematic': 'Kin', '2. Global Weekly': 'Global', '3. XGBoost': 'XGB'}

    seg_metrics = []
    for m_name, sec_pred, log_pred, mu_for_nll, sigma2_for_nll in model_specs:
        point = log_space_point_metrics_from_log_prediction(seconds_np, log_pred)
        abs_err_sec = np.abs(seconds_np - sec_pred)
        mape_sec = mape_percent(seconds_np, sec_pred)
        bias_sec = np.mean(seconds_np - sec_pred)

        if m_name in conformal_key:
            quantile_dict = conformal[conformal_key[m_name]].predict_quantiles(log_pred)
        else:
            quantile_dict = gaussian_quantiles(mu_for_nll, np.sqrt(sigma2_for_nll))
        dist = full_distributional_report(seconds_np, quantile_dict, mu_for_nll, sigma2_for_nll)

        seg_metrics.append({
            'Model': m_name,
            'Avg_True_Time': np.mean(seconds_np),
            'MAE_log': point['MAE_log'],
            'Bias_log': point['Bias_log'],
            'SymPct_%': point['SymPct'],
            'Seg_MAE_s': np.mean(abs_err_sec),
            'Seg_MAPE_%': mape_sec,
            'Seg_Bias_s': bias_sec,
            'CRPS_proxy_log': dist['CRPS_proxy'],
            'Coverage_90': dist.get('coverage_90', np.nan),
            'Coverage_80': dist.get('coverage_80', np.nan),
            'Coverage_50': dist.get('coverage_50', np.nan),
            'Width_90_log': dist.get('sharpness_90_log', np.nan),
            'NLL_log': dist['NLL'],
        })

    segment_results = pd.DataFrame(seg_metrics)
    print(segment_results.round(3).set_index('Model').to_string())

    # 7. Fast Trip-Level Cohort Analysis in Polars
    print("\n" + "=" * 110)
    print("TRIP-LEVEL EVALUATION BY COLD START COHORT")
    print("=" * 110)

    traj_stats_pl = (
        test_pl.group_by('trip_id')
        .agg([
            pl.col('seconds').sum().alias('total_true_time'),
            pl.col('Pred_Sec_GAT').sum().alias('total_pred_gat'),
            pl.col('Pred_Sec_XGB').sum().alias('total_pred_xgb'),
            pl.col('Pred_Sec_Kin').sum().alias('total_pred_kin'),
            pl.len().alias('segment_count'),
            pl.col('is_cold_start').sum().alias('cold_start_count'),
        ])
        .with_columns([
            (pl.col('cold_start_count') / pl.col('segment_count')).alias('cold_start_ratio'),
            (pl.col('total_true_time') - pl.col('total_pred_gat')).alias('GAT_Bias'),
            (pl.col('total_true_time') - pl.col('total_pred_xgb')).alias('XGB_Bias'),
            (pl.col('total_true_time') - pl.col('total_pred_kin')).alias('Kin_Bias'),
        ])
        .with_columns([
            pl.col('GAT_Bias').abs().alias('GAT_MAE'),
            pl.col('XGB_Bias').abs().alias('XGB_MAE'),
            pl.col('Kin_Bias').abs().alias('Kin_MAE'),
            (pl.col('GAT_Bias').abs() / pl.col('total_true_time') * 100.0).alias('GAT_MAPE'),
            (pl.col('XGB_Bias').abs() / pl.col('total_true_time') * 100.0).alias('XGB_MAPE'),
            (pl.col('Kin_Bias').abs() / pl.col('total_true_time') * 100.0).alias('Kin_MAPE'),
        ])
    )

    traj_stats = traj_stats_pl.to_pandas()

    def assign_cohort(ratio):
        if ratio == 0.0: return "0% Unseen"
        elif ratio <= 0.19: return "1-19% Unseen"
        elif ratio <= 0.39: return "20-39% Unseen"
        elif ratio <= 0.59: return "40-59% Unseen"
        elif ratio <= 0.79: return "60-79% Unseen"
        elif ratio <= 0.99: return "80-99% Unseen"
        else: return "100% Unseen"

    cohort_order = [
        "0% Unseen", "1-19% Unseen", "20-39% Unseen", 
        "40-59% Unseen", "60-79% Unseen", "80-99% Unseen", "100% Unseen"
    ]
    traj_stats['Cohort'] = pd.Categorical(traj_stats['cold_start_ratio'].apply(assign_cohort), categories=cohort_order, ordered=True)

    cohort_results = traj_stats.groupby('Cohort', observed=False).agg(
        Trips=('trip_id', 'count'),
        Avg_Segs=('segment_count', 'mean'),
        Avg_Time=('total_true_time', 'mean'),
        Kin_Bias=('Kin_Bias', 'mean'),
        XGB_Bias=('XGB_Bias', 'mean'),
        GAT_Bias=('GAT_Bias', 'mean'),
        Kin_MAE=('Kin_MAE', 'mean'),
        XGB_MAE=('XGB_MAE', 'mean'),
        GAT_MAE=('GAT_MAE', 'mean'),
        Kin_MAPE=('Kin_MAPE', 'mean'),
        XGB_MAPE=('XGB_MAPE', 'mean'),
        GAT_MAPE=('GAT_MAPE', 'mean')
    ).round(2)
    print("=" * 110)

    # 8. Save Metrics & Feature Importance
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    segment_results.to_csv(RESULTS_DIR / f"prior_metrics_{args.tag}.csv", index=False)
    segment_results.to_parquet(RESULTS_DIR / f"prior_metrics_{args.tag}.parquet", index=False)
    cohort_results.reset_index().to_csv(RESULTS_DIR / f"trip_cohort_metrics_{args.tag}.csv", index=False)
    cohort_results.reset_index().to_parquet(RESULTS_DIR / f"trip_cohort_metrics_{args.tag}.parquet", index=False)

    feat_imp_df = pd.DataFrame({
        'Feature': xgb_feat_cols,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    feat_imp_df.to_csv(RESULTS_DIR / "xgboost_feature_importance.csv", index=False)

    with open(RESULTS_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "tag": args.tag,
            "trajectory_rows_evaluated": len(test_pl),
            "trip_count_evaluated": len(traj_stats),
            "cold_start_edge_count": int(test_pl['is_cold_start'].sum()),
            "xgb_conformal_halfwidth_log_90": conformal['XGB'].symmetric_halfwidth(0.10),
            "kinematic_conformal_halfwidth_log_90": conformal['Kin'].symmetric_halfwidth(0.10),
            "gat_variance_scale": gat_var_scale,
            "calibration_row_count": len(artifacts['calibration_df']),
        }, f, indent=2)

    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1], color='teal')
    plt.xlabel('Relative Importance (Gain)')
    plt.title('XGBoost Baseline Feature Importance')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "xgboost_feature_importance.png", dpi=300)
    plt.close()

    # 9. Top-10 Edge Distribution Plots
    print("\nGenerating batch of 10 weekly distribution plots...")
    PLOTS_DIR = RESULTS_DIR / 'distribution_plots'
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    edge_counts = test_pl.group_by('edge_id').agg(pl.len().alias('count')).filter(pl.col('count') >= 50)
    if len(edge_counts) > 0:
        top_10_edges = edge_counts.sort('count', descending=True).head(10)['edge_id'].to_list()
        xgb_halfwidth = conformal['XGB'].symmetric_halfwidth(0.10)
        kin_halfwidth = conformal['Kin'].symmetric_halfwidth(0.10)

        # Slice out only data needed for the top 10 edges to keep plot generation light
        top_edges_df = test_pl.filter(pl.col('edge_id').is_in(top_10_edges)).select(['edge_id', 'hour_of_week', 'seconds']).to_pandas()

        for target_edge in top_10_edges:
            plot_daily_distribution(
                model, xgb_model, map_df, graph_data, log_base_time, global_weekly_prior,
                target_edge, top_edges_df, PLOTS_DIR, xgb_halfwidth, kin_halfwidth, gat_var_scale
            )
        print(f"All 10 plots successfully saved to: {PLOTS_DIR}")

def plot_daily_distribution(model, xgb_model, map_df, graph_data, log_base_time, global_weekly_prior, target_edge_id, test_df, output_dir, xgb_halfwidth_log, kin_halfwidth_log, gat_var_scale=1.0):
    hours = np.linspace(0, 167.9, 336)
    
    synthetic_lazy = pl.DataFrame({
        "datetime": pd.Timestamp("2024-01-01") + pd.to_timedelta(hours, unit="h")
    }).lazy().pipe(add_temporal_features_polars).collect()

    t_edge = torch.tensor([target_edge_id] * len(synthetic_lazy), dtype=torch.long, device=device)
    t_time = torch.tensor(synthetic_lazy.select(TEMPORAL_FEATURE_COLUMNS).to_numpy(), dtype=torch.float, device=device)
    global_delay = torch.tensor(
        evaluate_global_weekly_prior(synthetic_lazy["hour_of_week"].to_numpy(), global_weekly_prior),
        dtype=torch.float,
        device=device
    )

    with torch.no_grad():
        encoded_graph = model.encode(graph_data.x, graph_data.edge_index)
        mu, log_var = model.predict(encoded_graph, graph_data.edge_index, t_edge, t_time, log_base_time, global_delay)
    
    mu_np = mu.cpu().numpy()
    var = np.exp(log_var.cpu().numpy()) * gat_var_scale
    pred_mean = lognormal_mean_backtransform(mu_np, var)
    pred_std = np.sqrt(np.expm1(var)) * np.exp(mu_np + var / 2)

    map_features = map_df[map_df['edge_id'] == target_edge_id].iloc[0]
    cat_cols = [c for c in map_df.columns if c.startswith('cat_')]
    xgb_feat_cols = ['speed_limit', 'segment_length', 'tortuosity'] + TEMPORAL_FEATURE_COLUMNS + cat_cols

    xgb_synth_df = synthetic_lazy.to_pandas()
    for col in ['speed_limit', 'segment_length', 'tortuosity'] + cat_cols:
        xgb_synth_df[col] = map_features[col]
    
    log_pred_xgb = (
        log_base_time[target_edge_id].cpu().numpy()
        + global_delay.cpu().numpy()
        + xgb_model.predict(xgb_synth_df[xgb_feat_cols])
    )
    pred_xgb = np.exp(log_pred_xgb)
    xgb_upper = np.exp(log_pred_xgb + xgb_halfwidth_log)
    xgb_lower = np.exp(log_pred_xgb - xgb_halfwidth_log)

    base_log = log_base_time[target_edge_id].cpu().numpy()
    base_time_sec = np.exp(base_log)
    base_time_array = np.full_like(hours, base_time_sec)
    kin_upper = np.exp(base_log + kin_halfwidth_log)
    kin_lower = np.exp(base_log - kin_halfwidth_log)

    edge_data = test_df[test_df['edge_id'] == target_edge_id]

    plt.figure(figsize=(14, 7))
    plt.fill_between(hours, kin_lower, kin_upper, color='red', alpha=0.05, label='Kinematic Variance', zorder=1)
    plt.fill_between(hours, xgb_lower, xgb_upper, color='green', alpha=0.1, label='XGBoost Variance', zorder=1)
    plt.fill_between(hours, pred_mean - pred_std, pred_mean + pred_std, color='blue', alpha=0.2, label='GAT Variance', zorder=1)
    
    plt.scatter(edge_data['hour_of_week'], edge_data['seconds'], color='black', alpha=0.3, s=8, label=f'True Traversals (n={len(edge_data)})', zorder=2)
    
    plt.plot(hours, base_time_array, color='red', linestyle='--', linewidth=2, label=f'Kinematic Mean ({base_time_sec:.1f}s)', zorder=3)
    plt.plot(hours, pred_xgb, label='XGBoost Mean', color='green', linestyle=':', linewidth=2.5, zorder=3)
    plt.plot(hours, pred_mean, label='GAT Mean', color='blue', linewidth=2.5, zorder=3)
    
    plt.title(f"Weekly Temporal Prior Comparison (Edge {target_edge_id})")
    plt.xlabel("Continuous Hour of Week (0 = Mon 00:00, 167.9 = Sun 23:59)")
    plt.ylabel("Traversal Time (Seconds)")
    plt.xlim(0, 168)
    plt.xticks(range(0, 169, 12))
    plt.grid(True, alpha=0.3, zorder=0)
    for day_boundary in range(0, 169, 24):
        plt.axvline(day_boundary, color='gray', linestyle='--', alpha=0.4, zorder=0)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"distribution_edge_{target_edge_id}_film.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    verify_model()
