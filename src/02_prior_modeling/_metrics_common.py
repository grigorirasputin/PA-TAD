"""
_metrics_common.py

Shared, log-space-first evaluation utilities for the PA-TAD prior /
Bayesian-update pipeline (used by _06_evaluate_prior.py and
_07_bayesian_update.py).

DESIGN CHOICES (flag any of these for discussion before relying on results):

1. Everything is compared in log-seconds space by default. That is the
   native space of the GAT (trained via Gaussian NLL on log(seconds)) and
   of the XGBoost residual head (trained on
   log(seconds) - log_base_time - global_weekly_delay). Comparing raw
   seconds-space Bias/MAE conflates "how good is the model" with "which
   space happened to be its training objective", which is exactly the
   issue Table 1 had (XGBoost's near-zero Bias(s) is an artifact of its
   objective, not evidence of calibration).

2. Any uncertainty needed for a baseline without a native predictive
   distribution (Kinematic, Global-Weekly, XGBoost) is estimated on a
   held-out CALIBRATION set (the val split saved by 05), never on the
   test/eval set the headline numbers are reported on. This is exactly
   split-conformal prediction's train -> calibrate -> test discipline,
   and it also fixes a real bug: the previous code computed
   xgb_log_std / kin_log_std directly on the test set.

3. Seconds-space numbers are only ever produced via a single, explicitly
   bias-corrected back-transform (log-normal mean for GAT, which has a
   sigma^2; Duan smearing -- fit on the calibration set -- for point
   models). Naive exp(mu) reports the median of a log-normal, not its
   mean, which silently injects a downward bias whenever sigma^2 > 0.

4. All four models (Kinematic, Global-Weekly, XGBoost, GAT) get compared
   on the SAME distributional scoring rule -- mean pinball loss across a
   shared quantile grid, i.e. a CRPS proxy -- even though only GAT has a
   native parametric density. GAT's quantiles come from its (recalibrated)
   Gaussian; the others' come from empirical conformal residual quantiles.
   GAT additionally gets a genuine Gaussian NLL, reported as a bonus
   column and marked N/A for the point models, since forcing a Gaussian
   NLL onto them would quietly reintroduce the assumption conformal is
   designed to avoid.
"""

import numpy as np
from scipy.stats import norm

EPS = 1e-6
DEFAULT_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


# ---------------------------------------------------------------------
# 1. LOG-SPACE POINT METRICS
# ---------------------------------------------------------------------

def to_log_seconds(seconds):
    """Safe log transform shared by every script that touches raw seconds."""
    return np.log(np.clip(np.asarray(seconds, dtype=np.float64), 1.0, None))


def log_space_point_metrics(y_true_sec, y_pred_sec):
    """MAE / Bias / symmetric-percentage-error, all computed on log(seconds).

    SymPct uses the log-accuracy-ratio back-transform
        100 * (exp(|log(y_true) - log(y_pred)|) - 1)
    which reduces to ordinary percentage error for small errors but stays
    symmetric and well-behaved for large ones (plain MAPE blows up
    asymmetrically as y_pred -> 0 and is capped at 100% on the other side).
    """
    log_true = to_log_seconds(y_true_sec)
    log_pred = to_log_seconds(y_pred_sec)
    delta = log_true - log_pred  # true - pred, so +ve = under-prediction
    return {
        "MAE_log": float(np.mean(np.abs(delta))),
        "Bias_log": float(np.mean(delta)),
        "SymPct": float(100.0 * np.mean(np.expm1(np.abs(delta)))),
    }


def log_space_point_metrics_from_log_prediction(y_true_sec, log_pred):
    """Log-space point metrics for a model's native log prediction."""
    log_true = to_log_seconds(y_true_sec)
    log_pred = np.asarray(log_pred, dtype=np.float64)
    delta = log_true - log_pred
    return {
        "MAE_log": float(np.mean(np.abs(delta))),
        "Bias_log": float(np.mean(delta)),
        "SymPct": float(100.0 * np.mean(np.expm1(np.abs(delta)))),
    }


def mape_percent(y_true, y_pred, eps=1e-3):
    """Safe seconds-space Mean Absolute Percentage Error in percent (%)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


# ---------------------------------------------------------------------
# 2. CORRECTED BACK-TRANSFORM TO SECONDS (fit on calibration set only)
# ---------------------------------------------------------------------

def fit_duan_smearing(log_true_cal, log_pred_cal):
    """Duan's smearing estimator: E[y] ~= exp(pred_log) * mean(exp(residual)),
    residual fit on a held-out calibration set. Distribution-free correction
    for the exp(median) != mean bias whenever residual variance > 0.
    """
    resid = np.asarray(log_true_cal) - np.asarray(log_pred_cal)
    return float(np.mean(np.exp(resid)))


def backtransform_seconds(log_pred, smearing_factor):
    return np.exp(log_pred) * smearing_factor


def lognormal_mean_backtransform(mu_log, sigma2_log):
    """Exact log-normal mean, for models (GAT) with an explicit sigma^2."""
    return np.exp(mu_log + 0.5 * sigma2_log)


def lognormal_prediction_interval(mu_log, sigma2_log, standard_deviations=2.0):
    """Median and positive, asymmetric lognormal prediction bounds."""
    mu_log = np.asarray(mu_log, dtype=np.float64)
    log_std = np.sqrt(np.clip(np.asarray(sigma2_log, dtype=np.float64), EPS, None))
    offset = standard_deviations * log_std
    return np.exp(mu_log), np.exp(mu_log - offset), np.exp(mu_log + offset)


# ---------------------------------------------------------------------
# 3. GAT NATIVE / RECALIBRATED GAUSSIAN NLL
# ---------------------------------------------------------------------

def gaussian_nll(log_true, mu_log, sigma2_log):
    sigma2_log = np.clip(sigma2_log, EPS, None)
    return float(np.mean(
        0.5 * np.log(2 * np.pi * sigma2_log)
        + 0.5 * (log_true - mu_log) ** 2 / sigma2_log
    ))


def fit_variance_scale(log_true_cal, mu_log_cal, sigma2_log_cal):
    """Closed-form MLE scale factor s^2 such that
    sigma2_calibrated = s^2 * sigma2_raw
    minimizes Gaussian NLL on the calibration set. Standard 'temperature
    scaling' fix for a Gaussian head whose reported variance is
    systematically over- or under-confident. (MLE of a scalar multiplicative
    variance correction is just the mean squared standardized residual.)
    """
    z2 = (np.asarray(log_true_cal) - np.asarray(mu_log_cal)) ** 2 / np.clip(sigma2_log_cal, EPS, None)
    return float(np.mean(z2))  # this is s^2, apply as sigma2 * s2_hat


# ---------------------------------------------------------------------
# 4. QUANTILES / SPLIT-CONFORMAL RESIDUAL CALIBRATION
# ---------------------------------------------------------------------

class ResidualQuantileCalibrator:
    """Empirical (split-conformal-style) predictive quantiles for a point
    forecaster with no native uncertainty (Kinematic, Global-Weekly,
    XGBoost).

    Fit on a held-out CALIBRATION set: quantile_shift(level) is the
    finite-sample-corrected empirical quantile of the SIGNED log-space
    residual (log_true - log_pred). At prediction time, the level-q
    predictive quantile of log(seconds) for a new point is
        log_pred + quantile_shift(level).

    `symmetric_halfwidth(alpha)` instead uses the standard split-conformal
    correction on ABSOLUTE residuals; the resulting interval
    [pred - q, pred + q] carries a finite-sample marginal coverage
    guarantee of at least 1-alpha under exchangeability of calibration and
    test residuals (Lei et al. 2018 / Vovk et al.). The signed per-level
    quantiles above do not carry that formal guarantee individually (no
    (n+1)/n correction is exactly conservative in the two-sided/asymmetric
    case), but they are the standard, defensible way to get an empirical
    predictive distribution for pinball-loss / CRPS scoring, and coverage
    is checked empirically on the test set regardless.
    """

    def __init__(self, levels=DEFAULT_LEVELS):
        self.levels = tuple(levels)
        self.quantile_shift_ = {}
        self._abs_resid_sorted = None
        self.n_cal_ = 0

    @staticmethod
    def _conformal_quantile(values, level):
        n = len(values)
        rank = int(np.ceil((n + 1) * level))
        rank = min(max(rank, 1), n)
        return float(np.sort(values)[rank - 1])

    def fit(self, log_true_cal, log_pred_cal):
        resid = np.asarray(log_true_cal, dtype=np.float64) - np.asarray(log_pred_cal, dtype=np.float64)
        self.n_cal_ = len(resid)
        if self.n_cal_ == 0:
            raise ValueError("ResidualQuantileCalibrator.fit received an empty calibration set.")
        for level in self.levels:
            self.quantile_shift_[level] = self._conformal_quantile(resid, level)
        self._abs_resid_sorted = np.sort(np.abs(resid))
        return self

    def symmetric_halfwidth(self, alpha):
        """Half-width q_hat s.t. [pred-q_hat, pred+q_hat] has >= 1-alpha
        marginal coverage (split-conformal guarantee)."""
        return self._conformal_quantile(self._abs_resid_sorted, 1 - alpha)

    def predict_quantiles(self, log_pred):
        log_pred = np.asarray(log_pred, dtype=np.float64)
        return {level: log_pred + shift for level, shift in self.quantile_shift_.items()}


def gaussian_quantiles(mu_log, sigma_log, levels=DEFAULT_LEVELS):
    mu_log = np.asarray(mu_log, dtype=np.float64)
    sigma_log = np.asarray(sigma_log, dtype=np.float64)
    return {level: mu_log + norm.ppf(level) * sigma_log for level in levels}


# ---------------------------------------------------------------------
# 5. DISTRIBUTIONAL SCORES SHARED ACROSS ALL MODELS
# ---------------------------------------------------------------------

def pinball_loss(y_true_log, y_pred_quantile, level):
    diff = y_true_log - y_pred_quantile
    return float(np.mean(np.maximum(level * diff, (level - 1) * diff)))


def avg_pinball_crps_proxy(y_true_log, quantile_dict):
    """Trapezoidal CRPS proxy over the available quantile range.

    CRPS is twice the integral of pinball loss across quantile levels. The
    supplied distribution-free grid covers 5--95%, so this deliberately
    reports a shared truncated proxy rather than claiming an exact CRPS.
    """
    levels = np.asarray(sorted(quantile_dict), dtype=np.float64)
    losses = np.asarray(
        [pinball_loss(y_true_log, quantile_dict[level], level) for level in levels],
        dtype=np.float64,
    )
    trapezoidal_integral = np.sum((losses[:-1] + losses[1:]) * np.diff(levels) / 2.0)
    return float(2.0 * trapezoidal_integral)


def coverage_and_sharpness(y_true_log, quantile_dict, level_pairs=((0.05, 0.95), (0.10, 0.90), (0.25, 0.75))):
    out = {}
    for lo, hi in level_pairs:
        if lo not in quantile_dict or hi not in quantile_dict:
            continue
        lower, upper = quantile_dict[lo], quantile_dict[hi]
        covered = (y_true_log >= lower) & (y_true_log <= upper)
        nominal = int(round((hi - lo) * 100))
        out[f"coverage_{nominal}"] = float(np.mean(covered))
        out[f"sharpness_{nominal}_log"] = float(np.mean(upper - lower))
    return out


def full_distributional_report(y_true_sec, quantile_dict_log, mu_log=None, sigma2_log=None):
    """One-stop report used identically for every model in 06 and 07.
    Pass mu_log / sigma2_log only for models with a native Gaussian (GAT)."""
    log_true = to_log_seconds(y_true_sec)
    report = {"CRPS_proxy": avg_pinball_crps_proxy(log_true, quantile_dict_log)}
    report.update(coverage_and_sharpness(log_true, quantile_dict_log))
    if mu_log is not None and sigma2_log is not None:
        report["NLL"] = gaussian_nll(log_true, mu_log, sigma2_log)
    else:
        report["NLL"] = np.nan
    return report


# ---------------------------------------------------------------------
# 6. CLOSED-FORM GAUSSIAN DISTRIBUTIONAL METRICS (space-agnostic)
# ---------------------------------------------------------------------
# Used wherever a model produces a genuine (mean, variance) pair in some
# additive space -- e.g. the empirical-Bayes conjugate posteriors in
# _07_bayesian_update.py, which are Gaussian in SECONDS for Level 1
# (additive) and Gaussian in LOG-SECONDS for Level 2/3 (proportional /
# feature-weighted). These helpers are deliberately unit-agnostic: pass
# whatever (y_true, mu, sigma2) triple lives in the model's own update
# space and the result is a proper (not proxy) closed-form CRPS/NLL/
# coverage for a Gaussian predictive distribution in that space.

Z_90 = float(norm.ppf(0.95))  # two-sided 90% interval half-width, in units of sigma


def gaussian_crps(y_true, mu, sigma2):
    """Exact closed-form CRPS for a Gaussian predictive distribution
    (Gneiting & Raftery, 2007), evaluated pointwise and averaged. Unlike
    the pinball-loss proxy above, this is the true CRPS whenever the
    predictive distribution genuinely is Gaussian in the supplied space.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.sqrt(np.clip(np.asarray(sigma2, dtype=np.float64), EPS, None))
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def gaussian_interval_coverage_width(y_true, mu, sigma2, z=Z_90):
    """Symmetric +/- z*sigma interval coverage & mean width, in whatever
    (additive) space mu/sigma2 live. Default z corresponds to a nominal
    90% two-sided Gaussian interval."""
    y_true = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.sqrt(np.clip(np.asarray(sigma2, dtype=np.float64), EPS, None))
    lower, upper = mu - z * sigma, mu + z * sigma
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    width = float(np.mean(upper - lower))
    return coverage, width


def gaussian_distributional_report(y_true, mu, sigma2):
    """Space-agnostic bundle of {CRPS, NLL, Coverage_90, Width_90} for a
    Gaussian predictive distribution with mean `mu` and variance `sigma2`,
    both expressed in the same additive space as `y_true` (seconds for
    Level 1, log-seconds for the Prior / Level 2 / Level 3 updates)."""
    coverage_90, width_90 = gaussian_interval_coverage_width(y_true, mu, sigma2)
    return {
        "CRPS": gaussian_crps(y_true, mu, sigma2),
        "NLL": gaussian_nll(y_true, mu, sigma2),
        "Coverage_90": coverage_90,
        "Width_90": width_90,
    }
