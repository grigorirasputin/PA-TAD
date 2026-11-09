import logging
import random
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch


def setup_logger(region, tag, log_dir=None):
    logger = logging.getLogger(f"PA_TAD_{region}_{tag}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"pa_tad_{region}_{tag}.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



def set_all_seeds(seed: int):
    """Seeds Python, NumPy, and PyTorch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """
    Monitors validation metric (e.g. joint F1) and saves best checkpoint.
    """
    def __init__(self, logger=None, patience=7, delta=0.001, mode="max"):
        self.logger = logger
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_checkpoint_path = None

    def __call__(self, score, model, save_dir, filename, extra_state=None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / filename

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(score, model, ckpt_path, extra_state)
            return True

        improved = (score > self.best_score + self.delta) if self.mode == "max" else (score < self.best_score - self.delta)
        if improved:
            if self.logger:
                self.logger.info(f"Validation metric improved ({self.best_score:.4f} -> {score:.4f}). Saving checkpoint.")
            self.best_score = score
            self._save_checkpoint(score, model, ckpt_path, extra_state)
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.logger:
                self.logger.info(f"EarlyStopping counter: {self.counter}/{self.patience} (best: {self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def _save_checkpoint(self, score, model, ckpt_path, extra_state):
        state = {
            "score": score,
            "state": model.state_dict(),
        }
        if extra_state:
            state.update(extra_state)
        torch.save(state, ckpt_path)
        self.best_checkpoint_path = ckpt_path


def compute_trip_scores(probs, mask):
    """
    Aggregates edge-level anomaly probabilities into one score per trajectory.

    For trajectories with n <= 3 valid edges, the score is the mean edge probability.
    Otherwise it is the mean of the top-k edge probabilities with k = max(2, ceil(n / 4)),
    so a sustained excursion (several high-probability edges) outranks a single spike.
    """
    scores = []
    for p_seq, m_seq in zip(probs, mask):
        valid = p_seq[m_seq]
        n_valid = len(valid)
        if n_valid == 0:
            scores.append(0.0)
        elif n_valid <= 3:
            scores.append(valid.mean().item())
        else:
            # Dynamically select top quarter or at least 2 edges
            k = max(2, int(np.ceil(0.25 * n_valid)))
            k = min(k, n_valid)
            topk_vals, _ = torch.topk(valid, k=k)
            # Weighted average favoring the peak anomaly with sustained support
            mean_topk = topk_vals.mean().item()
            scores.append(mean_topk)
    return np.array(scores, dtype=np.float32)


def compute_binary_metrics(targ, pred, fixed_th=None):
    if len(targ) == 0 or targ.sum() == 0 or targ.sum() == len(targ):
        th = fixed_th if fixed_th is not None else 0.5
        b_pred = (pred >= th).astype(int) if len(pred) > 0 else np.array([], dtype=int)
        return {
            "th": float(th), "roc": 0.0, "pr": 0.0, "acc": 0.0,
            "prec": 0.0, "rec": 0.0, "f1": 0.0, "b_pred": b_pred,
        }
    if fixed_th is None:
        p, r, ths = precision_recall_curve(targ, pred)
        fixed_th = float(ths[np.argmax(2 * p * r / (p + r + 1e-8))]) if len(ths) > 0 else 0.5
    b_pred = (pred >= fixed_th).astype(int)
    return {
        "th": float(fixed_th),
        "roc": float(roc_auc_score(targ, pred)),
        "pr": float(average_precision_score(targ, pred)),
        "acc": float(accuracy_score(targ, b_pred)),
        "prec": float(precision_score(targ, b_pred, zero_division=0)),
        "rec": float(recall_score(targ, b_pred, zero_division=0)),
        "f1": float(f1_score(targ, b_pred, zero_division=0)),
        "b_pred": b_pred,
    }


def evaluate_model(model, loader, edge_idx, device, th_s=None, th_t=None):
    model.eval()
    res = {"s_p": [], "s_t": [], "t_p": [], "t_t": [], "sub": [], "cls": []}
    ordered_tids = []

    with torch.no_grad():
        for b in loader:
            nodes, secs, time, s_l, t_l, sub, mask, lengths, cls_labels, tids = b
            nodes = nodes.to(device)
            secs = secs.to(device)
            time = time.to(device)
            s_l = s_l.to(device)
            t_l = t_l.to(device)
            sub = sub.to(device)
            mask = mask.to(device)
            cls_labels = cls_labels.to(device)

            s_log, t_log = model(nodes, secs, time, edge_idx, lengths)

            s_probs = torch.sigmoid(s_log) * mask
            t_probs = torch.sigmoid(t_log) * mask

            res["s_p"].append(compute_trip_scores(s_probs, mask))
            res["s_t"].append((s_l * mask > 0).float().max(1)[0].cpu().numpy())
            res["t_p"].append(compute_trip_scores(t_probs, mask))
            res["t_t"].append((t_l * mask > 0).float().max(1)[0].cpu().numpy())
            res["sub"].append(sub.max(1)[0].cpu().numpy())
            res["cls"].append(cls_labels.cpu().numpy())
            ordered_tids.extend([str(t) for t in tids])

    if len(res["s_p"]) == 0:
        return {
            "s_roc": 0.0, "s_pr": 0.0, "s_prec": 0.0, "s_rec": 0.0, "s_f1": 0.0,
            "s_roc_pure": 0.0, "s_pr_pure": 0.0, "s_prec_pure": 0.0, "s_rec_pure": 0.0, "s_f1_pure": 0.0,
            "t_roc": 0.0, "t_pr": 0.0, "t_prec": 0.0, "t_rec": 0.0, "t_f1": 0.0,
            "t_roc_pure": 0.0, "t_pr_pure": 0.0, "t_prec_pure": 0.0, "t_rec_pure": 0.0, "t_f1_pure": 0.0,
            "recall_SPRW": 0.0, "recall_DETOUR": 0.0, "recall_OVERSHOOT": 0.0,
            "s_scores": np.array([], dtype=np.float32),
            "t_scores": np.array([], dtype=np.float32),
            "trip_ids": ordered_tids,
        }

    res = {k: np.concatenate(v) for k, v in res.items()}

    sm = compute_binary_metrics(res["s_t"], res["s_p"], th_s)
    tm = compute_binary_metrics(res["t_t"], res["t_p"], th_t)

    pure_spatial_mask = np.isin(res["cls"], [0, 1])
    if pure_spatial_mask.sum() > 0:
        sm_pure = compute_binary_metrics(
            res["s_t"][pure_spatial_mask],
            res["s_p"][pure_spatial_mask],
            fixed_th=sm["th"],
        )
    else:
        sm_pure = sm

    pure_temporal_mask = np.isin(res["cls"], [0, 2])
    if pure_temporal_mask.sum() > 0:
        tm_pure = compute_binary_metrics(
            res["t_t"][pure_temporal_mask],
            res["t_p"][pure_temporal_mask],
            fixed_th=tm["th"],
        )
    else:
        tm_pure = tm

    out = {f"s_{k}": v for k, v in sm.items() if k != "b_pred"} | {f"t_{k}": v for k, v in tm.items() if k != "b_pred"}
    out |= {f"s_{k}_pure": v for k, v in sm_pure.items() if k != "b_pred"}
    out |= {f"t_{k}_pure": v for k, v in tm_pure.items() if k != "b_pred"}

    for v, n in [(1, "SPRW"), (2, "DETOUR"), (3, "OVERSHOOT")]:
        idx = res["sub"] == v
        out[f"recall_{n}"] = float(sm["b_pred"][idx].mean()) if idx.sum() > 0 else 0.0

    out["s_scores"] = res["s_p"]
    out["t_scores"] = res["t_p"]
    out["trip_ids"] = ordered_tids

    return out
