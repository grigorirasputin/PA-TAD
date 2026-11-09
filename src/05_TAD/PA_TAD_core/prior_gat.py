import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

BASE_DIR = Path(__file__).resolve().parents[3]


class BayesianTrafficGAT(nn.Module):
    """
    Topological Graph Attention Network predicting expected travel time
    residuals and log-variance on road network nodes/edges.
    Incorporates the 3-part decomposition from 05_train_prior.py:
        mu_final = log_base_time + mu_adj
    """
    def __init__(self, in_channels, time_channels=6, hidden_dim=64, num_heads=2, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.time_channels = time_channels
        self.hidden_dim = hidden_dim

        self.gat1 = GATConv(in_channels, hidden_dim, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        combined_dim = hidden_dim + time_channels
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(32, 1)
        self.log_var_head = nn.Linear(32, 1)

        # Global harmonic delay ridge prior buffers (fitted during prior training)
        self.register_buffer("global_time_weight", torch.zeros(1, time_channels, dtype=torch.float32))
        self.register_buffer("global_time_bias", torch.zeros(1, dtype=torch.float32))

    def encode(self, x, edge_index):
        h = F.leaky_relu(self.gat1(x, edge_index))
        h = self.dropout(h)
        h = F.leaky_relu(self.gat2(h, edge_index))
        return h

    def compute_global_delay(self, time_features):
        return F.linear(time_features, self.global_time_weight, self.global_time_bias).squeeze(-1)

    def predict(self, h_nodes, target_nodes, time_features, log_base_time, global_log_delay=None):
        q_embs = h_nodes[target_nodes]
        fused = torch.cat([q_embs, time_features], dim=-1)
        hid = self.fusion_mlp(fused)

        mu_adj = torch.clamp(self.mu_head(hid).squeeze(-1), min=-2.0, max=4.0)
        log_var = torch.clamp(self.log_var_head(hid).squeeze(-1), min=-4.0, max=4.0)

        if global_log_delay is None:
            global_log_delay = self.compute_global_delay(time_features)

        mu_final = log_base_time[target_nodes] + global_log_delay + mu_adj
        return mu_final, log_var

    def forward(self, x, edge_index, target_nodes, time_features, log_base_time, global_log_delay=None):
        h_nodes = self.encode(x, edge_index)
        return self.predict(h_nodes, target_nodes, time_features, log_base_time, global_log_delay)


def gaussian_nll_loss(mu, log_var, target):
    var = torch.exp(log_var)
    loss = 0.5 * np.log(2 * np.pi) + 0.5 * log_var + ((target - mu) ** 2) / (2.0 * var + 1e-6)
    return loss.mean()


def _extract_time_features(t, time_channels=5):
    """
    Robust extractor for continuous time features from a trip item in TrajectoryDataset.
    Includes 'time_feats' (used by TrajectoryDataset) alongside legacy key aliases.
    """
    for key in ("time_feats", "time_feat", "time_features", "time"):
        if key in t and t[key] is not None:
            feat = t[key]
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat).float()
            elif not isinstance(feat, torch.Tensor):
                feat = torch.tensor(feat, dtype=torch.float32)
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            if feat.shape[-1] == time_channels:
                return feat.float()
            elif feat.shape[-1] > time_channels:
                return feat[:, :time_channels].float()
            else:
                pad = torch.zeros((feat.shape[0], time_channels - feat.shape[-1]), dtype=feat.dtype)
                return torch.cat([feat, pad], dim=-1).float()

    seq_len = len(t["nodes"])
    candidates = ["hours_sin", "hours_cos", "day_sin", "day_cos", "norm_t", "norm_time", "week_sin", "week_cos"]
    found_cols = []
    for c in candidates:
        if c in t and t[c] is not None:
            v = t[c]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            if v.dim() == 0:
                v = v.repeat(seq_len)
            found_cols.append(v.view(seq_len, 1).float())

    if found_cols:
        cat_feat = torch.cat(found_cols, dim=-1)
        if cat_feat.shape[-1] >= time_channels:
            return cat_feat[:, :time_channels]
        else:
            pad = torch.zeros((seq_len, time_channels - cat_feat.shape[-1]), dtype=torch.float32)
            return torch.cat([cat_feat, pad], dim=-1)

    return torch.zeros((seq_len, time_channels), dtype=torch.float32)


def _collect_normal_observations(ds, time_channels=5, max_samples=500_000, seed=42):
    """
    Extracts normal (non-anomalous) segment traversals from a TrajectoryDataset
    so the Bayesian Traffic GAT Prior is not contaminated by synthetic anomalies.
    """
    if ds is None or len(ds.trips) == 0:
        return None, None, None

    normal_nodes, normal_secs, normal_time = [], [], []
    all_nodes, all_secs, all_time = [], [], []

    for t in ds.trips:
        nodes = t["nodes"].long() if isinstance(t["nodes"], torch.Tensor) else torch.tensor(t["nodes"], dtype=torch.long)
        secs = t["seconds"].float() if isinstance(t["seconds"], torch.Tensor) else torch.tensor(t["seconds"], dtype=torch.float32)
        if secs.dim() > 1:
            secs = secs.squeeze(-1)
        t_feat = _extract_time_features(t, time_channels=time_channels)

        all_nodes.append(nodes)
        all_secs.append(secs)
        all_time.append(t_feat)

        s_lbl = t.get("s_lbl", None)
        t_lbl = t.get("t_lbl", None)
        if s_lbl is not None and t_lbl is not None:
            s_t = s_lbl if isinstance(s_lbl, torch.Tensor) else torch.tensor(s_lbl)
            t_t = t_lbl if isinstance(t_lbl, torch.Tensor) else torch.tensor(t_lbl)
            norm_mask = (s_t == 0) & (t_t == 0)
            if norm_mask.any():
                normal_nodes.append(nodes[norm_mask])
                normal_secs.append(secs[norm_mask])
                normal_time.append(t_feat[norm_mask])
        else:
            normal_nodes.append(nodes)
            normal_secs.append(secs)
            normal_time.append(t_feat)

    use_nodes = normal_nodes if len(normal_nodes) > 0 else all_nodes
    use_secs = normal_secs if len(normal_secs) > 0 else all_secs
    use_time = normal_time if len(normal_time) > 0 else all_time

    p_nodes = torch.cat(use_nodes)
    p_secs = torch.cat(use_secs)
    p_time = torch.cat(use_time)

    n_total = len(p_nodes)
    if max_samples is not None and n_total > max_samples:
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(n_total, generator=g)[:max_samples]
        p_nodes = p_nodes[perm]
        p_secs = p_secs[perm]
        p_time = p_time[perm]

    return p_nodes, p_secs, p_time


def _fit_global_harmonic_prior(p_nodes, p_secs, p_time, log_base_time, ridge=1e-3):
    """
    Fits a ridge-regressed global temporal delay curve (analogous to
    fit_global_weekly_prior in 05_train_prior.py) on training observations.
    """
    with torch.no_grad():
        device = log_base_time.device
        nodes_dev = p_nodes.to(device)
        secs_dev = p_secs.to(device)
        time_dev = p_time.to(device)

        residual_target = secs_dev - log_base_time[nodes_dev]
        ones = torch.ones((time_dev.shape[0], 1), device=device, dtype=torch.float32)
        design = torch.cat([ones, time_dev], dim=-1)  # [N, 1 + time_channels]

        reg = torch.eye(design.shape[1], device=device, dtype=torch.float32) * ridge
        reg[0, 0] = 0.0
        lhs = design.T @ design + reg
        rhs = design.T @ residual_target.unsqueeze(-1)
        weights = torch.linalg.solve(lhs, rhs).squeeze(-1)

        bias = weights[0:1]
        weight = weights[1:].unsqueeze(0)
        return weight, bias


def train_or_load_gat_prior(
    train_ds,
    real_features,
    edge_index,
    log_base_time,
    checkpoint_path: Path,
    device: torch.device,
    val_ds=None,
    logger=None,
    time_channels: int = 6,
    hidden_dim: int = 64,
    epochs: int = 30,
    patience: int = 7,
    batch_size: int = 32768,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    force_retrain: bool = False,
    max_samples: int = 500_000,
    tag: str = "",
):
    """
    Trains or loads a Zero-Leakage Bayesian Traffic GAT Prior on the training split.
    Uses detached once-per-epoch graph encoding and global harmonic delay conditioning
    from 05_train_prior.py, plus normal-traversal filtering and validation early stopping.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    gat_prior = BayesianTrafficGAT(
        in_channels=real_features.shape[1],
        time_channels=time_channels,
        hidden_dim=hidden_dim,
        num_heads=2,
        dropout=0.1,
    ).to(device)

    if checkpoint_path.exists() and not force_retrain:
        if logger:
            logger.info(f"Loading existing GAT Prior weights from {checkpoint_path.name}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        gat_prior.load_state_dict(state_dict, strict=False)
        gat_prior.eval()
        return gat_prior

    if logger:
        logger.info("Pre-training Zero-Leakage Bayesian GAT Prior across normal training traversals...")

    p_nodes, p_secs, p_time = _collect_normal_observations(
        train_ds, time_channels=time_channels, max_samples=max_samples, seed=42
    )
    if p_nodes is None or len(p_nodes) == 0:
        if logger:
            logger.warning("No sample nodes found in train_ds! Returning initialized GAT Prior.")
        gat_prior.eval()
        return gat_prior

    # Prepare validation observations for early stopping
    v_nodes, v_secs, v_time = _collect_normal_observations(
        val_ds, time_channels=time_channels, max_samples=max_samples // 4 if max_samples else None, seed=43
    )
    if v_nodes is None or len(v_nodes) == 0:
        # Hold out 10% of training traversals if val_ds was not passed
        n_total = len(p_nodes)
        n_val = max(1, int(0.10 * n_total))
        v_nodes, p_nodes = p_nodes[:n_val], p_nodes[n_val:]
        v_secs, p_secs = p_secs[:n_val], p_secs[n_val:]
        v_time, p_time = p_time[:n_val], p_time[n_val:]

    # Fit global harmonic temporal delay prior on training split
    g_weight, g_bias = _fit_global_harmonic_prior(p_nodes, p_secs, p_time, log_base_time)
    gat_prior.global_time_weight.copy_(g_weight)
    gat_prior.global_time_bias.copy_(g_bias)

    optimizer = torch.optim.Adam(gat_prior.parameters(), lr=lr, weight_decay=weight_decay)
    n_obs = len(p_nodes)
    n_val_obs = len(v_nodes)

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = copy.deepcopy(gat_prior.state_dict())

    for ep in range(1, epochs + 1):
        gat_prior.train()
        optimizer.zero_grad(set_to_none=True)

        # Encode static graph once per epoch (05_train_prior.py optimization)
        h_nodes = gat_prior.encode(real_features, edge_index)
        h_detached = h_nodes.detach().requires_grad_(True)

        ep_loss = 0.0
        n_batches = 0
        perm_ep = torch.randperm(n_obs)

        for i in range(0, n_obs, batch_size):
            idx = perm_ep[i:i + batch_size]
            b_tgt = p_nodes[idx].to(device, non_blocking=True)
            b_sec = p_secs[idx].to(device, non_blocking=True)
            b_tim = p_time[idx].to(device, non_blocking=True)

            mu, log_var = gat_prior.predict(h_detached, b_tgt, b_tim, log_base_time)
            loss = gaussian_nll_loss(mu, log_var, b_sec)
            loss.backward()

            ep_loss += loss.item()
            n_batches += 1

        if h_detached.grad is not None:
            h_nodes.backward(h_detached.grad / max(n_batches, 1))

        torch.nn.utils.clip_grad_norm_(gat_prior.parameters(), max_norm=2.0)
        optimizer.step()
        avg_train_loss = ep_loss / max(n_batches, 1)

        # Validation pass
        gat_prior.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            h_val = gat_prior.encode(real_features, edge_index)
            for i in range(0, n_val_obs, batch_size):
                b_tgt = v_nodes[i:i + batch_size].to(device, non_blocking=True)
                b_sec = v_secs[i:i + batch_size].to(device, non_blocking=True)
                b_tim = v_time[i:i + batch_size].to(device, non_blocking=True)
                v_mu, v_log_var = gat_prior.predict(h_val, b_tgt, b_tim, log_base_time)
                val_loss_total += gaussian_nll_loss(v_mu, v_log_var, b_sec).item()
                val_batches += 1
        avg_val_loss = val_loss_total / max(val_batches, 1)

        if logger and (ep % 5 == 0 or ep == 1 or ep == epochs):
            logger.info(
                f"  [GAT Prior Epoch {ep:02d}/{epochs:02d}] "
                f"Train NLL: {avg_train_loss:.4f} | Val NLL: {avg_val_loss:.4f}"
            )

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_weights = copy.deepcopy(gat_prior.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if logger:
                    logger.info(f"  [GAT Prior] Early stopping triggered at epoch {ep} (Best Val NLL: {best_val_loss:.4f})")
                break

    gat_prior.load_state_dict(best_weights)
    torch.save({"model_state": gat_prior.state_dict(), "best_val_nll": best_val_loss}, checkpoint_path)
    if logger:
        logger.info(f"Saved pre-trained GAT Prior to {checkpoint_path.name} (Best Val NLL: {best_val_loss:.4f})")

    gat_prior.eval()
    return gat_prior