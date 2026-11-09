import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class PA_TAD_Lite(nn.Module):
    """
    PA-TAD-Lite: the prior-free variant of PA-TAD. Gated ego-MLP + GAT node encoding,
    a Bi-GRU backbone, and disentangled spatial/temporal adapters and heads.
    """
    def __init__(self, num_nodes, real_feats, cfg):
        super().__init__()
        self.real_feats = nn.Parameter(real_feats, requires_grad=False)
        in_dim = real_feats.shape[1]
        emb_dim = cfg["node_emb_dim"]

        # 1. Ego-Node Feature Projection (preserves sharp local segment attributes)
        self.ego_mlp1 = nn.Linear(in_dim, emb_dim * 2)
        self.ego_mlp2 = nn.Linear(emb_dim * 2, emb_dim)

        # 2. Topological GAT Context (captures multi-hop intersection neighborhood)
        self.gat1 = GATConv(in_dim, emb_dim, heads=2, concat=True)
        self.gat2 = GATConv(emb_dim * 2, emb_dim, heads=1, concat=False)
        self.topo_gate = nn.Parameter(torch.full((1, emb_dim), 0.05))

        gru_in_dim = emb_dim + 1 + cfg["time_dim"]
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

        # Disentangled Projection Adapters: Isolate spatial vs temporal representations
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

        # Classification Heads
        self.s_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(adapter_dim, 1),
        )
        self.t_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(adapter_dim, 1),
        )

    def forward(self, nodes, secs, time, edge_idx, lengths):
        ego_embs = self.ego_mlp2(F.elu(self.ego_mlp1(self.real_feats)))
        topo_embs = self.gat2(F.elu(self.gat1(self.real_feats, edge_idx)), edge_idx)
        n_embs = (ego_embs + self.topo_gate * topo_embs)[nodes]

        packed, _ = self.gru(
            nn.utils.rnn.pack_padded_sequence(
                torch.cat([n_embs, secs, time], dim=-1),
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
        )
        out, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=nodes.size(1))
        s_feat = self.spatial_adapter(out)
        t_feat = self.temporal_adapter(out)
        s_logits = self.s_head(s_feat).squeeze(-1)
        t_logits = self.t_head(t_feat).squeeze(-1)
        return s_logits, t_logits


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce
        return loss.mean()


class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable log variances for homoscedastic uncertainty balancing [spatial, temporal]
        self.log_vars = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.focal_s = BinaryFocalLoss(alpha=0.6, gamma=2.0)
        self.focal_t = BinaryFocalLoss(alpha=0.75, gamma=2.0)

    def forward(self, s_log, t_log, s_l, t_l, mask):
        l_s = self.focal_s(s_log[mask], s_l[mask])
        l_t = self.focal_t(t_log[mask], t_l[mask])
        loss = (
            0.5 * torch.exp(-self.log_vars[0]) * l_s + 0.5 * self.log_vars[0]
            + 0.5 * torch.exp(-self.log_vars[1]) * l_t + 0.5 * self.log_vars[1]
        )
        return loss, l_s.item(), l_t.item()