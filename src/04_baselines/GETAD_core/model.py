"""
GETAD_core/model.py
Core model architectures and loss functions for GETAD
(Mbuya et al., SIGSPATIAL '25).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

PAD_TOKEN = 0
SOT_TOKEN = 1
EOT_TOKEN = 2
NUM_SPECIAL_TOKENS = 3


class GETAD(nn.Module):
    """
    Autoregressive Transformer-Decoder for Trajectory Anomaly Detection.
    - Captures road topology via GAT.
    - Captures multi-hop structural distance via Graph Positional Embeddings (GPE).
    - Incorporates empirical transition frequencies in GAT attention.
    - Prevents future token leakage via strictly autoregressive causal masking.
    """
    def __init__(self, num_nodes, real_node_features, config):
        super(GETAD, self).__init__()
        self.num_nodes = num_nodes
        self.d_model = config["d_model"]
        self.d_max = config["d_max"]
        self.vocab_size = num_nodes + NUM_SPECIAL_TOKENS
        self.edge_dim = config.get("edge_dim", 1)

        # 4.1 Graph-Based Road Segment Embedding (GAT)
        real_dim = real_node_features.shape[1]
        self.real_feats = nn.Parameter(real_node_features, requires_grad=False)

        learnable_dim = max(0, config["node_in_dim"] - real_dim)
        if learnable_dim > 0:
            self.learnable_feats = nn.Parameter(torch.randn(num_nodes, learnable_dim))
        else:
            self.learnable_feats = None

        total_in_dim = real_dim + learnable_dim
        self.gat1 = GATConv(
            total_in_dim,
            config["node_emb_dim"],
            heads=2,
            concat=True,
            edge_dim=self.edge_dim,
        )
        self.gat2 = GATConv(
            config["node_emb_dim"] * 2,
            self.d_model,
            heads=1,
            concat=False,
            edge_dim=self.edge_dim,
        )

        # Special token embeddings: PAD, SOT, EOT
        self.special_tokens_embed = nn.Embedding(NUM_SPECIAL_TOKENS, self.d_model)

        # 4.2 Graph Positional Embedding (GPE)
        # Distances: [0, 1, ..., d_max, d_max + 1] -> total (d_max + 2) entries
        self.gpe_embed = nn.Embedding(self.d_max + 2, self.d_model)
        self.gpe_scale = nn.Parameter(torch.tensor(0.1))

        # 4.3 Trajectory Modeling with Transformer Decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config["n_heads"],
            dim_feedforward=self.d_model * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config["n_layers"])

        # Sequence Output Head: predicts next road segment token from vocabulary
        self.fc_out = nn.Linear(self.d_model, self.vocab_size)

        # 4.4 Supervised Link Prediction Head
        self.link_predictor = nn.Sequential(
            nn.Linear(self.d_model * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def get_node_embeddings(self, edge_index, edge_attr=None):
        if self.learnable_feats is not None:
            combined_features = torch.cat([self.real_feats, self.learnable_feats], dim=1)
        else:
            combined_features = self.real_feats

        x = F.elu(self.gat1(combined_features, edge_index, edge_attr=edge_attr))
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        return x

    def forward(self, tokens, gpe_matrix, edge_index, mask, edge_attr=None):
        # 1. Compute node embeddings for all graph segments
        node_embs = self.get_node_embeddings(edge_index, edge_attr=edge_attr)

        # Full vocabulary embedding table: [PAD, SOT, EOT, Segment_0, Segment_1, ...]
        full_vocab_embs = torch.cat([self.special_tokens_embed.weight, node_embs], dim=0)

        # 2. Extract sequence representations
        seq_embs = full_vocab_embs[tokens]

        # 3. Compute Graph Positional Embeddings (GPE)
        gpe_matrix = torch.clamp(gpe_matrix, min=0, max=self.d_max + 1)
        p_ij = self.gpe_embed(gpe_matrix)
        # Mean pooling prevents GPE from dominating token embeddings on long trajectories
        ge_i = p_ij.mean(dim=2)

        # Combined trajectory token representation: z_i = h_{v_i} + \gamma * ge_{v_i}
        z = seq_embs + self.gpe_scale * ge_i

        # 4. Strictly Autoregressive Transformer with Matching Mask Types
        seq_len = tokens.size(1)
        # Boolean causal mask (True where masked/prohibited)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=z.device),
            diagonal=1,
        )
        # Boolean key padding mask (True where token is padding and ignored)
        key_padding_mask = ~mask.bool()

        transformer_out = self.transformer(
            z,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        logits = self.fc_out(transformer_out)
        return logits, node_embs


def link_prediction_loss(link_predictor, node_embs, pos_edge_index, max_samples=2048):
    """
    Subsampled link prediction loss to constrain road segment topology.
    Avoids OOM by subsampling edges per batch instead of computing over the entire map.
    Gradients flow directly back through node_embs into the GAT layers.
    """
    num_total_pos = pos_edge_index.size(1)
    num_samples = min(max_samples, num_total_pos)

    perm = torch.randperm(num_total_pos, device=node_embs.device)[:num_samples]
    pos_u = pos_edge_index[0, perm]
    pos_v = pos_edge_index[1, perm]

    neg_u = torch.randint(0, node_embs.size(0), (num_samples,), device=node_embs.device)
    neg_v = torch.randint(0, node_embs.size(0), (num_samples,), device=node_embs.device)

    pos_pairs = torch.cat([node_embs[pos_u], node_embs[pos_v]], dim=-1)
    pos_logits = link_predictor(pos_pairs).squeeze(-1)

    neg_pairs = torch.cat([node_embs[neg_u], node_embs[neg_v]], dim=-1)
    neg_logits = link_predictor(neg_pairs).squeeze(-1)

    labels = torch.cat([
        torch.ones(num_samples, device=node_embs.device),
        torch.zeros(num_samples, device=node_embs.device),
    ])
    logits = torch.cat([pos_logits, neg_logits])

    return F.binary_cross_entropy_with_logits(logits, labels)
