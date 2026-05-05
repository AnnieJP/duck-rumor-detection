"""
DUCK+ model: Temporally-Aware Rumour Detection with Adaptive Gated Fusion.

Three branches (matching the proposal exactly):
  1. Comment Tree   — BertTemporalGAT (from temporal_gat.py)
                      Standard GAT variant uses PyG GATConv with no edge bias.
  2. Comment Chain  — TwoTierTransformer: BERT per post, then 2-layer Transformer
                      over the sequence of CLS vectors (DUCK's two-tier design).
  3. User Tree      — GAT over reply/repost user network, nodes init from profile
                      features (GATprf): zero-vectors here since we lack Twitter API.

Fusion variants (controlled by `use_gated_fusion` flag):
  - Concat (DUCK baseline): z = fc( z_ct || z_cc || z_ut )
  - Gated  (DUCK+):         z = g1*z_ct + g2*z_cc + g3*z_ut,
                             where g = sigmoid(W_g [z_ct || z_cc || z_ut])

Model variants from Table 1:
  variant='baseline'  -> standard GAT comment tree + concat fusion
  variant='temp'      -> temporal GAT comment tree + concat fusion
  variant='gated'     -> standard GAT comment tree + gated fusion
  variant='full'      -> temporal GAT comment tree + gated fusion  (DUCK+ full)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean
from transformers import BertModel

from temporal_gat import BertTemporalGAT


# ---------------------------------------------------------------------------
# 1. Comment Tree — standard (no temporal bias, for baseline/gated variants)
# ---------------------------------------------------------------------------

class BertStandardGAT(nn.Module):
    """
    DUCK's original comment-tree: BERT pair encoding + standard GATConv.
    Mean-pools all nodes -> z_ct.
    """

    BERT_DIM = 768

    def __init__(self, hid_feats: int = 64, out_feats: int = 64,
                 dropout: float = 0.6):
        super().__init__()
        self.bert    = BertModel.from_pretrained('bert-base-uncased')
        self.conv1   = GATConv(self.BERT_DIM, hid_feats, heads=8,
                               dropout=dropout, concat=True)
        self.conv2   = GATConv(hid_feats * 8, out_feats, heads=8,
                               dropout=dropout, concat=False)
        self.dropout = dropout

    def forward(self, data) -> Tensor:
        out = self.bert(input_ids=data.input_ids,
                        attention_mask=data.attention_mask)
        x   = out.last_hidden_state[:, 0, :]          # (N_total, 768)
        ei  = data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, ei))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, ei)

        return scatter_mean(x, data.batch, dim=0)      # (B, out_feats)


# ---------------------------------------------------------------------------
# 2. Comment Chain — two-tier transformer
# ---------------------------------------------------------------------------

class TwoTierTransformer(nn.Module):
    """
    DUCK's two-tier comment chain.

    Tier 1: shared BERT encodes each post independently -> CLS (768-dim).
    Tier 2: 2-layer Transformer over the sequence of CLS vectors
            (one vector per post, in chronological order).
    Output: CLS token of tier-2 -> z_cc  (B, 768).

    Input contract (from DuckPlusDataset):
      data.input_ids_seq      : (N_total, MAX_LEN_SEQ)  — one row per node
      data.attention_mask_seq : (N_total, MAX_LEN_SEQ)
      data.batch              : (N_total,)
    Because DataLoader pads to the longest graph in the batch, we reconstruct
    per-graph sequences from the flattened representation.
    """

    BERT_DIM = 768
    MAX_COMMENTS = 120

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Tier-2: small 2-layer transformer, randomly initialised per DUCK
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.BERT_DIM,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True,
        )
        self.tier2 = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Learnable CLS token prepended before tier-2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.BERT_DIM))

    def forward(self, data) -> Tensor:
        # Tier 1: encode every node with BERT
        out = self.bert(
            input_ids=data.input_ids,
            attention_mask=data.attention_mask,
        )
        node_emb = out.last_hidden_state[:, 0, :]  # (N_total, 768)

        # Reconstruct per-graph sequences and pad into a batch tensor
        batch_idx  = data.batch                    # (N_total,)
        batch_size = int(batch_idx.max().item()) + 1
        device     = node_emb.device

        # Group node embeddings by graph, cap at MAX_COMMENTS
        seqs = []
        for g in range(batch_size):
            mask   = batch_idx == g
            embs_g = node_emb[mask][:self.MAX_COMMENTS]  # (n_g, 768)
            seqs.append(embs_g)

        max_len = max(s.size(0) for s in seqs)

        # Pad to (B, max_len, 768) and build key_padding_mask
        padded  = torch.zeros(batch_size, max_len, self.BERT_DIM, device=device)
        pad_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        for g, s in enumerate(seqs):
            n = s.size(0)
            padded[g, :n]   = s
            pad_mask[g, :n] = False   # False = attend, True = ignore

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)          # (B, 1, 768)
        seq = torch.cat([cls, padded], dim=1)                     # (B, 1+L, 768)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool,
                               device=device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1)        # (B, 1+L)

        # Tier 2
        out2  = self.tier2(seq, src_key_padding_mask=full_mask)   # (B, 1+L, 768)
        z_cc  = out2[:, 0, :]                                     # (B, 768) — CLS
        return z_cc


# ---------------------------------------------------------------------------
# 3. User Tree — GAT with profile feature initialisation (GATprf)
# ---------------------------------------------------------------------------

class UserGAT(nn.Module):
    """
    GAT over the reply/repost user network.
    Node features: profile features stored in data.x (shape N, user_feat_dim).
    Since we lack Twitter API access, data.x is zeros (6-dim) from preprocessing.
    Mean-pools all nodes -> z_ut  (B, out_feats).
    """

    def __init__(self, user_feat_dim: int = 6, hid_feats: int = 64,
                 out_feats: int = 64, dropout: float = 0.6):
        super().__init__()
        self.proj    = nn.Linear(user_feat_dim, hid_feats * 8)
        self.conv1   = GATConv(hid_feats * 8, hid_feats, heads=8,
                               dropout=dropout, concat=True)
        self.conv2   = GATConv(hid_feats * 8, out_feats, heads=8,
                               dropout=dropout, concat=False)
        self.dropout = dropout

    def forward(self, data) -> Tensor:
        x  = F.elu(self.proj(data.x))                 # (N_total, hid*8)
        ei = data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, ei))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, ei)

        return scatter_mean(x, data.batch, dim=0)      # (B, out_feats)


# ---------------------------------------------------------------------------
# 4. Adaptive Gated Fusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Replaces concatenation with learned sigmoid scalar gates (proposal §3.3).

    g = sigmoid(W_g [z_ct || z_cc || z_ut] + b_g)  ∈ R^3
    z = g1*z_ct + g2*z_cc + g3*z_ut
    """

    def __init__(self, ct_dim: int, cc_dim: int, ut_dim: int):
        super().__init__()
        self.gate = nn.Linear(ct_dim + cc_dim + ut_dim, 3)

    def forward(self, z_ct: Tensor, z_cc: Tensor, z_ut: Tensor) -> Tensor:
        combined = torch.cat([z_ct, z_cc, z_ut], dim=-1)
        g = torch.sigmoid(self.gate(combined))         # (B, 3)
        z = (g[:, 0:1] * z_ct
           + g[:, 1:2] * z_cc
           + g[:, 2:3] * z_ut)
        return z, g


# ---------------------------------------------------------------------------
# 5. Full DUCK+ model
# ---------------------------------------------------------------------------

class DuckPlus(nn.Module):
    """
    Full DUCK+ model supporting all four Table-1 variants.

    Args:
        variant        : 'baseline' | 'temp' | 'gated' | 'full'
        hid_feats      : GAT hidden dim per head
        ct_out         : comment-tree output dim  -> z_ct size
        ut_out         : user-tree output dim     -> z_ut size
        num_classes    : 4 for Twitter15/16
        dropout_gat    : GAT attention dropout
        dropout_chain  : tier-2 transformer dropout
        user_feat_dim  : user profile feature dim (6 from preprocessing)
        edge_hid_dim   : edge MLP hidden dim (temporal GAT only)
    """

    CC_DIM = 768   # BERT output size, fixed

    def __init__(
        self,
        variant: str   = 'full',
        hid_feats: int = 64,
        ct_out: int    = 64,
        ut_out: int    = 64,
        num_classes: int = 4,
        dropout_gat: float   = 0.6,
        dropout_chain: float = 0.1,
        user_feat_dim: int   = 6,
        edge_hid_dim: int    = 32,
    ):
        super().__init__()
        assert variant in ('baseline', 'temp', 'gated', 'full'), \
            f"Unknown variant '{variant}'"
        self.variant = variant
        self.ct_out  = ct_out
        self.ut_out  = ut_out

        # Comment tree
        use_temporal = variant in ('temp', 'full')
        if use_temporal:
            self.comment_tree = BertTemporalGAT(
                hid_feats=hid_feats, out_feats=ct_out,
                dropout=dropout_gat, edge_hid_dim=edge_hid_dim,
            )
        else:
            self.comment_tree = BertStandardGAT(
                hid_feats=hid_feats, out_feats=ct_out,
                dropout=dropout_gat,
            )

        # Comment chain
        self.comment_chain = TwoTierTransformer(dropout=dropout_chain)

        # User tree
        self.user_tree = UserGAT(
            user_feat_dim=user_feat_dim,
            hid_feats=hid_feats, out_feats=ut_out,
            dropout=dropout_gat,
        )

        # Fusion
        use_gated = variant in ('gated', 'full')
        total_dim = ct_out + self.CC_DIM + ut_out
        if use_gated:
            self.fusion  = GatedFusion(ct_out, self.CC_DIM, ut_out)
            clf_in       = ct_out   # gated outputs same dim as ct_out when equal
            # When dims differ we fuse to the weighted sum; need a common space.
            # Simplest: project each to ct_out before gating.
            self.proj_cc = nn.Linear(self.CC_DIM, ct_out)
            self.proj_ut = nn.Linear(ut_out, ct_out)
            self.fusion  = GatedFusion(ct_out, ct_out, ct_out)
            clf_in       = ct_out
        else:
            self.fusion  = None
            clf_in       = total_dim

        self.classifier = nn.Linear(clf_in, num_classes)

    def forward(self, data):
        z_ct = self.comment_tree(data)           # (B, ct_out)
        z_cc = self.comment_chain(data)          # (B, 768)
        z_ut = self.user_tree(data)              # (B, ut_out)

        if self.fusion is not None:
            z_cc_p = self.proj_cc(z_cc)          # (B, ct_out)
            z_ut_p = self.proj_ut(z_ut)          # (B, ct_out)
            z, gates = self.fusion(z_ct, z_cc_p, z_ut_p)
        else:
            z     = torch.cat([z_ct, z_cc, z_ut], dim=-1)
            gates = None

        logits = self.classifier(z)
        return F.log_softmax(logits, dim=-1), gates
