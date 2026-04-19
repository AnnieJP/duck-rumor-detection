"""
Temporal GAT for DUCK+.

Two components:
  1. TemporalGATConv  — single GAT layer that accepts a precomputed edge
     scalar bias and adds it (scaled by learnable lambda) to the raw
     attention logit before softmax.  Matches the proposal equation:
         e~_ij = e_ij + lambda * e_feat_ij

  2. BertTemporalGAT  — comment-tree module (replaces bert_gat.SimpleGAT_BERT).
     Encodes parent-child post pairs with BERT, then runs two layers of
     TemporalGATConv with the temporal edge features from preprocessing,
     and mean-pools all nodes to produce z_ct.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from transformers import BertModel


# ---------------------------------------------------------------------------
# Edge-feature MLP: maps 6-dim temporal vector -> scalar bias
# ---------------------------------------------------------------------------

class EdgeMLP(nn.Module):
    def __init__(self, in_dim: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_feat: Tensor) -> Tensor:
        return self.net(edge_feat).squeeze(-1)   # (E,)


# ---------------------------------------------------------------------------
# Custom GAT conv with additive temporal bias on attention logit
# ---------------------------------------------------------------------------

class TemporalGATConv(MessagePassing):
    """
    Single-head GAT layer with an additive scalar bias on the attention logit.

    Standard GAT attention (Velickovic 2018):
        e_ij = LeakyReLU( a^T [W h_i || W h_j] )
    Temporal variant:
        e~_ij = e_ij + lambda * edge_scalar_ij
        alpha_ij = softmax_j( e~_ij )

    Args:
        in_channels   : input node feature dimension
        out_channels  : output node feature dimension per head
        heads         : number of attention heads
        concat        : if True, concatenate head outputs; else average
        dropout       : dropout on attention weights
        edge_feat_dim : dimension of raw edge features (default 6)
        edge_hid_dim  : hidden dim of edge MLP
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        concat: bool = True,
        dropout: float = 0.6,
        edge_feat_dim: int = 6,
        edge_hid_dim: int = 32,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.heads        = heads
        self.concat       = concat
        self.dropout      = dropout

        self.W  = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.a  = nn.Parameter(torch.empty(1, heads, 2 * out_channels))

        self.edge_mlp = EdgeMLP(edge_feat_dim, edge_hid_dim)
        # lambda: learnable scalar, initialised to 0.1 as per proposal
        self.lam = nn.Parameter(torch.tensor(0.1))

        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.bias  = nn.Parameter(torch.zeros(
            heads * out_channels if concat else out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_feat: Tensor,
    ) -> Tensor:
        """
        Args:
            x          : (N, in_channels)
            edge_index : (2, E)  [source, target] in COO format
            edge_feat  : (E, 6)  temporal/structural edge features
        Returns:
            (N, heads*out_channels) if concat else (N, out_channels)
        """
        # Project all nodes: (N, H, F)
        Wh = self.W(x).view(-1, self.heads, self.out_channels)

        # Edge scalar from MLP: (E,)
        e_feat = self.edge_mlp(edge_feat)

        return self.propagate(edge_index, Wh=Wh, e_feat=e_feat)

    def message(self, Wh_i: Tensor, Wh_j: Tensor,
                e_feat: Tensor, index: Tensor, size_i) -> Tensor:
        """
        Wh_i, Wh_j : (E, H, F) — projected features for source/target nodes
        e_feat      : (E,)      — scalar edge bias
        """
        # Attention logit: (E, H)
        alpha = self.leaky(
            (self.a * torch.cat([Wh_i, Wh_j], dim=-1)).sum(dim=-1)
        )
        # Additive temporal bias broadcast over heads
        alpha = alpha + self.lam * e_feat.unsqueeze(-1)

        # Normalise per target node
        alpha = softmax(alpha, index, num_nodes=size_i)   # (E, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted sum: (E, H*F) if concat, else (E, H, F) averaged in update
        weighted = alpha.unsqueeze(-1) * Wh_j   # (E, H, F)
        if self.concat:
            return weighted.view(alpha.size(0), self.heads * self.out_channels)
        else:
            return weighted   # (E, H, F) — update() will mean over H

    def update(self, aggr_out: Tensor) -> Tensor:
        if self.concat:
            # aggr_out is already (N, H*F) from message()
            out = aggr_out
        else:
            # aggr_out is (N, H, F) — average over heads
            out = aggr_out.mean(dim=1)
        return out + self.bias


# ---------------------------------------------------------------------------
# BERT + Temporal GAT comment-tree module
# ---------------------------------------------------------------------------

class BertTemporalGAT(nn.Module):
    """
    Comment-tree module for DUCK+ (replaces SimpleGAT_BERT in bert_gat.py).

    Pipeline per story:
      1. BERT encodes each (parent, child) post pair -> CLS token (768-dim)
      2. Two layers of TemporalGATConv with temporal edge features
      3. Mean-pool all nodes -> z_ct  (shape: [batch, out_feats])

    Args:
        hid_feats    : hidden dimension per GAT head
        out_feats    : output dimension (used as z_ct size)
        dropout      : attention dropout
        edge_hid_dim : hidden dim for edge MLP
        freeze_bert  : if True, freeze BERT weights
    """

    BERT_DIM = 768

    def __init__(
        self,
        hid_feats: int = 64,
        out_feats: int = 64,
        dropout: float = 0.6,
        edge_hid_dim: int = 32,
        freeze_bert: bool = False,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # Layer 1: 768 -> hid_feats * 8 (concat 8 heads)
        self.conv1 = TemporalGATConv(
            in_channels=self.BERT_DIM,
            out_channels=hid_feats,
            heads=8,
            concat=True,
            dropout=dropout,
            edge_hid_dim=edge_hid_dim,
        )
        # Layer 2: hid_feats*8 -> out_feats (average 8 heads)
        self.conv2 = TemporalGATConv(
            in_channels=hid_feats * 8,
            out_channels=out_feats,
            heads=8,
            concat=False,
            dropout=dropout,
            edge_hid_dim=edge_hid_dim,
        )
        self.dropout = dropout

    def forward(self, data) -> Tensor:
        """
        data must have:
            input_ids      : (N_total, seq_len)  — BERT token ids, one per node
            attention_mask : (N_total, seq_len)
            edge_index     : (2, E_total)
            edge_feat      : (E_total, 6)        — temporal features from .npz
            batch          : (N_total,)           — batch assignment
        Returns:
            z_ct : (batch_size, out_feats)
        """
        # BERT encoding for every node
        out = self.bert(
            input_ids=data.input_ids,
            attention_mask=data.attention_mask,
        )
        x = out.last_hidden_state[:, 0, :]   # CLS token: (N_total, 768)

        edge_index = data.edge_index
        edge_feat  = data.edge_feat           # (E_total, 6)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_feat))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_feat)

        # Mean-pool over all nodes per graph (the "all" aggregation from DUCK)
        from torch_scatter import scatter_mean
        z_ct = scatter_mean(x, data.batch, dim=0)
        return z_ct
