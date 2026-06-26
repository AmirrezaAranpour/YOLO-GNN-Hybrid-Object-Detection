"""Swappable GNN refinement head for Stage 2.

The experiment matrix is realized purely by the (block_a, block_b) knobs:

    GNNHead(block_a="gcn"|"gin"|"sage"|None, block_b="gat"|None)

  * block_b="gat", block_a=None        -> "GAT only"
  * block_a="gcn", block_b=None        -> "GCN only" (etc.)
  * block_a="gcn", block_b="gat"       -> proposed hybrid "GCN->GAT"
  * block_a="gat"(via b),block_b="gat" -> "GAT->GAT" control

Pipeline: input MLP -> [Block A structural aggregation] -> [Block B GAT] ->
per-node output head producing (class logits over C+1 incl. background, a
confidence delta, and optional box residuals dx,dy,dw,dh).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


def _gin_mlp(dim):
    return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))


def make_conv(kind: str, dim: int, heads: int = 4):
    """Return a conv mapping dim->dim. GAT uses `heads` then concatenates back to dim."""
    kind = kind.lower()
    if kind == "gcn":
        return GCNConv(dim, dim), "ew"            # supports edge_weight
    if kind == "sage":
        return SAGEConv(dim, dim), "plain"
    if kind == "gin":
        return GINConv(_gin_mlp(dim)), "plain"
    if kind == "gat":
        assert dim % heads == 0, "hidden_dim must be divisible by gat_heads"
        return GATConv(dim, dim // heads, heads=heads, concat=True), "plain"
    raise ValueError(f"unknown conv kind: {kind}")


class GNNBlock(nn.Module):
    """A stack of `n_layers` convs of one `kind`, residual + LayerNorm + dropout."""

    def __init__(self, kind, dim, n_layers, heads=4, dropout=0.2):
        super().__init__()
        self.kind = kind.lower()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.modes = []
        for _ in range(n_layers):
            conv, mode = make_conv(kind, dim, heads)
            self.convs.append(conv)
            self.modes.append(mode)
            self.norms.append(nn.LayerNorm(dim))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv, mode, norm in zip(self.convs, self.modes, self.norms):
            h = conv(x, edge_index, edge_weight) if (mode == "ew" and edge_weight is not None) \
                else conv(x, edge_index)
            h = norm(h)
            x = x + F.relu(h)                      # residual
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GNNHead(nn.Module):
    def __init__(self, in_dim, num_classes, block_a=None, block_b="gat",
                 hidden_dim=256, a_layers=2, b_layers=2, gat_heads=4,
                 dropout=0.2, predict_box_residual=True):
        super().__init__()
        assert block_a or block_b, "need at least one of block_a / block_b"
        self.predict_box_residual = predict_box_residual

        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.block_a = GNNBlock(block_a, hidden_dim, a_layers, gat_heads, dropout) if block_a else None
        self.block_b = GNNBlock(block_b, hidden_dim, b_layers, gat_heads, dropout) if block_b else None

        self.num_classes = num_classes
        self.cls_head = nn.Linear(hidden_dim, num_classes + 1)   # +1 background/reject
        self.conf_head = nn.Linear(hidden_dim, 1)                # confidence delta
        if predict_box_residual:
            self.box_head = nn.Linear(hidden_dim, 4)             # dx, dy, dw, dh

        # Residual-refinement init: zero the output heads so at start the GNN
        # adds NOTHING -> cls logits come purely from the YOLO score prior (added
        # in forward), boxes are unchanged, quality=0.5. The GNN can then only
        # *refine* the detector, never destroy it. bg bias is pushed negative so
        # a candidate is background only when all YOLO foreground scores are low.
        for h in [self.cls_head, self.conf_head] + ([self.box_head] if predict_box_residual else []):
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        self.cls_head.bias.data[num_classes] = -4.0             # background prior

    def forward(self, x, edge_index, edge_weight=None):
        x = self.input_mlp(x)
        if self.block_a is not None:
            x = self.block_a(x, edge_index, edge_weight)
        if self.block_b is not None:
            x = self.block_b(x, edge_index, edge_weight)   # GAT ignores edge_weight
        out = {
            "cls_logits": self.cls_head(x),
            "conf_delta": self.conf_head(x).squeeze(-1),
        }
        if self.predict_box_residual:
            out["box_residual"] = self.box_head(x)
        return out

    def forward_with_prior(self, x, edge_index, edge_weight, yolo_scores):
        """Like forward(), but adds the YOLO class-score logits as a prior to the
        foreground class logits (residual refinement). yolo_scores: (N, C) in [0,1]."""
        out = self.forward(x, edge_index, edge_weight)
        p = yolo_scores.clamp(1e-4, 1 - 1e-4)
        out["cls_logits"][:, :self.num_classes] = \
            out["cls_logits"][:, :self.num_classes] + torch.log(p / (1 - p))
        return out

    @staticmethod
    def variant_name(block_a, block_b):
        a = (block_a or "").upper()
        b = (block_b or "").upper()
        if a and b:
            return f"{a}->{b}"
        return a or b


def build_from_config(cfg_model: dict, in_dim: int) -> GNNHead:
    return GNNHead(
        in_dim=in_dim,
        num_classes=cfg_model["num_classes"],
        block_a=cfg_model.get("block_a"),
        block_b=cfg_model.get("block_b"),
        hidden_dim=cfg_model["hidden_dim"],
        a_layers=cfg_model["a_layers"],
        b_layers=cfg_model["b_layers"],
        gat_heads=cfg_model["gat_heads"],
        dropout=cfg_model["dropout"],
        predict_box_residual=cfg_model["predict_box_residual"],
    )
