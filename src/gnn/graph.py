"""Per-image graph construction for the GNN refinement stage.

A node = one pre-NMS candidate box. Edges combine spatial proximity (k-NN over
box centers, or IoU) with feature cosine similarity, so the GNN can relate both
boxes that are physically close (a weed at a crop stem) and boxes that look alike
(a cluster of the same species). Everything is config-tunable for sweeps.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce, to_undirected


def _knn_edges(points: torch.Tensor, k: int) -> torch.Tensor:
    """k-NN edge_index (2,E) by Euclidean distance. Plain torch (no pyg-lib),
    fine for the small graphs here (N<=~150 candidates)."""
    n = points.size(0)
    dist = torch.cdist(points, points)
    dist.fill_diagonal_(float("inf"))
    idx = dist.topk(k, dim=1, largest=False).indices       # (n,k) nearest
    src = torch.arange(n, device=points.device).repeat_interleave(k)
    dst = idx.reshape(-1)
    return torch.stack([src, dst], 0)


def _pairwise_iou(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """boxes (N,4) xyxy -> (N,N) IoU."""
    area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0) * \
           (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0)
    lt = torch.max(boxes_xyxy[:, None, :2], boxes_xyxy[None, :, :2])
    rb = torch.min(boxes_xyxy[:, None, 2:], boxes_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area[:, None] + area[None, :] - inter
    return inter / union.clamp(min=1e-6)


def build_image_graph(centers, feats, boxes_xyxy=None, k=12,
                      spatial_metric="center", use_feature_sim=True,
                      sim_thresh=0.5, edge_weight="blend"):
    """Build one image's graph.

    Args:
        centers: (N,2) normalized box centers.
        feats:   (N,D) node features (for cosine similarity edges).
        boxes_xyxy: (N,4) needed if spatial_metric='iou'.
    Returns:
        edge_index (2,E) undirected+coalesced, edge_weight (E,) in [0,1].
    """
    n = centers.size(0)
    device = centers.device
    if n <= 1:
        return torch.empty(2, 0, dtype=torch.long, device=device), \
               torch.empty(0, device=device)

    kk = min(k, n - 1)

    # ---- spatial edges ----
    if spatial_metric == "iou":
        assert boxes_xyxy is not None
        iou = _pairwise_iou(boxes_xyxy)
        iou.fill_diagonal_(-1)
        idx = iou.topk(kk, dim=1).indices                 # top-kk by IoU
        src = torch.arange(n, device=device).repeat_interleave(kk)
        dst = idx.reshape(-1)
        sp_edge = torch.stack([src, dst], 0)
    else:                                                  # center distance k-NN
        sp_edge = _knn_edges(centers, kk)                  # (2,E) directed

    # ---- feature-similarity edges ----
    edges = [sp_edge]
    if use_feature_sim and feats is not None:
        fn = F.normalize(feats, dim=1)
        sim = fn @ fn.t()
        sim.fill_diagonal_(-1)
        mask = sim >= sim_thresh
        si, sj = mask.nonzero(as_tuple=True)
        if si.numel():
            edges.append(torch.stack([si, sj], 0))

    edge_index = to_undirected(torch.cat(edges, dim=1))

    # ---- edge weights ----
    s, d = edge_index
    cd = (centers[s] - centers[d]).pow(2).sum(1).sqrt()
    spatial_close = torch.exp(-cd / (cd.mean() + 1e-6))    # 1 near, ->0 far
    if use_feature_sim and feats is not None:
        fn = F.normalize(feats, dim=1)
        sim = (fn[s] * fn[d]).sum(1).clamp(min=0)
    else:
        sim = torch.zeros_like(spatial_close)

    if edge_weight == "spatial":
        w = spatial_close
    elif edge_weight == "similarity":
        w = sim
    else:                                                  # blend
        w = 0.5 * spatial_close + 0.5 * sim

    edge_index, w = coalesce(edge_index, w, num_nodes=n, reduce="mean")
    return edge_index, w
