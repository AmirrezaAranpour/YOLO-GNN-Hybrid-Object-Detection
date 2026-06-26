"""Stage 2 dataset: cached candidates -> per-image graphs with node targets.

Each cached image becomes one graph. Nodes = candidate boxes. Targets per node
via IoU matching to GT (in shared letterbox space):
  IoU >= pos_iou  -> class = GT species (0..C-1), box-regression target, quality=IoU
  else            -> background class (= C), no regression

Node feature layout (139-d): [cx,cy,w,h (norm)] + [C class scores] + [conf] + [ROI P3].
Feature-similarity edges use the appearance part (scores+conf+ROI), not geometry
(geometry is already captured by the spatial k-NN edges).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn.graph import build_image_graph  # noqa: E402


def encode_box_residual(cand_xyxy, gt_xyxy):
    """(dx,dy,dw,dh) regressing cand -> gt, standard log-space encoding."""
    pw = (cand_xyxy[:, 2] - cand_xyxy[:, 0]).clamp(min=1e-3)
    ph = (cand_xyxy[:, 3] - cand_xyxy[:, 1]).clamp(min=1e-3)
    px = (cand_xyxy[:, 0] + cand_xyxy[:, 2]) / 2
    py = (cand_xyxy[:, 1] + cand_xyxy[:, 3]) / 2
    gw = (gt_xyxy[:, 2] - gt_xyxy[:, 0]).clamp(min=1e-3)
    gh = (gt_xyxy[:, 3] - gt_xyxy[:, 1]).clamp(min=1e-3)
    gx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2
    gy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2
    return torch.stack([(gx - px) / pw, (gy - py) / ph,
                        torch.log(gw / pw), torch.log(gh / ph)], 1)


def decode_box_residual(cand_xyxy, res):
    pw = (cand_xyxy[:, 2] - cand_xyxy[:, 0]).clamp(min=1e-3)
    ph = (cand_xyxy[:, 3] - cand_xyxy[:, 1]).clamp(min=1e-3)
    px = (cand_xyxy[:, 0] + cand_xyxy[:, 2]) / 2
    py = (cand_xyxy[:, 1] + cand_xyxy[:, 3]) / 2
    nx = res[:, 0] * pw + px
    ny = res[:, 1] * ph + py
    nw = torch.exp(res[:, 2].clamp(max=4)) * pw
    nh = torch.exp(res[:, 3].clamp(max=4)) * ph
    return torch.stack([nx - nw / 2, ny - nh / 2, nx + nw / 2, ny + nh / 2], 1)


class CandidateGraphDataset(Dataset):
    def __init__(self, cache_dir, graph_cfg, num_classes, pos_iou=0.5, geom_dim=4):
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        assert self.files, f"no cache in {cache_dir}"
        self.g = graph_cfg
        self.C = num_classes
        self.pos_iou = pos_iou
        self.geom_dim = geom_dim
        # Preload + build all graphs ONCE (≈100MB); avoids per-epoch disk loads.
        self.data = [self._build(f) for f in self.files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _build(self, f):
        d = torch.load(f, weights_only=False)
        nf = d["node_feat"].float()
        boxes = d["boxes_xyxy"].float()
        n = nf.size(0)
        centers = nf[:, :2]
        app = nf[:, self.geom_dim:]                       # appearance for sim edges

        ei, ew = build_image_graph(
            centers, app, boxes_xyxy=boxes, k=self.g["k"],
            spatial_metric=self.g["spatial_metric"],
            use_feature_sim=self.g["use_feature_sim"],
            sim_thresh=self.g["sim_thresh"], edge_weight=self.g["edge_weight"])

        gtb, gtl = d["gt_boxes"].float(), d["gt_labels"].long()
        y = torch.full((n,), self.C, dtype=torch.long)    # default background
        reg = torch.zeros(n, 4)
        mask = torch.zeros(n, dtype=torch.bool)
        quality = torch.zeros(n)
        gt_assigned = boxes.clone()
        if gtb.numel() and n:
            iou = box_iou(boxes, gtb)                      # (n, G)
            best_iou, best_g = iou.max(1)
            pos = best_iou >= self.pos_iou
            y[pos] = gtl[best_g[pos]]
            mask = pos
            quality = best_iou.clamp(0, 1)
            if pos.any():
                gassign = gtb[best_g[pos]]
                reg[pos] = encode_box_residual(boxes[pos], gassign)
                gt_assigned[pos] = gassign

        data = Data(x=nf, edge_index=ei, edge_attr=ew, y=y)
        data.reg_target = reg
        data.reg_mask = mask
        data.quality = quality
        data.cand_boxes = boxes
        data.num_nodes = n
        return data


def compute_class_weights(dataset, num_classes, device="cpu"):
    """Inverse-sqrt-frequency weights over C species + background (index C)."""
    counts = torch.zeros(num_classes + 1)
    for i in range(len(dataset)):
        y = dataset[i].y
        counts += torch.bincount(y, minlength=num_classes + 1).float()
    counts = counts.clamp(min=1)
    w = counts.sum() / counts
    w = w / w[num_classes]                                 # normalize so bg weight = 1
    w = w.sqrt()                                           # soften
    return w.to(device), counts
