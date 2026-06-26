"""Stage 2 evaluation with pycocotools (comparable to the Ultralytics B0 numbers).

Runs the trained GNN head over the cached candidates, decodes refined detections,
un-letterboxes to ORIGINAL pixels, then computes COCO mAP@[.5:.95], mAP@.5,
per-class AP, and small-object mAP using OUR data-driven area threshold (1920px^2).
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn.dataset import decode_box_residual  # noqa: E402
from gnn.graph import build_image_graph  # noqa: E402


def _unletterbox(boxes, gain, padw, padh):
    b = boxes.clone()
    b[:, [0, 2]] = (b[:, [0, 2]] - padw) / gain
    b[:, [1, 3]] = (b[:, [1, 3]] - padh) / gain
    return b


# memoize loaded cache + prebuilt graph per file (graph is model-independent, so
# it is constant across the periodic evals within a run).
_PREP = {}


def _prep(f, graph_cfg, geom_dim):
    if f in _PREP:
        return _PREP[f]
    d = torch.load(f, weights_only=False)
    nf = d["node_feat"].float()
    boxes = d["boxes_xyxy"].float()
    if nf.size(0):
        ei, ew = build_image_graph(
            nf[:, :2], nf[:, geom_dim:], boxes_xyxy=boxes, k=graph_cfg["k"],
            spatial_metric=graph_cfg["spatial_metric"],
            use_feature_sim=graph_cfg["use_feature_sim"],
            sim_thresh=graph_cfg["sim_thresh"], edge_weight=graph_cfg["edge_weight"])
    else:
        ei = torch.empty(2, 0, dtype=torch.long); ew = torch.empty(0)
    out = {"nf": nf, "boxes": boxes, "ei": ei, "ew": ew,
           "gt_boxes": d["gt_boxes"].float(), "gt_labels": d["gt_labels"],
           "gain": d["gain"], "padw": d["padw"], "padh": d["padh"],
           "ow": d["ow"], "oh": d["oh"]}
    _PREP[f] = out
    return out


@torch.no_grad()
def run_predictions(model, cache_dir, graph_cfg, num_classes, device,
                    score_thresh=0.05, nms_iou=0.6, geom_dim=4, max_det=300):
    """Return COCO-style (gt_dict, dt_list) in original-image pixels."""
    model.eval()
    files = sorted(Path(cache_dir).glob("*.pt"))
    images, gt_anns, dts = [], [], []
    ann_id = 1
    for img_id, f in enumerate(files, 1):
        p = _prep(f, graph_cfg, geom_dim)
        ow, oh = p["ow"], p["oh"]
        images.append({"id": img_id, "width": ow, "height": oh})

        # GT in original px
        gtb = _unletterbox(p["gt_boxes"], p["gain"], p["padw"], p["padh"])
        for b, c in zip(gtb, p["gt_labels"].tolist()):
            x0, y0, x1, y1 = b.tolist()
            w, h = max(0, x1 - x0), max(0, y1 - y0)
            gt_anns.append({"id": ann_id, "image_id": img_id, "category_id": c + 1,
                            "bbox": [x0, y0, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1

        nf = p["nf"].to(device)
        boxes = p["boxes"]
        n = nf.size(0)
        if n == 0:
            continue
        yolo_scores = nf[:, geom_dim:geom_dim + num_classes]
        out = model.forward_with_prior(nf, p["ei"].to(device), p["ew"].to(device), yolo_scores)
        probs = F.softmax(out["cls_logits"], dim=1)            # (n, C+1)
        quality = torch.sigmoid(out["conf_delta"])             # (n,)
        fg = probs[:, :num_classes] * quality.unsqueeze(1)     # foreground scores
        score, cls = fg.max(dim=1)

        bx = boxes.to(device)
        ref = decode_box_residual(bx, out["box_residual"]) if "box_residual" in out else bx
        ref = _unletterbox(ref.cpu(), p["gain"], p["padw"], p["padh"])

        keep = score.cpu() >= score_thresh
        if keep.sum() == 0:
            continue
        rb, rs, rc = ref[keep], score.cpu()[keep], cls.cpu()[keep]
        nms_keep = batched_nms(rb, rs, rc, nms_iou)[:max_det]
        for b, s, c in zip(rb[nms_keep], rs[nms_keep], rc[nms_keep]):
            x0, y0, x1, y1 = b.tolist()
            dts.append({"image_id": img_id, "category_id": int(c) + 1,
                        "bbox": [x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
                        "score": float(s)})

    categories = [{"id": i + 1, "name": str(i)} for i in range(num_classes)]
    gt_dict = {"images": images, "annotations": gt_anns, "categories": categories}
    return gt_dict, dts


def coco_metrics(gt_dict, dts, num_classes, class_names, small_area=1920.0):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        if not dts:
            return {"mAP50_95": 0.0, "mAP50": 0.0, "mAP_small": 0.0,
                    "per_class_mAP50_95": {n: 0.0 for n in class_names}}
        coco_dt = coco_gt.loadRes(dts)
        E = COCOeval(coco_gt, coco_dt, "bbox")
        # custom area ranges: all / small(<1920) / large  (COCO default uses 32^2..)
        E.params.areaRng = [[0, 1e10], [0, small_area], [small_area, 1e10], [small_area, 1e10]]
        E.params.areaRngLbl = ["all", "small", "large", "xl"]
        E.params.maxDets = [10, 100, 300]
        E.evaluate(); E.accumulate(); E.summarize()

    # precision: [T, R, K, A, M]
    prec = E.eval["precision"]
    def ap(area_idx):
        p = prec[:, :, :, area_idx, -1]
        p = p[p > -1]
        return float(p.mean()) if p.size else 0.0
    def ap50(area_idx=0):
        p = prec[0, :, :, area_idx, -1]   # T=0 -> IoU 0.5
        p = p[p > -1]
        return float(p.mean()) if p.size else 0.0

    per_class = {}
    for k, name in enumerate(class_names):
        p = prec[:, :, k, 0, -1]
        p = p[p > -1]
        per_class[name] = float(p.mean()) if p.size else 0.0

    return {"mAP50_95": ap(0), "mAP50": ap50(0), "mAP_small": ap(1),
            "per_class_mAP50_95": per_class}


def evaluate(model, cache_dir, graph_cfg, num_classes, class_names, device,
             score_thresh=0.01, nms_iou=0.6, small_area=1920.0):
    gt_dict, dts = run_predictions(model, cache_dir, graph_cfg, num_classes,
                                   device, score_thresh, nms_iou)
    return coco_metrics(gt_dict, dts, num_classes, class_names, small_area)
