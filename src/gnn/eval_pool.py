"""Reference baseline: the SAME top-N candidate pool the GNN sees, but with plain
per-class NMS and the raw YOLO scores (no GNN refinement). This is the
apples-to-apples control for "does the GNN add value over naive NMS on the same
candidates?" — distinct from full B0 (which uses YOLO's complete pre-NMS set).

    python src/gnn/eval_pool.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torchvision.ops import batched_nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import REPO_ROOT, load_config  # noqa: E402
from gnn.eval import _unletterbox, coco_metrics  # noqa: E402


def main(nms_iou=0.6, score_thresh=0.05, tag="baseline"):
    dcfg = load_config()
    names = dcfg["fine_names"]; C = len(names)
    small = dcfg["small_object"]["area_px_max"]
    cache = Path(dcfg["paths"]["out_root"]) / "cand_cache/test"

    images, gt, dts = [], [], []; aid = 1
    for iid, f in enumerate(sorted(cache.glob("*.pt")), 1):
        d = torch.load(f, weights_only=False); ow, oh = d["ow"], d["oh"]
        images.append({"id": iid, "width": ow, "height": oh})
        gtb = _unletterbox(d["gt_boxes"].float(), d["gain"], d["padw"], d["padh"])
        for b, c in zip(gtb, d["gt_labels"].tolist()):
            x0, y0, x1, y1 = b.tolist()
            gt.append({"id": aid, "image_id": iid, "category_id": c + 1,
                       "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                       "iscrowd": 0}); aid += 1
        boxes = d["boxes_xyxy"].float(); s, cl = d["scores"].float().max(1)
        ref = _unletterbox(boxes, d["gain"], d["padw"], d["padh"])
        keep = s >= score_thresh
        if keep.sum() == 0:
            continue
        rb, rs, rc = ref[keep], s[keep], cl[keep]
        nk = batched_nms(rb, rs, rc, nms_iou)[:300]
        for b, sc, c in zip(rb[nk], rs[nk], rc[nk]):
            x0, y0, x1, y1 = b.tolist()
            dts.append({"image_id": iid, "category_id": int(c) + 1,
                        "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(sc)})

    gt_dict = {"images": images, "annotations": gt,
               "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    m = coco_metrics(gt_dict, dts, C, names, small_area=small)
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "stage": "stage2_ref", "variant": "Cand+NMS", "eval_method": "pycoco",
           "step": tag,
           "label_set": "fine", "seed": 0,
           "mAP50_95": round(m["mAP50_95"], 4), "mAP50": round(m["mAP50"], 4),
           "mAP_small": round(m["mAP_small"], 4),
           "per_class_mAP50_95": {k: round(v, 4) for k, v in m["per_class_mAP50_95"].items()},
           "params": 0}
    with open(REPO_ROOT / "logs/experiments.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"Cand+NMS (pool ceiling) mAP50-95={row['mAP50_95']} mAP50={row['mAP50']} "
          f"mAP_small={row['mAP_small']}")
    print("  per-class:", row["per_class_mAP50_95"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--nms_iou", type=float, default=0.6)
    ap.add_argument("--score_thresh", type=float, default=0.05)
    a = ap.parse_args()
    main(nms_iou=a.nms_iou, score_thresh=a.score_thresh, tag=a.tag)
