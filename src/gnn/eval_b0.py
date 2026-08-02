"""Evaluate the frozen B0 YOLO with the SAME pycocotools evaluator used for the
GNN variants, so the results table compares like with like (incl. our 1920px^2
small-object threshold). Uses authentic YOLO post-NMS predictions.

    python src/gnn/eval_b0.py --ckpt runs/stage1_yolo/b0_yolo11s_seed0/weights/best.pt --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import REPO_ROOT, load_config  # noqa: E402
from gnn.eval import coco_metrics  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    from ultralytics import YOLO
    dcfg = load_config()
    names = dcfg["fine_names"]
    C = len(names)
    small_area = dcfg["small_object"]["area_px_max"]
    out_root = Path(dcfg["paths"]["out_root"])
    splits = json.load(open(out_root / "splits.json"))
    test_stems = splits["test"]

    model = YOLO(str(REPO_ROOT / args.ckpt))

    # GT from cache (original px already handled by eval.run_predictions for GNN;
    # here build GT directly from label files un-normalized to original px).
    images, gt_anns = [], []
    stem2id = {}
    ann_id = 1
    for img_id, stem in enumerate(test_stems, 1):
        stem2id[stem] = img_id
        # original size from the cached file (fast) — all are 2046x1080 anyway
        d = torch.load(out_root / "cand_cache/test" / f"{stem}.pt", weights_only=False)
        ow, oh = d["ow"], d["oh"]
        images.append({"id": img_id, "width": ow, "height": oh})
        for line in (out_root / "labels_fine" / f"{stem}.txt").read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            c = int(c); cx, cy, w, h = float(cx) * ow, float(cy) * oh, float(w) * ow, float(h) * oh
            gt_anns.append({"id": ann_id, "image_id": img_id, "category_id": c + 1,
                            "bbox": [cx - w / 2, cy - h / 2, w, h], "area": w * h, "iscrowd": 0})
            ann_id += 1

    dts = []
    t0 = time.time()
    # per-image predict + cache clearing: robust on an 8GB GPU (conf=0.001 keeps
    # many boxes per dense image; batched predict can OOM).
    for i, stem in enumerate(test_stems):
        img_id = stem2id[stem]
        r = model.predict(str(out_root / "images" / f"{stem}.jpg"), imgsz=args.imgsz,
                          conf=0.001, iou=0.7, max_det=300, device=args.device,
                          verbose=False)[0]
        b = r.boxes
        if b is not None and b.shape[0]:
            xyxy = b.xyxy.cpu(); conf = b.conf.cpu(); cls = b.cls.cpu().int()
            for box, s, c in zip(xyxy, conf, cls):
                x0, y0, x1, y1 = box.tolist()
                dts.append({"image_id": img_id, "category_id": int(c) + 1,
                            "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(s)})
        del r
        if i % 25 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    fps = round(len(test_stems) / (time.time() - t0), 1)

    gt_dict = {"images": images, "annotations": gt_anns,
               "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    m = coco_metrics(gt_dict, dts, C, names, small_area=small_area)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": "stage1_yolo", "variant": "B0", "eval_method": "pycoco",
        "step": args.tag,
        "label_set": "fine", "seed": args.seed,
        "mAP50_95": round(m["mAP50_95"], 4), "mAP50": round(m["mAP50"], 4),
        "mAP_small": round(m["mAP_small"], 4),
        "per_class_mAP50_95": {k: round(v, 4) for k, v in m["per_class_mAP50_95"].items()},
        "params": sum(p.numel() for p in model.model.parameters()),
        "fps_eval": fps, "ckpt": args.ckpt,
    }
    with open(REPO_ROOT / "logs/experiments.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"B0 seed{args.seed} (pycoco) mAP50-95={row['mAP50_95']} mAP50={row['mAP50']} "
          f"mAP_small={row['mAP_small']}")
    print("  per-class:", row["per_class_mAP50_95"])


if __name__ == "__main__":
    main()
