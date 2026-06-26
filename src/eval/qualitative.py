"""Qualitative B0 vs GNN-hybrid comparison on dense/occluded test images.

Draws side-by-side panels: YOLO B0 (post-NMS) | GNN-refined | ground truth, for a
handful of the densest test images. Saves to reports/qualitative/.

    python src/eval/qualitative.py --variant GCN->GAT --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from torchvision.ops import batched_nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import REPO_ROOT, load_config  # noqa: E402
from gnn.dataset import decode_box_residual  # noqa: E402
from gnn.eval import _unletterbox  # noqa: E402
from gnn.graph import build_image_graph  # noqa: E402
from gnn.models import GNNHead, build_from_config  # noqa: E402

COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#e377c2"]


def draw(img, boxes, classes, names, title):
    im = img.copy()
    d = ImageDraw.Draw(im)
    for b, c in zip(boxes, classes):
        d.rectangle([float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    outline=COLORS[int(c)], width=3)
    d.rectangle([0, 0, 360, 28], fill="black")
    d.text((6, 8), title, fill="white")
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="GCN->GAT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--score", type=float, default=0.25)
    args = ap.parse_args()

    dcfg = load_config(); gcfg = yaml.safe_load(open(REPO_ROOT / "configs/gnn.yaml"))
    names = dcfg["fine_names"]; C = len(names)
    grcfg = gcfg["graph"]; mcfg = dict(gcfg["model"])
    out_root = Path(dcfg["paths"]["out_root"])
    dev = args.device

    # pick densest test images
    splits = json.load(open(out_root / "splits.json"))
    master = json.load(open(out_root / "annotations_master.json"))
    from collections import Counter
    cnt = Counter(b["stem"] for b in master if b["stem"] in set(splits["test"]))
    picks = [s for s, _ in cnt.most_common(args.n)]

    # B0 model
    from ultralytics import YOLO
    yolo = YOLO(str(REPO_ROOT / gcfg["yolo_ckpt"]))

    # GNN model
    a, b = args.variant.lower().split("->") if "->" in args.variant else (args.variant.lower(), None)
    a = None if a in ("none", "") else a
    mcfg["block_a"], mcfg["block_b"] = a, b
    cache0 = torch.load(out_root / "cand_cache/test" / f"{picks[0]}.pt", weights_only=False)
    in_dim = cache0["node_feat"].size(1)
    gnn = build_from_config(mcfg, in_dim).to(dev).eval()
    ckpt = REPO_ROOT / gcfg["output"]["project"] / f"{args.variant.replace('->','_')}_seed{args.seed}/best.pt"
    gnn.load_state_dict(torch.load(ckpt, map_location=dev))

    outdir = Path(dcfg["paths"]["reports"]).parent / "qualitative"
    outdir.mkdir(parents=True, exist_ok=True)

    for stem in picks:
        img = Image.open(out_root / "images" / f"{stem}.jpg").convert("RGB")
        d = torch.load(out_root / "cand_cache/test" / f"{stem}.pt", weights_only=False)

        # B0 panel
        r = yolo.predict(str(out_root / "images" / f"{stem}.jpg"), imgsz=gcfg["imgsz"],
                         conf=args.score, iou=0.7, max_det=300, device=dev, verbose=False)[0]
        b0_boxes = r.boxes.xyxy.cpu(); b0_cls = r.boxes.cls.cpu().int()
        panel_b0 = draw(img, b0_boxes, b0_cls, names, f"B0 YOLO11s  ({len(b0_boxes)} det)")

        # GNN panel
        nf = d["node_feat"].float().to(dev); boxes = d["boxes_xyxy"].float()
        ei, ew = build_image_graph(nf[:, :2], nf[:, 4:], boxes_xyxy=boxes, k=grcfg["k"],
                                   spatial_metric=grcfg["spatial_metric"],
                                   use_feature_sim=grcfg["use_feature_sim"],
                                   sim_thresh=grcfg["sim_thresh"], edge_weight=grcfg["edge_weight"])
        out = gnn.forward_with_prior(nf, ei.to(dev), ew.to(dev), nf[:, 4:4 + C])
        probs = F.softmax(out["cls_logits"], 1); qual = torch.sigmoid(out["conf_delta"])
        fg = probs[:, :C] * qual.unsqueeze(1); score, cls = fg.max(1)
        ref = decode_box_residual(boxes.to(dev), out["box_residual"]).cpu()
        ref = _unletterbox(ref, d["gain"], d["padw"], d["padh"])
        keep = score.cpu() >= args.score
        rb, rs, rc = ref[keep], score.cpu()[keep], cls.cpu()[keep]
        nk = batched_nms(rb, rs, rc, 0.6)[:300]
        panel_gnn = draw(img, rb[nk], rc[nk], names, f"{args.variant}  ({len(nk)} det)")

        # GT panel
        gtb = _unletterbox(d["gt_boxes"].float(), d["gain"], d["padw"], d["padh"])
        panel_gt = draw(img, gtb, d["gt_labels"].int(), names, f"Ground truth ({len(gtb)})")

        W, H = img.size
        canvas = Image.new("RGB", (W * 3 + 20, H), "white")
        canvas.paste(panel_b0, (0, 0)); canvas.paste(panel_gnn, (W + 10, 0))
        canvas.paste(panel_gt, (2 * W + 20, 0))
        canvas.save(outdir / f"compare_{stem}.jpg", quality=80)
        print(f"  {stem}: B0={len(b0_boxes)} GNN={len(nk)} GT={len(gtb)} -> compare_{stem}.jpg")
    print(f"saved -> {outdir}")


if __name__ == "__main__":
    main()
