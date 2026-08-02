"""Diagnostic: GT-recall of the cached candidate pool — the Stage-1 ceiling the
GNN can never exceed. For each cached image we ask: what fraction of GT boxes have
at least one candidate with IoU >= thr? Reported overall, and split by image
density (dense = many GT), since the dense images are where the pool truncation
bites. Run before/after a topk change to see the ceiling move.

    python src/eval/pool_recall.py [split] [iou_thr] [dense_gt_min]
    python src/eval/pool_recall.py test 0.5 100
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision.ops import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import load_config  # noqa: E402


def main(split="test", iou_thr=0.5, dense_gt_min=100):
    iou_thr = float(iou_thr); dense_gt_min = int(dense_gt_min)
    dcfg = load_config()
    cache = Path(dcfg["paths"]["out_root"]) / "cand_cache" / split
    files = sorted(cache.glob("*.pt"))
    assert files, f"no cache in {cache}"

    tot_gt = matched = 0
    dense_gt = dense_matched = 0
    n_cands = []
    for f in files:
        d = torch.load(f, weights_only=False)
        boxes = d["boxes_xyxy"].float()
        gtb = d["gt_boxes"].float()
        n_cands.append(boxes.size(0))
        g = gtb.size(0)
        if g == 0:
            continue
        if boxes.size(0) == 0:
            tot_gt += g
            if g >= dense_gt_min:
                dense_gt += g
            continue
        iou = box_iou(gtb, boxes)                    # (G, N)
        best = iou.max(1).values                     # best candidate per GT
        hit = (best >= iou_thr).sum().item()
        tot_gt += g; matched += hit
        if g >= dense_gt_min:
            dense_gt += g; dense_matched += hit

    n_cands = torch.tensor(n_cands, dtype=torch.float)
    print(f"[{split}] pool GT-recall@IoU{iou_thr:g}  (candidates/img: "
          f"mean={n_cands.mean():.0f} max={int(n_cands.max())})")
    print(f"  overall : {matched}/{tot_gt} = {matched/max(tot_gt,1):.4f}")
    if dense_gt:
        print(f"  dense (>= {dense_gt_min} GT/img): "
              f"{dense_matched}/{dense_gt} = {dense_matched/dense_gt:.4f}")
    else:
        print(f"  dense (>= {dense_gt_min} GT/img): none")
    return matched / max(tot_gt, 1)


if __name__ == "__main__":
    main(*sys.argv[1:])
