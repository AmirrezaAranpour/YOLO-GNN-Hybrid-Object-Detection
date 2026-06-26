"""Stage 2 candidate extraction from a frozen YOLO11.

For each image we run the frozen detector once and keep the top-N PRE-NMS
candidates (low conf threshold, so weak/borderline boxes survive — the GNN's job
is to rescue them). Per candidate we store:
  - box geometry (xyxy + normalized cx,cy,w,h) in letterboxed 1024 space
  - the 6 class scores + max confidence
  - an ROI-pooled appearance vector from the P3 neck map (optional)
and per image we store the GT boxes/labels mapped into the SAME letterbox space,
plus the letterbox gain/pad and original size so we can un-letterbox at eval.

Everything stays in letterbox space so candidates and GT are directly comparable
for IoU matching; we only un-letterbox for final mAP reporting.

Caches go to data/acre_yolo/cand_cache/<split>/<stem>.pt so GNN training never
re-runs YOLO.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import roi_align

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import REPO_ROOT, load_config  # noqa: E402


def letterbox(img: Image.Image, new=1024):
    """Resize+pad to (new,new) preserving aspect. Returns (chw float tensor in
    [0,1], gain, padw, padh)."""
    w, h = img.size
    gain = min(new / w, new / h)
    nw, nh = round(w * gain), round(h * gain)
    padw, padh = (new - nw) / 2, (new - nh) / 2
    img_r = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new, new), (114, 114, 114))
    canvas.paste(img_r, (int(round(padw - 0.1)), int(round(padh - 0.1))))
    x = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float().div(255)
    return x, gain, padw, padh


def gt_to_letterbox(boxes_norm_xywh, labels, ow, oh, gain, padw, padh):
    """GT (normalized cx,cy,w,h in original image) -> xyxy in letterbox px."""
    if len(boxes_norm_xywh) == 0:
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
    b = torch.tensor(boxes_norm_xywh, dtype=torch.float32)
    cx, cy, bw, bh = b[:, 0] * ow, b[:, 1] * oh, b[:, 2] * ow, b[:, 3] * oh
    x0 = (cx - bw / 2) * gain + padw
    y0 = (cy - bh / 2) * gain + padh
    x1 = (cx + bw / 2) * gain + padw
    y1 = (cy + bh / 2) * gain + padh
    return torch.stack([x0, y0, x1, y1], 1), torch.tensor(labels, dtype=torch.long)


class CandidateExtractor:
    def __init__(self, ckpt, imgsz=1024, device="cuda", roi_feat=True):
        from ultralytics import YOLO
        self.yolo = YOLO(str(REPO_ROOT / ckpt))
        self.model = self.yolo.model.to(device).eval()
        self.imgsz = imgsz
        self.device = device
        self.roi_feat = roi_feat
        self.nc = self.model.nc if hasattr(self.model, "nc") else len(self.model.names)
        # hook the Detect head to grab its input neck feature maps [P3,P4,P5]
        self._neck = {}
        detect = self.model.model[-1]
        detect.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        x = args[0]
        self._neck["maps"] = x  # list of 3 tensors

    @torch.no_grad()
    def extract(self, img_path, topk=150, conf_thresh=0.05):
        img = Image.open(img_path).convert("RGB")
        ow, oh = img.size
        x, gain, padw, padh = letterbox(img, self.imgsz)
        x = x.unsqueeze(0).to(self.device)

        out = self.model(x)
        preds = out[0] if isinstance(out, (list, tuple)) else out   # (1, 4+nc, N)
        preds = preds[0].transpose(0, 1)                            # (N, 4+nc)
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4:4 + self.nc]                           # already sigmoided
        conf, _ = scores.max(dim=1)

        keep = conf >= conf_thresh
        idx_all = torch.arange(preds.size(0), device=self.device)[keep]
        conf_k = conf[keep]
        if conf_k.numel() > topk:
            top = conf_k.topk(topk).indices
            sel = idx_all[top]
        else:
            sel = idx_all

        bxywh = boxes_xywh[sel]
        x0 = bxywh[:, 0] - bxywh[:, 2] / 2
        y0 = bxywh[:, 1] - bxywh[:, 3] / 2
        x1 = bxywh[:, 0] + bxywh[:, 2] / 2
        y1 = bxywh[:, 1] + bxywh[:, 3] / 2
        boxes_xyxy = torch.stack([x0, y0, x1, y1], 1).clamp(0, self.imgsz)
        sscores = scores[sel]
        sconf = conf[sel]

        # node features: geom(4 norm) + scores(nc) + conf(1) [+ roi appearance]
        geom = torch.stack([
            (x0 + x1) / 2 / self.imgsz, (y0 + y1) / 2 / self.imgsz,
            (x1 - x0) / self.imgsz, (y1 - y0) / self.imgsz], 1)
        feats = [geom, sscores, sconf.unsqueeze(1)]
        if self.roi_feat and "maps" in self._neck:
            p3 = self._neck["maps"][0]                            # (1,C,H,W) stride 8
            rois = torch.cat([torch.zeros(boxes_xyxy.size(0), 1, device=self.device),
                              boxes_xyxy], 1)
            roi = roi_align(p3, rois, output_size=1, spatial_scale=1.0 / 8, aligned=True)
            feats.append(roi.flatten(1))                          # (M, C)
        node_feat = torch.cat(feats, 1).cpu()

        return {
            "boxes_xyxy": boxes_xyxy.cpu(), "scores": sscores.cpu(),
            "conf": sconf.cpu(), "node_feat": node_feat,
            "gain": gain, "padw": padw, "padh": padh, "ow": ow, "oh": oh,
        }


def precompute(cfg_path="configs/gnn.yaml"):
    gcfg = __import__("yaml").safe_load(open(REPO_ROOT / cfg_path))
    dcfg = load_config()
    out_root = Path(dcfg["paths"]["out_root"])
    splits = json.load(open(out_root / "splits.json"))

    ext = CandidateExtractor(gcfg["yolo_ckpt"], imgsz=gcfg["imgsz"],
                             roi_feat=gcfg["candidates"]["roi_feat"])
    cache_root = out_root / "cand_cache"

    # GT per stem from label files (normalized cx,cy,w,h)
    def load_gt(stem):
        p = out_root / "labels_fine" / f"{stem}.txt"
        bs, ls = [], []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            ls.append(int(c)); bs.append([float(cx), float(cy), float(w), float(h)])
        return bs, ls

    n = 0
    for split, stems in splits.items():
        outdir = cache_root / split
        outdir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            d = ext.extract(out_root / "images" / f"{stem}.jpg",
                            topk=gcfg["candidates"]["topk"],
                            conf_thresh=gcfg["candidates"]["conf_thresh"])
            bs, ls = load_gt(stem)
            gtb, gtl = gt_to_letterbox(bs, ls, d["ow"], d["oh"],
                                       d["gain"], d["padw"], d["padh"])
            d["gt_boxes"] = gtb; d["gt_labels"] = gtl; d["stem"] = stem
            torch.save(d, outdir / f"{stem}.pt")
            n += 1
            if n % 100 == 0:
                print(f"  cached {n} images... (last {split}/{stem}: "
                      f"{d['node_feat'].shape[0]} cands, dim {d['node_feat'].shape[1]})")
    print(f"done: cached {n} images -> {cache_root}")
    print(f"node feature dim = {d['node_feat'].shape[1]}")


if __name__ == "__main__":
    precompute(*(sys.argv[1:2] or []))
