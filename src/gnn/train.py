"""Stage 2 — train one GNN refinement variant on the frozen-YOLO candidate cache.

Only the GNN head trains (YOLO is frozen and already baked into the cache). Each
experiment-matrix row is one (block_a, block_b) choice. Final test metrics are
computed with pycocotools (same evaluator used for B0) and logged.

    python src/gnn/train.py --block_a gcn --block_b gat --seed 0
    python src/gnn/train.py --block_a none --block_b gat        # GAT only
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader
from torchvision.ops import complete_box_iou_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import REPO_ROOT, load_config  # noqa: E402
from gnn.dataset import CandidateGraphDataset, compute_class_weights, decode_box_residual  # noqa: E402
from gnn.eval import evaluate  # noqa: E402
from gnn.models import GNNHead, build_from_config  # noqa: E402


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def focal_ce(logits, y, weight, gamma=2.0):
    ce = F.cross_entropy(logits, y, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def none_if_str(v):
    return None if (v is None or str(v).lower() == "none") else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gnn.yaml")
    ap.add_argument("--block_a", default=None)
    ap.add_argument("--block_b", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    gcfg = yaml.safe_load(open(REPO_ROOT / args.config))
    dcfg = load_config()
    mcfg, tcfg, grcfg = dict(gcfg["model"]), gcfg["train"], gcfg["graph"]
    if args.block_a is not None:
        mcfg["block_a"] = none_if_str(args.block_a)
    if args.block_b is not None:
        mcfg["block_b"] = none_if_str(args.block_b)
    mcfg["block_a"] = none_if_str(mcfg["block_a"])
    mcfg["block_b"] = none_if_str(mcfg["block_b"])
    variant = GNNHead.variant_name(mcfg["block_a"], mcfg["block_b"])
    epochs = args.epochs or tcfg["epochs"]
    device = args.device or (f"cuda:{tcfg['device']}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    C = mcfg["num_classes"]
    names = dcfg["fine_names"]
    cache = Path(dcfg["paths"]["out_root"]) / "cand_cache"
    small_area = dcfg["small_object"]["area_px_max"]

    train_ds = CandidateGraphDataset(cache / "train", grcfg, C, tcfg["pos_iou"])
    in_dim = train_ds[0].x.size(1)
    model = build_from_config(mcfg, in_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    cls_w, counts = compute_class_weights(train_ds, C, device) if tcfg["use_class_weights"] \
        else (None, None)
    loader = DataLoader(train_ds, batch_size=tcfg["batch_images"], shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"[{variant} seed{args.seed}] in_dim={in_dim} params={n_params:,} "
          f"device={device} epochs={epochs}")
    if counts is not None:
        print("  node class counts (incl bg):",
              {(*names, 'bg')[i]: int(counts[i]) for i in range(C + 1)})

    best_map, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for batch in loader:
            batch = batch.to(device)
            yolo_scores = batch.x[:, 4:4 + C]                  # geom(4) then C scores
            out = model.forward_with_prior(batch.x, batch.edge_index, batch.edge_attr, yolo_scores)
            loss = focal_ce(out["cls_logits"], batch.y, cls_w, tcfg["focal_gamma"])
            # quality (IoU-aware) loss
            loss = loss + tcfg["conf_loss_weight"] * F.binary_cross_entropy_with_logits(
                out["conf_delta"], batch.quality)
            # box CIoU on positives
            pos = batch.reg_mask
            if mcfg["predict_box_residual"] and pos.any():
                pred_box = decode_box_residual(batch.cand_boxes[pos], out["box_residual"][pos])
                tgt_box = decode_box_residual(batch.cand_boxes[pos], batch.reg_target[pos])
                loss = loss + tcfg["box_loss_weight"] * \
                    complete_box_iou_loss(pred_box, tgt_box, reduction="mean")
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        sched.step()

        if ep % 5 == 0 or ep == epochs:
            m = evaluate(model, cache / "val", grcfg, C, names, device,
                         nms_iou=0.6, small_area=small_area)
            tag = "*" if m["mAP50_95"] > best_map else " "
            if m["mAP50_95"] > best_map:
                best_map = m["mAP50_95"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ep{ep:3d} loss={tot/len(loader):.3f} "
                  f"val mAP50-95={m['mAP50_95']*100:.2f} mAP50={m['mAP50']*100:.2f} {tag}")
    train_min = (time.time() - t0) / 60

    # ---- final TEST eval with best val checkpoint ----
    if best_state:
        model.load_state_dict(best_state)
    # FPS: time the GNN forward over test graphs
    test_m = evaluate(model, cache / "test", grcfg, C, names, device,
                      nms_iou=0.6, small_area=small_area)

    # save weights
    outdir = REPO_ROOT / gcfg["output"]["project"] / f"{variant.replace('->','_')}_seed{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "best.pt")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": "stage2_gnn", "variant": variant, "eval_method": "pycoco",
        "label_set": "fine", "seed": args.seed, "epochs": epochs,
        "block_a": mcfg["block_a"], "block_b": mcfg["block_b"],
        "k": grcfg["k"], "topk": gcfg["candidates"]["topk"],
        "mAP50_95": round(test_m["mAP50_95"], 4), "mAP50": round(test_m["mAP50"], 4),
        "mAP_small": round(test_m["mAP_small"], 4),
        "per_class_mAP50_95": {k: round(v, 4) for k, v in test_m["per_class_mAP50_95"].items()},
        "params": n_params, "best_val_mAP50_95": round(best_map, 4),
        "train_minutes": round(train_min, 1), "run_dir": str(outdir),
    }
    log = REPO_ROOT / gcfg["output"]["exp_log"]
    with open(log, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"\n[{variant} seed{args.seed}] TEST mAP50-95={row['mAP50_95']} "
          f"mAP50={row['mAP50']} mAP_small={row['mAP_small']} params={n_params:,}")
    print("  per-class:", row["per_class_mAP50_95"])
    print(f"  logged -> {gcfg['output']['exp_log']}")


if __name__ == "__main__":
    main()
