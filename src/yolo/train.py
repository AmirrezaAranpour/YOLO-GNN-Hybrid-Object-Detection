"""Stage 1 — train the YOLO11 baseline (B0) and log test metrics.

Thin, config-driven wrapper around Ultralytics. Trains on the fine 6-class set,
evaluates on the held-out TEST split, and appends one row to the unified
experiment log (logs/experiments.jsonl) carrying everything the final results
table needs: mAP, per-class AP, P/R, params, FLOPs, FPS.

Usage:
    python src/yolo/train.py                      # uses configs/yolo.yaml
    python src/yolo/train.py --seed 1
    python src/yolo/train.py --epochs 3 --name smoke   # quick pipeline check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import REPO_ROOT, load_config  # noqa: E402


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yolo.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    cfg_path = (REPO_ROOT / args.config)
    cfg = yaml.safe_load(open(cfg_path))
    tr, aug, out = cfg["train"], cfg["aug"], cfg["output"]

    seed = args.seed if args.seed is not None else tr["seed"]
    epochs = args.epochs if args.epochs is not None else tr["epochs"]
    batch = args.batch if args.batch is not None else tr["batch"]
    imgsz = args.imgsz if args.imgsz is not None else tr["imgsz"]
    data_yaml = str(REPO_ROOT / (args.data or cfg["data"]))
    base_name = args.name or out["run_name"]
    run_name = f"{base_name}_seed{seed}"

    from ultralytics import YOLO
    model = YOLO(cfg["model"])

    t0 = time.time()
    model.train(
        data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
        device=tr["device"], workers=tr["workers"], optimizer=tr["optimizer"],
        cos_lr=tr["cos_lr"], amp=tr["amp"], seed=seed,
        deterministic=tr["deterministic"], cache=tr["cache"], patience=tr["patience"],
        project=str(REPO_ROOT / out["project"]), name=run_name, exist_ok=True,
        **aug,
    )
    train_min = (time.time() - t0) / 60

    # ---- evaluate on the held-out TEST split ----
    metrics = model.val(data=data_yaml, split="test", imgsz=imgsz,
                        batch=batch, device=tr["device"],
                        project=str(REPO_ROOT / out["project"]),
                        name=f"{run_name}_test", exist_ok=True)

    names = model.names  # {id: name}
    maps = metrics.box.maps.tolist()  # per-class mAP50-95
    per_class = {names[i]: round(float(m), 4) for i, m in enumerate(maps)}

    # params / FLOPs
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params
        n_params = int(get_num_params(model.model))
        gflops = round(float(get_flops(model.model, imgsz)), 2)
    except Exception:
        info = model.info(verbose=False)
        n_params = int(info[1]) if info else None
        gflops = float(info[3]) if info and len(info) > 3 else None

    speed = metrics.speed  # ms: preprocess/inference/postprocess
    infer_ms = float(speed.get("inference", 0))
    total_ms = sum(float(v) for v in speed.values())
    fps = round(1000.0 / total_ms, 1) if total_ms else None

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": "stage1_yolo", "variant": "B0", "model": cfg["model"],
        "label_set": "fine" if "fine" in data_yaml else "coarse",
        "seed": seed, "epochs": epochs, "imgsz": imgsz, "batch": batch,
        "split": "test",
        "mAP50_95": round(float(metrics.box.map), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class_mAP50_95": per_class,
        "params": n_params, "gflops": gflops,
        "infer_ms_per_img": round(infer_ms, 2), "fps": fps,
        "train_minutes": round(train_min, 1),
        "run_dir": str(REPO_ROOT / out["project"] / run_name),
    }
    append_jsonl(REPO_ROOT / out["exp_log"], row)

    print("\n" + "=" * 60)
    print(f"B0 seed={seed}  TEST  mAP50-95={row['mAP50_95']}  mAP50={row['mAP50']}")
    print(f"P={row['precision']} R={row['recall']}  params={n_params:,} GFLOPs={gflops} FPS={fps}")
    print("per-class mAP50-95:", per_class)
    print(f"logged -> {out['exp_log']}")


if __name__ == "__main__":
    main()
