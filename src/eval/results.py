"""Aggregate logs/experiments.jsonl into the results table + ablation plots.

Regenerates everything from the log alone (no retraining). Each experiment row
is one (variant, seed) run; we report mean +/- std across seeds per variant, the
way the experiment matrix expects (deltas are small, single runs unconvincing).

    python src/eval/results.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import REPO_ROOT, load_config  # noqa: E402

# canonical order for the matrix (rows missing from the log are simply skipped)
VARIANT_ORDER = [
    "B0", "Cand+NMS", "GAT", "GCN", "GIN", "SAGE",
    "GAT->GAT", "GCN->GAT", "GIN->GAT", "SAGE->GAT",
]


def load_rows(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"no experiment log at {log_path}")
    return [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (None, None)
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return (m, s)


def fmt(m, s, scale=100, nd=2):
    if m is None:
        return "-"
    return f"{m*scale:.{nd}f}±{s*scale:.{nd}f}" if scale != 1 else f"{m:.{nd}f}±{s:.{nd}f}"


def aggregate(rows, label_set="fine", eval_method="pycoco"):
    # Only compare rows scored by the SAME evaluator (pycocotools), so B0 and
    # the GNN variants are apples-to-apples (incl. our small-object threshold).
    # dedupe by (variant, seed): if a variant+seed was run more than once, keep
    # the LAST logged row (later runs supersede earlier ones).
    by_vs = {}
    for r in rows:
        if r.get("label_set", "fine") != label_set:
            continue
        if eval_method and r.get("eval_method") != eval_method:
            continue
        by_vs[(r["variant"], r.get("seed"))] = r
    by_variant = defaultdict(list)
    for (v, _s), r in by_vs.items():
        by_variant[v].append(r)
    variants = [v for v in VARIANT_ORDER if v in by_variant] + \
               [v for v in by_variant if v not in VARIANT_ORDER]

    classes = []
    for r in rows:
        for c in r.get("per_class_mAP50_95", {}):
            if c not in classes:
                classes.append(c)

    agg = {}
    for v in variants:
        rs = by_variant[v]
        d = {
            "n_seeds": len(rs),
            "mAP50_95": mean_std([r["mAP50_95"] for r in rs]),
            "mAP50": mean_std([r["mAP50"] for r in rs]),
            "precision": mean_std([r.get("precision") for r in rs]),
            "recall": mean_std([r.get("recall") for r in rs]),
            "params": rs[0].get("params"),
            "gflops": rs[0].get("gflops"),
            "fps": mean_std([r.get("fps", r.get("fps_eval")) for r in rs]),
            "per_class": {c: mean_std([r.get("per_class_mAP50_95", {}).get(c) for r in rs])
                          for c in classes},
        }
        if rs[0].get("mAP_small") is not None:
            d["mAP_small"] = mean_std([r.get("mAP_small") for r in rs])
        agg[v] = d
    return agg, variants, classes


def write_table(agg, variants, classes, out_md, out_csv):
    has_small = any("mAP_small" in agg[v] for v in variants)
    head = ["variant", "seeds", "mAP50-95", "mAP50", "P", "R"]
    if has_small:
        head.append("mAP_small")
    head += [f"AP:{c}" for c in classes] + ["params(M)", "GFLOPs", "FPS"]

    rows_md = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    rows_csv = [",".join(head)]
    for v in variants:
        d = agg[v]
        cells = [v, str(d["n_seeds"]), fmt(*d["mAP50_95"]), fmt(*d["mAP50"]),
                 fmt(*d["precision"]), fmt(*d["recall"])]
        if has_small:
            cells.append(fmt(*d.get("mAP_small", (None, None))))
        cells += [fmt(*d["per_class"][c]) for c in classes]
        pm = f"{d['params']/1e6:.2f}" if d["params"] else "-"
        cells += [pm, str(d["gflops"] or "-"), fmt(*d["fps"], scale=1, nd=1)]
        rows_md.append("| " + " | ".join(cells) + " |")
        rows_csv.append(",".join(c.replace("±", "+-") for c in cells))

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("# Experiment results (mean±std over seeds)\n\n" + "\n".join(rows_md) + "\n")
    out_csv.write_text("\n".join(rows_csv) + "\n")


def plot_overall(agg, variants, outpath):
    m = [agg[v]["mAP50_95"][0] * 100 for v in variants]
    e = [agg[v]["mAP50_95"][1] * 100 for v in variants]
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(variants)), 5))
    ax.bar(variants, m, yerr=e, capsize=4, color="#4477aa")
    ax.set_ylabel("mAP@[.5:.95] (%)")
    ax.set_title("Overall detection accuracy by variant (mean±std)")
    ax.tick_params(axis="x", rotation=30)
    for i, (mm, ee) in enumerate(zip(m, e)):
        ax.text(i, mm + ee, f"{mm:.1f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(outpath, dpi=120); plt.close(fig)


def plot_per_class(agg, variants, classes, outpath):
    fig, ax = plt.subplots(figsize=(max(9, 1.5 * len(classes)), 5))
    x = np.arange(len(classes)); w = 0.8 / max(1, len(variants))
    for j, v in enumerate(variants):
        vals = [agg[v]["per_class"][c][0] * 100 for c in classes]
        ax.bar(x + j * w, vals, w, label=v)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(classes, rotation=20)
    ax.set_ylabel("AP@[.5:.95] (%)"); ax.set_title("Per-class AP by variant")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(outpath, dpi=120); plt.close(fig)


def main(label_set="fine"):
    cfg = load_config()
    log_path = REPO_ROOT / "logs/experiments.jsonl"
    outdir = Path(cfg["paths"]["reports"]).parent / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(log_path)
    agg, variants, classes = aggregate(rows, label_set=label_set)
    write_table(agg, variants, classes,
                outdir / f"results_{label_set}.md", outdir / f"results_{label_set}.csv")
    plot_overall(agg, variants, outdir / f"overall_mAP_{label_set}.png")
    plot_per_class(agg, variants, classes, outdir / f"per_class_AP_{label_set}.png")

    print(f"variants: {variants}")
    for v in variants:
        d = agg[v]
        print(f"  {v:10s} seeds={d['n_seeds']} "
              f"mAP50-95={fmt(*d['mAP50_95'])} mAP50={fmt(*d['mAP50'])} "
              f"params={d['params']/1e6:.2f}M FPS={fmt(*d['fps'],scale=1,nd=1)}")
    print(f"wrote table + plots -> {outdir}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
