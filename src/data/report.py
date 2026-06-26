"""Task-0 data report: class histograms, bbox-size distribution, the small-object
threshold (derived from the ACTUAL data, not the COCO 32px convention), and
sample images with boxes drawn.

Writes figures + numbers to paths.reports and updates the small_object section
of the config with the chosen threshold.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.config import load_config  # noqa: E402

# distinct colors per fine class (maize,bean,ryegrass,mustard,matricaria,lambsquarter)
CLASS_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#e377c2"]


def percentiles(a, ps):
    return {p: float(np.percentile(a, p)) for p in ps}


def fig_class_hist(master, fine_names, coarse_names, outdir):
    fine = Counter(b["fine_id"] for b in master if b["fine_id"] is not None)
    coarse = Counter(b["coarse_id"] for b in master)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    vals = [fine[i] for i in range(len(fine_names))]
    bars = ax[0].bar(fine_names, vals, color=CLASS_COLORS)
    ax[0].set_yscale("log")
    ax[0].set_title("FINE (6-class) instance counts  [log scale]")
    ax[0].set_ylabel("instances")
    ax[0].tick_params(axis="x", rotation=30)
    for b, v in zip(bars, vals):
        ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    cvals = [coarse[i] for i in range(len(coarse_names))]
    cbars = ax[1].bar(coarse_names, cvals, color=["#2ca02c", "#d62728"])
    ax[1].set_title("COARSE (2-class) instance counts")
    for b, v in zip(cbars, cvals):
        ax[1].text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "class_histogram.png", dpi=120)
    plt.close(fig)


def fig_box_sizes(master, fine_names, outdir):
    areas = np.array([b["area_px"] for b in master], dtype=float)
    side = np.sqrt(areas)                                  # sqrt-area = object "size" in px
    rel = areas / (master[0]["img_w"] * master[0]["img_h"]) * 100  # % of image area
    ar = np.array([b["bw"] / max(b["bh"], 1) for b in master])

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].hist(side, bins=80, color="#4477aa")
    ax[0, 0].set_title("sqrt(bbox area) = object size (px)")
    ax[0, 0].set_xlabel("px"); ax[0, 0].axvline(32, color="r", ls="--", label="COCO small=32px")
    ax[0, 0].legend()
    ax[0, 1].hist(np.log10(areas + 1), bins=80, color="#66ccee")
    ax[0, 1].set_title("log10(bbox area px^2)"); ax[0, 1].set_xlabel("log10 area")
    ax[1, 0].hist(np.clip(rel, 0, 5), bins=80, color="#228833")
    ax[1, 0].set_title("bbox area as % of image (clipped at 5%)"); ax[1, 0].set_xlabel("%")
    ax[1, 1].hist(np.clip(ar, 0, 5), bins=80, color="#ccbb44")
    ax[1, 1].set_title("aspect ratio w/h (clipped at 5)"); ax[1, 1].set_xlabel("w/h")
    fig.tight_layout()
    fig.savefig(outdir / "bbox_size_distribution.png", dpi=120)
    plt.close(fig)

    # per-class size (sqrt area) boxplot
    per_cls = [np.sqrt([b["area_px"] for b in master if b["fine_id"] == i])
               for i in range(len(fine_names))]
    fig, axb = plt.subplots(figsize=(11, 5))
    axb.boxplot(per_cls, tick_labels=fine_names, showfliers=False)
    axb.set_ylabel("object size = sqrt(area) px")
    axb.set_title("Per-class object size distribution (outliers hidden)")
    axb.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(outdir / "bbox_size_per_class.png", dpi=120)
    plt.close(fig)

    return areas, side


def choose_small_threshold(areas, side):
    """Pick a data-driven small-object cutoff and report COCO comparison."""
    ps = percentiles(side, [5, 10, 25, 33, 50, 66, 75, 90, 95])
    area_ps = percentiles(areas, [5, 10, 25, 33, 50, 66, 75, 90, 95])
    coco_small_frac = float((areas < 32 ** 2).mean())     # COCO small = area<1024
    coco_med_frac = float(((areas >= 32 ** 2) & (areas < 96 ** 2)).mean())
    # data-driven choice: lower tercile of object size -> area cutoff
    side_p33 = ps[33]
    area_cut = float(round(side_p33 ** 2))
    return {
        "size_px_percentiles_sqrt_area": ps,
        "area_px_percentiles": area_ps,
        "coco_small_fraction(area<1024)": coco_small_frac,
        "coco_medium_fraction(1024..9216)": coco_med_frac,
        "chosen_rule": "small = area < (33rd pct of sqrt-area)^2",
        "chosen_size_px_sqrt": round(side_p33, 1),
        "chosen_area_px_max": area_cut,
        "fraction_small_under_chosen": float((areas < area_cut).mean()),
    }


def draw_samples(cfg, master, outdir, n=6):
    out_root = Path(cfg["paths"]["out_root"])
    fine_names = cfg["fine_names"]
    by_img = defaultdict(list)
    for b in master:
        by_img[b["stem"]].append(b)

    # choose varied samples: densest, a maize-heavy, a bean-heavy, ones with rare weeds
    def cnt(stem, cls):
        return sum(1 for b in by_img[stem] if b["fine_id"] == cls)
    stems = list(by_img)
    picks = []
    picks.append(max(stems, key=lambda s: len(by_img[s])))               # densest overall
    picks.append(max(stems, key=lambda s: cnt(s, 0)))                    # most maize
    picks.append(max(stems, key=lambda s: cnt(s, 1)))                    # most bean
    picks.append(max(stems, key=lambda s: cnt(s, 3)))                    # most mustard (rare)
    picks.append(max(stems, key=lambda s: cnt(s, 2)))                    # most ryegrass (rare)
    picks.append(max(stems, key=lambda s: cnt(s, 4)))                    # most matricaria (rare)
    seen = set()
    picks = [p for p in picks if not (p in seen or seen.add(p))][:n]

    for stem in picks:
        img = Image.open(out_root / "images" / f"{stem}.jpg").convert("RGB")
        dr = ImageDraw.Draw(img)
        for b in by_img[stem]:
            fid = b["fine_id"]
            col = CLASS_COLORS[fid] if fid is not None else "#888888"
            dr.rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=col, width=3)
        # legend
        present = sorted({b["fine_id"] for b in by_img[stem] if b["fine_id"] is not None})
        for j, fid in enumerate(present):
            dr.rectangle([10, 10 + 26 * j, 34, 30 + 26 * j], fill=CLASS_COLORS[fid])
            dr.text((40, 12 + 26 * j), fine_names[fid], fill="white")
        img.save(outdir / f"sample_{stem}.jpg", quality=85)
    return picks


def main(cfg_path="configs/data.yaml"):
    cfg = load_config(cfg_path)
    out_root = Path(cfg["paths"]["out_root"])
    outdir = Path(cfg["paths"]["reports"])
    outdir.mkdir(parents=True, exist_ok=True)
    fine_names, coarse_names = cfg["fine_names"], cfg["coarse_names"]

    master = json.load(open(out_root / "annotations_master.json"))

    fig_class_hist(master, fine_names, coarse_names, outdir)
    areas, side = fig_box_sizes(master, fine_names, outdir)
    small = choose_small_threshold(areas, side)
    picks = draw_samples(cfg, master, outdir)

    stats = {
        "n_boxes": len(master),
        "fine_imbalance_ratio_max_over_min": None,
        "small_object": small,
        "sample_images": picks,
    }
    fc = Counter(b["fine_id"] for b in master if b["fine_id"] is not None)
    fvals = [fc[i] for i in range(len(fine_names))]
    stats["fine_imbalance_ratio_max_over_min"] = round(max(fvals) / min(fvals), 1)
    (outdir / "report_stats.json").write_text(json.dumps(stats, indent=2))

    # Emit the chosen threshold as a sidecar (do NOT clobber the commented
    # config). Set configs/data.yaml:small_object.area_px_max to this value.
    (outdir / "small_object_threshold.json").write_text(json.dumps({
        "area_px_max": small["chosen_area_px_max"],
        "size_px_sqrt": small["chosen_size_px_sqrt"],
        "rule": small["chosen_rule"],
        "coco_small_fraction": small["coco_small_fraction(area<1024)"],
    }, indent=2))

    print("=== SMALL-OBJECT ANALYSIS ===")
    print(f"object size sqrt(area) percentiles (px): "
          + ", ".join(f"p{p}={v:.0f}" for p, v in small['size_px_percentiles_sqrt_area'].items()))
    print(f"COCO 'small' (area<1024px) would be only {small['coco_small_fraction(area<1024)']*100:.2f}% of boxes")
    print(f"CHOSEN small threshold: area < {small['chosen_area_px_max']:.0f} px^2 "
          f"(sqrt {small['chosen_size_px_sqrt']:.0f}px) -> "
          f"{small['fraction_small_under_chosen']*100:.1f}% of boxes are 'small'")
    print(f"fine imbalance (max/min instances): {stats['fine_imbalance_ratio_max_over_min']}x")
    print(f"\nfigures + samples -> {outdir}")
    print("samples:", picks)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
