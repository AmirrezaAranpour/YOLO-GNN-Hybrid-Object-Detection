# Task 0 — ACRE Crop-Weed Data Audit & Conversion Report

All numbers below were extracted from the actual 1000 XML/JPG files, not from the
datasheet or the provided COCO example. Reproduce with:

```bash
python src/data/convert.py      # XML -> YOLO (fine 6-class + coarse 2-class)
python src/data/split.py        # stratified 60/20/20 + Ultralytics yamls
python src/data/report.py       # figures + this report's numbers
```

## 1. Dataset at a glance

| Property | Value |
|---|---|
| Images | **1000** (Maize: 500 / Bean: 500, 10 batches × 100) |
| Resolution | **2046 × 1080** for *all* 1000 images (verified, no exceptions) |
| jpg ↔ xml pairing | perfect 1:1, 0 orphans |
| Annotation format | per-image XML, polygon instance-segmentation (`<clipping>`) |
| Total instances | **56,530** polygons |
| Instances / image | min 0, **mean 56.5**, median 46, max **295** |
| Background-only images | **2** (`rgb-2022-10-06-17-19-51`, `rgb-2022-10-10-15-59-00`) |
| Encoding fallback (iso-8859-5) needed | 0 files |
| `isTrusted=false` instances | 11 / 56,530 (negligible) |

Boxes are the tight min/max extent of each polygon, clamped to the image. Our
coarse box count (56,530) **matches the official COCO conversion exactly
(delta = 0)** → the geometry pipeline is correct. 0 boxes hit the degenerate-size
filters.

## 2. The class story (this is the important part)

The XML carries **two** label fields per instance: a `<class>` (crop/weed/`unknow`)
and a French species `<name>`. The official script only ever produced the coarse
crop/weed labels. For the 6-species task we read `<name>`, and the raw data is
messier than "6 clean classes":

**Verified `<name>` vocabulary (all 1000 files):**

| French `<name>` | → species (id) | group | instances |
|---|---|---|---:|
| `chenopode`  | lambsquarter (5) | weed | **46,638** |
| `haricot`    | bean (1)         | crop | 4,119 |
| `mais`       | maize (0)        | crop | 1,821 |
| `ray-grass`  | ryegrass (2)     | weed | 1,002 |
| `matricaire` | matricaria (4)   | weed | 955 |
| `moutarde`   | mustard (3)      | weed | 710 |
| `inconnu`    | *unknown* — OOV  | — | 861 |
| *(missing)*  | *unnamed* — OOV  | — | 379 |
| `feverole`   | faba bean — OOV  | crop | 19 |
| `Pourpier`   | purslane — OOV   | weed | 11 |
| `digitaire`  | crabgrass — OOV  | weed | 8 |
| `pois`       | pea — OOV        | crop | 4 |
| `setaire`    | foxtail — OOV    | weed | 3 |

Three findings that affect modeling:

1. **Severe imbalance.** lambsquarter alone is **82%** of all instances; the
   fine-class max/min ratio is **65.7×** (46,638 vs 710). See
   `class_histogram.png`. This *will* dominate plain mAP and needs class-weighting
   / focal loss downstream.
2. **More than 6 species + unlabeled.** 1,285 instances (2.3%) are out-of-vocab
   (`inconnu`, unnamed, or 5 extra species with 3–19 examples each).
3. **A few label conflicts** (e.g. 1 box tagged `<class>crop` but `<name>matricaire`).
   Logged in `convert_summary.json`; we go by `<name>` for species and flag these.

### Decisions taken (configurable in `configs/data.yaml`, flagged per the brief)

- **Primary = fine 6-class** by species; **secondary = coarse crop/weed** (built,
  not defaulted-to). Coarse folds `unknow → weed` (matches the official script).
- **OOV policy = `drop` from fine labels** (the 5 micro-species have 3–19 examples —
  too few to learn or evaluate as classes, and `inconnu`/unnamed have no species).
  They are **kept in the coarse labels** via the `<class>` tag, so no real plant
  silently becomes background in the 2-class view. Net fine set = **55,245** boxes.
- Coarse totals: crop 5,994 / weed 50,536.

## 3. Bounding-box size distribution & "small object" definition

See `bbox_size_distribution.png` and `bbox_size_per_class.png`.

Object size = √area, percentiles (px): p5=23, p10=28, p25=39, **p33=44**,
p50=54, p75=83, p90=162, p95=247. Median object ≈ 54×54 px, i.e. ~0.13% of the
2046×1080 frame. Aspect ratios cluster near 1 (0.5–2).

**The COCO 32×32 (area<1024) convention is wrong for this dataset** — it would
label only **14.3%** of boxes "small" at this resolution. We instead define:

> **small object ⇔ bbox area < 1920 px² (≈ 44×44 px)** = the lower tercile of the
> actual size distribution → **33%** of boxes. (`small_object.area_px_max` in config.)

This gives a meaningful "small" bucket for the small-object mAP reporting later.

## 4. Stratified split (image-level, 60/20/20, seed 42)

Multi-label iterative stratification on the 6 fine classes. Result — every class
lands in all three splits at ~60/20/20 *image* presence, including the rare ones:

| | train | val | test |
|---|---:|---:|---:|
| **images** | 597 | 202 | 201 |
| maize (img) | 305 | 102 | 102 |
| bean (img) | 302 | 100 | 101 |
| ryegrass (img) | 223 | 74 | 74 |
| mustard (img) | 158 | 53 | 53 |
| matricaria (img) | 166 | 55 | 55 |
| lambsquarter (img) | 595 | 199 | 198 |

Full per-class instance & image counts: `data/acre_yolo/split_report.json`.
(The dataset also ships an official 70/10/20 split in `split_dictionary.json`; we
use our own 60/20/20 per the brief but it's available if you want to align.)

## 5. Outputs produced

```
data/acre_yolo/
  images/<stem>.jpg            symlinks to raw images
  labels_fine/<stem>.txt       YOLO 6-class labels  (PRIMARY)
  labels_coarse/<stem>.txt     YOLO 2-class labels  (SECONDARY)
  polygons/<stem>.json         original polygons + per-instance metadata (kept)
  annotations_master.json      flat per-box geometry+labels (analysis)
  splits.json / split_report.json
  fine/  coarse/               Ultralytics images|labels/{train,val,test} trees
  acre_fine.yaml acre_coarse.yaml   ready-to-train dataset configs
reports/task0/                 these figures + JSON + this report
```

## 6. Sample images (boxes drawn, colored by class)

`sample_*.jpg` — densest image, maize-heavy, bean-heavy, and the rare-weed
exemplars. They confirm boxes align to plants and show the dense/occluded
small-object regime (one image has 295 instances) that motivates the GNN stage.
