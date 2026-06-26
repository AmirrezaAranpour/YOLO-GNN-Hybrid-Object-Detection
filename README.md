# YOLO-GNN Hybrid Object Detection — ACRE Crop/Weed

A two-stage detector for precision agriculture: a **frozen YOLO11 baseline**
(Stage 1) followed by a **Graph Neural Network refinement head** (Stage 2) that
reasons over pre-NMS candidate detections. The hypothesis: by modeling the
*relationships* between nearby candidate boxes (a graph), a GNN can recover small
and overlapping crop/weed detections that single-box post-processing (NMS) misses.

The design is **deliberately two-stage** — the YOLO detector is frozen and the
GNN is trained separately — rather than end-to-end. This buys a clean ablation:
every accuracy change is attributable to the GNN, not to co-adapted detector
weights, which is what makes the experiment matrix below scientifically meaningful.

---

## TL;DR — Result

> **The GNN refinement does *not* beat the YOLO baseline on this dataset, and we
> can say precisely why.** This is a real, well-isolated negative result, not a
> pipeline bug.

| Model | mAP@[.5:.95] | mAP@.5 | mAP_small | AP lambsquarter |
|---|---|---|---|---|
| **B0 — YOLO11s (baseline)** | **31.15 ± 0.34** | **49.23** | 5.63 | **34.39** |
| Cand+NMS (same pool, *no* GNN) | 28.25 | 42.21 | 4.15 | 24.35 |
| SAGE→GAT (best GNN) | 27.43 ± 0.08 | 41.67 | 4.16 | 23.27 |
| GAT only | 27.41 ± 0.28 | 41.46 | 4.74 | 23.59 |
| (all GNN variants) | 26.65 – 27.43 | ~40–42 | ~4–5 | ~23 |

**Root cause:** the bottleneck is the **candidate pool**, not the GNN. The
lambsquarter AP collapse (34.4 → 24.4) happens at the *Cand+NMS* step — **before
the GNN does anything**. Keeping only the top-300 pre-NMS candidates truncates the
tiny, dense seedlings (up to 295 objects/image; GT-recall in the pool is only
~0.14 on the densest images). A GNN can only refine candidates it is handed; it
cannot recover objects the pool already dropped. This is fundamentally a Stage-1
**resolution/recall** limitation (we ran at 1024px on an 8 GB GPU vs the native
2046×1080), not a flaw in the refinement head.

The variant **ablation is still valid** (same pool for every row): attention (GAT)
is marginally but consistently best; structural hybrids (A→GAT) help weak Block-A
choices; the gains are not explained by parameter count or layer depth.

Full analysis: [`reports/STAGE2_REPORT.md`](reports/STAGE2_REPORT.md).

---

## The dataset — ACRE Crop/Weed

ACRE Crop-Weed (POLIMI / METRICS, CC-BY-4.0,
[Zenodo 8102217](https://zenodo.org/records/8102217)): 1000 field images at
2046×1080, instance-segmentation polygons in XML, 6 plant species
(maize, bean, ryegrass, mustard, matricaria, lambsquarter).

The **Task 0 audit** ([`reports/task0/DATA_REPORT.md`](reports/task0/DATA_REPORT.md))
surfaced several non-obvious facts that drove the whole design:

- **Labels are French** in the XML `<name>` tag (e.g. *chénopode* = lambsquarter),
  with a separate `<class>` tag of crop / weed / `unknow`.
- **Extreme class imbalance**: lambsquarter alone is **82%** of the 56,530
  instances; the rarest-to-commonest ratio is **65.7×**. → motivates focal +
  class-weighted loss.
- **More than 6 species present**: 1,285 out-of-vocabulary instances (`inconnu`,
  unnamed, and 5 micro-species with 3–19 examples each) are **dropped** from the
  6-class *fine* labels but **kept** in the 2-class *coarse* (crop/weed) labels.
- **Small-object threshold is data-driven**: area < **1920 px²** (≈44×44, the 33rd
  percentile of √area) — COCO's 32px convention would flag only ~14% of these boxes.
- **Our own 60/20/20 multi-label stratified split** (seed 42) is used so every
  species appears proportionally in train/val/test — not the dataset's official
  70/10/20.

The XML→YOLO conversion was validated against the official COCO annotations
(instance count delta = 0).

---

## Architecture

```
                  Stage 1 (frozen)                         Stage 2 (trained)
  image ──► YOLO11s ──► pre-NMS candidates ──► graph build ──► GNN head ──► refined
            (P3/P4/P5    top-300 boxes +          kNN over       Block A         boxes
             neck maps)  ROI appearance          space+feat      → Block B    + scores
                         = 139-dim nodes          edges          (GAT)
```

- **Candidate nodes (139-dim):** geometry (4) + YOLO class scores (6) + objectness
  (1) + an ROI-aligned appearance vector pooled from the P3 neck map (128).
- **Graph:** per-image k-NN (k=12) over a *blend* of spatial proximity and feature
  cosine similarity (sim ≥ 0.5), undirected + coalesced.
- **GNN head:** swappable **Block A** (GCN / GIN / GraphSAGE / None) → **Block B**
  (GAT, 4 heads), residual connections + LayerNorm per layer. Outputs class logits,
  an IoU-quality delta, and a box residual.
- **Residual-refinement design (key):** all output heads are zero-initialized and
  the background bias is −4.0, and `forward_with_prior()` injects YOLO's own score
  logits as a prior. So the GNN *starts as the identity* on YOLO's predictions and
  can only refine them — it cannot reset a well-calibrated detector from scratch.
  (Without this, mAP was ~1 point lower.)
- **Stage-2 loss:** focal CE (γ=2.0, class-weighted) + IoU-aware BCE quality + CIoU
  on positive matches (IoU ≥ 0.5).

---

## The experiment matrix

Every row changes only `(block_a, block_b)` in [`configs/gnn.yaml`](configs/gnn.yaml);
nothing else differs. 8 GNN variants × 3 seeds, plus baselines, all evaluated with
the **same pycocotools** harness (including our 1920px² small-object range):

| Block A \ Block B | None | GAT |
|---|---|---|
| **None** | — | GAT only |
| **GCN** | GCN only | GCN→GAT |
| **GIN** | GIN only | GIN→GAT |
| **GraphSAGE** | SAGE only | SAGE→GAT |

Plus two reference rows: **B0** (full YOLO, post-NMS) and **Cand+NMS** (the
identical top-300 pool with plain NMS and *no* GNN — the apples-to-apples control
for "does the GNN beat naive NMS on the same candidates?"). It doesn't.

**Cost:** the GNN adds ~7 ms/image (graph build + forward + decode + NMS), taking
end-to-end throughput from **38.8 FPS (B0) → ~30.5 FPS** — i.e. ~27% slower for no
accuracy gain *on this dataset*.

---

## Repository layout

```
configs/
  data.yaml              all data knobs: class maps, OOV policy, split, small-object threshold
  yolo.yaml              Stage 1: YOLO11s training hyperparameters
  gnn.yaml               Stage 2: candidate/graph/model/training knobs (one row of the matrix)
src/data/
  config.py              config loader (resolves paths to repo root)
  convert.py             XML polygons -> YOLO labels (6-class fine + 2-class coarse) + master json
  split.py               multi-label stratified 60/20/20 split + Ultralytics data.yamls
  report.py              class histograms, bbox-size dist, small-object threshold, annotated samples
src/yolo/
  train.py               Stage 1: thin Ultralytics wrapper (train + test eval + log)
src/gnn/
  extract.py             pre-NMS candidate extraction from frozen YOLO -> cache
  graph.py               per-image kNN graph (spatial + feature-similarity edges)
  models.py              swappable GNNHead(block_a, block_b) + residual prior
  dataset.py             cache -> graphs with IoU-matched node targets
  train.py               Stage 2: train one GNN variant (focal + CIoU + IoU-quality)
  eval.py / eval_b0.py / eval_pool.py   pycocotools eval (GNN / B0 / pool control)
src/eval/
  results.py             aggregate experiments.jsonl -> table + ablation plots
  qualitative.py         B0 vs hybrid side-by-side on dense images
scripts/
  run_stage2_matrix.sh   runs B0 + Cand+NMS + 8 GNN variants x3 seeds + results
data/
  ACRE_raw/              unpacked Zenodo dataset (images + scripts + datasheet)
  acre_yolo/             generated YOLO dataset + cand_cache/ (regenerable)
logs/experiments.jsonl   one row per (variant, seed) run
reports/task0/           DATA_REPORT.md + figures
reports/results/         results_fine.{md,csv} + ablation plots
reports/STAGE2_REPORT.md Stage 2 analysis (the headline negative result)
reports/qualitative/     B0-vs-hybrid comparison images
```

---

## Reproduce

```bash
# Task 0 — data audit, conversion, split, report
python src/data/convert.py
python src/data/split.py
python src/data/report.py

# Stage 1 — YOLO11s baseline (per seed)
python src/yolo/train.py --seed 0     # repeat for seeds 1, 2

# Stage 2 — full matrix: B0 + Cand+NMS + 8 GNN variants x 3 seeds + results
bash scripts/run_stage2_matrix.sh

# Qualitative comparison panels on the densest test images
python src/eval/qualitative.py --variant SAGE->GAT --seed 0
```

Everything is config-driven — no hyperparameters are hardcoded in the source.

---

## What would actually move the needle

Documented in detail in the Stage 2 report; the ranking matters:

1. **Raise the candidate-pool ceiling** — train/infer Stage 1 at higher resolution
   (native 2046×1080 vs our 1024) and/or drop the top-K cap. This is the single
   most promising change, because Stage-1 recall on tiny dense objects is the
   binding constraint.
2. **Preserve YOLO's calibration** — use per-class *sigmoid* scoring refined by a
   residual instead of softmax-over-(C+1); the softmax recalibration appears to
   slightly degrade ranking.
3. **Match candidates at a lower IoU** (e.g. 0.3) for box regression, so the GNN
   can pull partially-overlapping candidates onto small GT rather than discarding
   them as background.

---

## Environment

Python 3.13 (miniconda). torch 2.10 (cu128), torchvision 0.27, ultralytics 8.4,
torch_geometric 2.8, pycocotools, numpy, scikit-learn, PIL, matplotlib, pyyaml,
tqdm.

**Hardware note:** developed on an 8 GB GPU (RTX 3070 laptop), which is the
binding constraint throughout — it forced imgsz=1024 and batch=4 in Stage 1, and
is the root of the recall ceiling described above. PyG's `knn_graph` requires
`pyg-lib` (unavailable for this torch build), so graph construction uses a manual
`torch.cdist` + `topk` k-NN, which is fine for ≤300 nodes/graph.
