# Stage 2 — GNN Refinement: Results & Analysis

Two-stage detector: a **frozen** YOLO11s (Stage 1) whose pre-NMS candidates are
refined by a small GNN head (Stage 2). Full experiment matrix, 3 seeds each,
evaluated with **pycocotools** (same evaluator for every row, incl. our 1920px²
small-object threshold). Reproduce: `bash scripts/run_stage2_matrix.sh`.

## Headline result

> **The GNN refinement does not improve detection on this dataset.** Every GNN
> variant (≈26.6–27.4 mAP@[.5:.95]) lands *below* both the full YOLO baseline
> (**B0 = 31.2**) and the naive "same candidate pool + NMS, no GNN" control
> (**Cand+NMS = 28.3**). This is a real, well-isolated negative result, not a
> pipeline bug — the evaluator reproduces B0 from raw candidates, and B0 itself
> scores as expected.

| variant | mAP@[.5:.95] | mAP@.5 | mAP_small | AP lambsquarter | params |
|---|---|---|---|---|---|
| **B0 (YOLO11s)** | **31.15 ± 0.34** | **49.23 ± 0.68** | 5.63 | **34.39** | 9.42M |
| Cand+NMS (pool, no GNN) | 28.25 | 42.21 | 4.15 | 24.35 | — |
| GAT only | 27.41 ± 0.28 | 41.46 | 4.74 | 23.59 | +0.24M |
| GCN only | 27.04 ± 0.38 | 40.51 | 4.27 | 23.25 | +0.24M |
| GIN only | 26.65 ± 0.39 | 40.24 | 4.74 | 23.13 | +0.37M |
| GraphSAGE only | 26.90 ± 0.19 | 40.55 | 4.51 | 23.14 | +0.37M |
| GAT→GAT (control) | 26.95 ± 0.61 | 41.02 | 4.73 | 23.39 | +0.37M |
| GCN→GAT | 26.86 ± 0.40 | 41.04 | 3.90 | 23.45 | +0.37M |
| GIN→GAT | 27.11 ± 0.18 | 41.03 | 4.78 | 23.32 | +0.50M |
| **SAGE→GAT** (best hybrid) | **27.43 ± 0.08** | 41.67 | 4.16 | 23.27 | +0.50M |

Full table incl. all 6 per-class APs: `reports/results/results_fine.{md,csv}`.
Plots: `overall_mAP_fine.png`, `per_class_AP_fine.png`. Qualitative B0-vs-hybrid
panels: `reports/qualitative/compare_*.jpg`.

## Why the GNN underperforms — root cause is the candidate pool, not the GNN

The decisive number is **lambsquarter AP: 34.4 (B0) → 24.4 (Cand+NMS) → ~23 (all GNN)**.
The ~10-point collapse happens at the *Cand+NMS* step, i.e. before the GNN does
anything. Lambsquarter is 82% of instances and dominated by tiny seedlings in
dense images (up to 295 objects/img). Keeping only the **top-300 pre-NMS
candidates** truncates most of them: measured GT-recall@IoU0.5 in the top-300
pool is only ~0.14 on the densest images. The GNN can only refine candidates it
is given; it cannot recover objects the pool already dropped. So both the pool
control and every GNN variant inherit this ceiling, ~3 points below full B0.

Qualitatively this is stark: on a 169-GT image, **B0 emits 124 detections vs the
hybrid's 51** — the two-stage model under-detects dense scenes
(`reports/qualitative/compare_rgb-2022-10-10-15-33-03.jpg`).

## What the ablation does show (the intended scientific comparison)

Comparing GNN variants to each other (all on the identical pool) is still valid:

- **Attention helps, slightly.** GAT-containing variants are consistently at the
  top (GAT 27.41, SAGE→GAT 27.43) vs pure structural aggregation (GIN 26.65,
  GCN 27.04, SAGE 26.90). The edge is small (~0.5–0.8 mAP) but consistent across
  3 seeds (std ≤ 0.4).
- **Hybrids (A→GAT) help weak Block-A choices**: GIN 26.65 → GIN→GAT 27.11;
  SAGE 26.90 → SAGE→GAT 27.43. They are neutral for GCN (27.04 → 26.86).
- **The GAT→GAT control (26.95) does not beat single-block GAT (27.41)** — so the
  marginal hybrid gains are not just "more layers/params."
- **Gains (where they exist) are not from model size**: GNN heads add only
  0.24–0.50M params (2.5–5.3% on top of the 9.42M frozen YOLO).
- None of this changes the headline: the best GNN (27.4) is still below the pool
  control (28.3). Even the *learned rescoring/reclassification* slightly hurts vs
  plain NMS on YOLO's own (well-calibrated) scores.

## Cost

GNN refinement adds **~7 ms/img** (graph build + forward + decode + NMS), taking
end-to-end throughput from **38.8 FPS (B0) → ~30.5 FPS** — i.e. ~27% slower for
no accuracy gain on this dataset.

## Honest interpretation / what would actually move the needle

The bottleneck is Stage-1 recall on tiny dense objects, not the refinement head:

1. **Raise the candidate-pool ceiling** — train/infer Stage 1 at higher resolution
   (native is 2046×1080; we used 1024 due to the 8 GB GPU) and/or drop the top-K
   cap. This is the single most promising change.
2. **Preserve YOLO's calibration** — use per-class *sigmoid* scoring (YOLO's own
   paradigm) refined by a residual, instead of softmax-over-(C+1); the softmax +
   quality recalibration appears to slightly degrade ranking.
3. **Match candidates at lower IoU for box regression** (e.g. 0.3) so the GNN can
   pull partially-overlapping candidates onto small GT, instead of treating them
   as background.

## Deviations from the brief (all flagged, config-driven)

- **topk 150 → 300**: at 150, dense-image GT-recall was ~0.14; 300 roughly doubles
  it while keeping graphs small. (Still the limiting factor — see above.)
- **Residual-refinement design**: the GNN starts from the YOLO score/box prior
  (zero-init heads + class-score logit prior) so it can only refine, not reset,
  the detector. Without this, mAP was ~1 point lower (0.262 vs 0.270 for GCN→GAT).
- **Cand+NMS reference row** added as the apples-to-apples control for "does the
  GNN beat naive NMS on the same candidates?" (it doesn't).
