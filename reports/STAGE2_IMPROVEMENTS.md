# Stage 2 — Improvement Campaign: from negative result to near-B0 parity

The original Stage-2 study (`STAGE2_REPORT.md`) was a well-isolated **negative
result**: every GNN variant (≈26.6–27.4 mAP@[.5:.95]) sat below the full YOLO
baseline (**B0 = 31.15**) and even below the naive same-pool control
(**Cand+NMS = 28.25**). It correctly diagnosed the cause — the **candidate-pool
ceiling**, not the GNN — and named three fixes. This campaign executed those
fixes step by step, measuring each against the original baseline (S0).

**Result:** the best GNN went **27.43 → 30.25** mAP@[.5:.95] (**+2.82**), closing
the gap to B0 from **3.72 → 0.90**, and lambsquarter AP (82% of instances) rose
**23.3 → 32.0** (**+8.7**). The pool control (Cand+NMS) reached **31.03**, at B0
parity — confirming the ceiling was the whole story.

---

## Method — one isolated change per step, measured vs S0

Every experiment row carries a `step` tag; `src/eval/compare.py` builds the
progression below and `src/eval/pool_recall.py` measures the pool ceiling
directly. Iterated on the focus variant **SAGE→GAT** + fixed anchors (B0,
Cand+NMS); the full 8-variant matrix was re-run once at the end.

### Focus-variant progression (SAGE→GAT)

| step | change | mAP@[.5:.95] | Δ vs S0 | AP lambsquarter | gap to B0 |
|---|---|---|---|---|---|
| **S0** | original baseline | 27.43 ± 0.08 | — | 23.27 | −3.72 |
| S1 | calibration: softmax → **per-class sigmoid** (train+eval) + grad-clip | 27.69 ± 0.65 | +0.27 | 23.95 | −3.46 |
| S2 | pos_iou 0.5 → 0.3 | 27.40 ± 0.40 | −0.03 | 23.72 | *reverted* |
| S3 | candidate pool **topk 300 → 600** (conf 0.05→0.03) | 29.22 ± 0.26 | +1.80 | 30.23 | −1.93 |
| **S4** | candidate pool **topk 600 → 900** + capped-edge graph | **29.73 ± 0.13** | **+2.31** | **31.80** | **−1.42** |

`pos_iou=0.3` was tested on both the thin (S2) and rich (S3b) pool and was neutral
both times → reverted; the loose-match label noise offsets the small recall gain.

### The pool ceiling moved (the decisive diagnostic)

`pool_recall.py` — fraction of GT boxes that have *any* candidate at IoU ≥ 0.5:

| pool | overall GT-recall | dense images (≥100 GT/img) | cands/img |
|---|---|---|---|
| topk=300 (S0–S2) | 0.553 | 0.337 | mean 273, max 300 |
| topk=600 (S3) | 0.751 | 0.578 | mean 473, max 600 |
| **topk=900 (S4)** | **0.835** | **0.743** | mean 602, max 900 |

The GNN can only refine candidates it is handed; lifting this ceiling is exactly
what moved the numbers. The cap is still binding (max=900) → higher resolution /
more candidates would help further (see *Remaining gap*).

---

## Final matrix at topk=900 — before / after (mean ± std, 3 seeds)

| variant | mAP@[.5:.95] before | after | Δ | AP lqtr before → after |
|---|---|---|---|---|
| **B0 (anchor)** | 31.15 | 31.15 | — | 34.4 → 34.4 |
| **Cand+NMS** | 28.25 | **31.03** | +2.78 | 24.4 → 33.0 |
| **GCN** ⭐ best GNN | 27.04 | **30.25 ± 0.18** | **+3.21** | 23.3 → 32.1 |
| GIN | 26.65 | 30.00 ± 0.26 | +3.35 | 23.1 → 31.9 |
| SAGE | 26.90 | 29.83 ± 0.09 | +2.93 | 23.1 → 31.8 |
| GIN→GAT | 27.11 | 29.76 ± 0.45 | +2.65 | 23.3 → 31.9 |
| GAT | 27.41 | 29.69 ± 0.58 | +2.28 | 23.6 → 31.9 |
| SAGE→GAT | 27.43 | 29.59 ± 0.30 | +2.16 | 23.3 → 32.0 |
| GCN→GAT | 26.86 | 29.58 ± 0.18 | +2.72 | 23.5 → 31.8 |
| GAT→GAT | 26.95 | 29.51 ± 0.15 | +2.56 | 23.4 → 31.8 |

Baseline table preserved at `reports/results/baseline_fine.{md,csv}`; new table at
`reports/results/results_fine.{md,csv}`.

### New scientific finding — the attention advantage inverts on a rich pool

In the original (thin-pool) study, attention (GAT) was marginally **best**. On the
richer topk=900 pool the ranking **flips**: simple structural convolutions win and
attention is the weakest family:

- **GCN (30.25) > GIN (30.00) > SAGE (29.83)** ≫ **GAT (29.69), GAT→GAT (29.51)**.
- Adding a GAT block on top of a structural block *never helps* here (e.g. GCN
  30.25 → GCN→GAT 29.58).

Interpretation: with a dense, high-recall pool the graphs are larger and noisier;
uniform neighborhood smoothing (GCN) regularizes the YOLO prior better than
learned attention, which appears to over-fit / de-rank on the noisier dense
neighborhoods. This is a clean, reproducible ablation result (std ≤ 0.6, 3 seeds).

---

## What changed (code + config)

- `src/gnn/eval.py` — per-class **sigmoid** decode (was softmax-over-C+1).
- `src/gnn/train.py` — **sigmoid focal loss** (coherent with the decode) +
  `clip_grad_norm_`; `--tag` step logging.
- `src/gnn/graph.py` — feature-similarity edges **capped to top-k per node**
  (was `sim ≥ thresh`, which explodes to O(N²) and OOMs GAT on the 900-node pool).
- `configs/gnn.yaml` — `topk 300→900`, `conf 0.05→0.03`, `cls_loss=sigmoid_focal`,
  `grad_clip=5.0`, `batch_images 8→2` (8 GB VRAM headroom at 900 nodes).
- New: `src/eval/compare.py` (progression vs S0), `src/eval/pool_recall.py`
  (pool ceiling probe).
- Fixed 1000 broken image symlinks (pointed at the old `GNN_NEW/` project path).

## Remaining gap and next steps

Best GNN (GCN, 30.25) is **0.9 below B0** and ~0.8 below the pool control — i.e.
the learned rescoring still slightly trails plain NMS on the same candidates, and
the pool cap (max=900) is still binding on the densest images. The two levers left
are the original report's #1 fix, now the clear frontier:

1. **Higher Stage-1 resolution** (retrain YOLO at 1280+ or tiled inference) — lifts
   the pool recall further; the cap is still binding at 900.
2. **Preserve YOLO's ranking more** — the GNN's rescoring costs ~0.8 vs NMS; a
   lighter-touch residual (or a rank-preserving loss) could recover it.

Reproduce: `TAG=final bash scripts/run_stage2_matrix.sh` then
`python src/eval/compare.py` and `python src/eval/pool_recall.py`.
