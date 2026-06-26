#!/usr/bin/env bash
# Run the full experiment matrix: B0 (pycoco eval) + 8 GNN variants, each x3 seeds.
# Assumes the GPU is free (B0 seeds trained, candidate cache precomputed).
set -e
cd "$(dirname "$0")/.."

SEEDS="0 1 2"
B0_CKPT_TMPL="runs/stage1_yolo/b0_yolo11s_seed%d/weights/best.pt"

echo "===== B0 baseline (pycocotools eval, 3 seeds) ====="
for s in $SEEDS; do
  ckpt=$(printf "$B0_CKPT_TMPL" "$s")
  [ -f "$ckpt" ] && python3 src/gnn/eval_b0.py --ckpt "$ckpt" --seed "$s" \
    || echo "skip B0 seed $s (no ckpt $ckpt)"
done

echo "===== Cand+NMS reference (same candidate pool, no GNN) ====="
python3 src/gnn/eval_pool.py

# variant := "block_a block_b"  (none = skip that block)
VARIANTS=(
  "none gat"   # GAT only
  "gcn none"   # GCN only
  "gin none"   # GIN only
  "sage none"  # GraphSAGE only
  "gat gat"    # GAT->GAT control
  "gcn gat"    # GCN->GAT  (proposed)
  "gin gat"    # GIN->GAT  (proposed)
  "sage gat"   # SAGE->GAT (proposed)
)

for v in "${VARIANTS[@]}"; do
  set -- $v; A=$1; B=$2
  for s in $SEEDS; do
    echo "===== GNN block_a=$A block_b=$B seed=$s ====="
    python3 src/gnn/train.py --block_a "$A" --block_b "$B" --seed "$s"
  done
done

echo "===== building results table/plots ====="
python3 src/eval/results.py fine
echo "DONE."
