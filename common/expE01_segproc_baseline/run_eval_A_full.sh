#!/usr/bin/env bash
# expE04a（SEGMENT 全量 scratch）の評価。expE04b（warm start 全量, 0.6047）の対照。
# ★qa_v005（= fold v001）。expE04b と同じ val でないと比較にならない。
set -uo pipefail
cd /data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv-dl1/bin/python"; D="$PWD/workspace/expE01_segproc_baseline"; R="$D/results"
TAG=eval_expE04a_segfull_scratch_16f448_n2000
step(){ echo "=== $(date '+%F %T') $* ==="; }
step "INFER $TAG (SEGMENT 16f@448 qa_v005)"
stdbuf -oL -eL "$PY" "$D/run_infer.py" --track SEGMENT --fold 0 --part val --limit 2000 \
  --n-frames 16 --size 448 --version v005 --grid 1.0 \
  --adapter "$R/expE04a_seg_scratch_FULL/fold0/adapter" --out-tag "$TAG"
step "INFER rc=$?"
stdbuf -oL -eL "$PY" "$D/eval_seg.py" "$R/$TAG" --track SEGMENT
step "JUDGE rc=$?"
