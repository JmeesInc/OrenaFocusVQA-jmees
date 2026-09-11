#!/bin/bash
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena/workspace/expE01_segproc_baseline
PY=/mnt/data/data4/src/shunsuke/MICCAI2026/Orena/.venv/bin/python
T=eval_expV07b_seg32f768_anchor_n2000
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
{ echo "=== $(date +%F\ %T) START $T ==="
  $PY run_infer.py --track SEGMENT --fold 0 --part val --limit 2000 --version v004 \
    --n-frames 32 --size 768 --anchor --compute-dtype bfloat16 \
    --adapter results/expV07b_seg_ep2_cont/fold0/adapter --out-tag $T
  echo "=== $(date +%F\ %T) INFER rc=$? ==="
  $PY eval_seg.py results/$T --track SEGMENT
  echo "=== $(date +%F\ %T) EVAL rc=$? DONE ==="; } >> $T.log 2>&1
