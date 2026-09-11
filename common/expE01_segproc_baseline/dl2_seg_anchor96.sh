#!/usr/bin/env bash
# 96f + アンカー（問題文中の hh:mm:ss のフレームを必ず含める）。
# ★対照は 96f 固定（eval_expE06g_seg96f448_n2000, SCORE 0.6936）＝**差はアンカーだけ**。
# ★64f では +0.0034（7勝3敗, n.s.）と極小だったが、**競技なので期待値が正なら積む**方針。
#   96f では刻みが 1.26s とさらに細かいので効果はさらに小さい可能性があるが、
#   悪化リスクは低い（アンカーは枚数を変えず、最も近い一様点を置換するだけ）。
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"; D="$PWD/workspace/expE01_segproc_baseline"; R="$D/results"
AD="$R/expE06g_joint_frame_segment_FULL/fold0/adapter"
TAG="eval_expE06g_seg96f448_anchor_n2000"
step(){ echo "########## $(date '+%F %T') $* ##########"; }
step "INFER $TAG"
"$PY" "$D/run_infer.py" --track SEGMENT --fold 0 --part val --limit 2000 \
  --n-frames 96 --size 448 --version v004 --anchor --adapter "$AD" --out-tag "$TAG"
rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && exit 1
"$PY" "$D/eval_seg.py" "$R/$TAG" --track SEGMENT; step "JUDGE rc=$?"
"$PY" "$D/compare_runs.py" "$R/eval_expE06g_seg96f448_n2000" "$R/$TAG" --labels 96f 96f_anchor
step "DONE rc=$?"
