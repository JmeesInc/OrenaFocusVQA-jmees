#!/usr/bin/env bash
# SEGMENT: 予算線の反対端「解像度に全振り」を測る。
#
# ★容量の実測（probe_capacity.py, RTX8000 4bit）は `frames × (size/448)^2 <= 96`。
#   448px なら 96枚、768px なら 32枚。**両立しない**ので、どちらが支配的かを決着させる。
#     - temporal_grounding はフレーム数で動く（16f 0.385 → 96f 0.587）
#     - object_recognition は解像度で動く（expE01: 448→768 推論で fo_class +0.038**、
#       SEGMENT 全体で +0.0131 p=0.049* ＝ 今回の候補で唯一 p<0.05 が付いた施策）
# ★対照は 96f@448 + アンカー（現行ベスト 0.6958）。アンカーは両方に付けて条件を揃える。
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"; D="$PWD/workspace/expE01_segproc_baseline"; R="$D/results"
AD="$R/expE06g_joint_frame_segment_FULL/fold0/adapter"
step(){ echo "########## $(date '+%F %T') $* ##########"; }

run_one () {  # $1=タグ $2...=引数
  local TAG="$1"; shift
  if [ -f "$R/$TAG/results_merged.csv" ]; then step "SKIP $TAG（判定済み）"; return 0; fi
  step "INFER $TAG  [$*]"
  "$PY" "$D/run_infer.py" --track SEGMENT --fold 0 --part val --limit 2000 \
    --version v004 --adapter "$AD" --out-tag "$TAG" "$@"
  local rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && return 1
  "$PY" "$D/eval_seg.py" "$R/$TAG" --track SEGMENT; step "JUDGE rc=$?"
  "$PY" "$D/compare_runs.py" "$R/eval_expE06g_seg96f448_anchor_n2000" "$R/$TAG" \
    --labels 96f448_anc "$TAG"
  step "COMPARE rc=$?"
}

# ① 解像度に全振り: 32枚 @768px（予算線の反対端）
run_one "eval_expE06g_seg32f768_anchor_n2000" --n-frames 32 --size 768 --anchor
# ② 中間点: 64枚 @560px（予算線上のもう一点。frames×(560/448)^2 = 100 ≒ 予算ぴったり）
run_one "eval_expE06g_seg64f560_anchor_n2000" --n-frames 64 --size 560 --anchor

step "QUEUE DONE"
