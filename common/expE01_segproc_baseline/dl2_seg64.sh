#!/usr/bin/env bash
# expE06g（F+S 全量）を SEGMENT 64フレームで評価する。
# ★根拠: reachability.py の実測で SEGMENT の time 問題は
#   **16f では 75.4% が「刻み > ±5s」で到達不能、64f では 0.0%** になる。
#   PROCEDURE と違い SEGMENT は 64f で分解能の壁が完全に消えるので、
#   temporal_grounding（現状 0.3849, N=795）が最も大きく動く可能性が高い。
# ★latency 上限 15s に対し dl1 実測 64f@448 = 10.5s。4090 ならさらに速い。
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2   # 4090
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"
D="$PWD/workspace/expE01_segproc_baseline"; R="$D/results"
AD="$R/expE06g_joint_frame_segment_FULL/fold0/adapter"
TAG="eval_expE06g_seg64f448_n2000"
echo "=== $(date '+%F %T') INFER SEGMENT 64f ==="
"$PY" "$D/run_infer.py" --track SEGMENT --fold 0 --part val --limit 2000 \
  --n-frames 64 --size 448 --version v004 --adapter "$AD" --out-tag "$TAG"
echo "=== $(date '+%F %T') INFER rc=$? ==="
"$PY" "$D/eval_seg.py" "$R/$TAG" --track SEGMENT
echo "=== $(date '+%F %T') JUDGE rc=$? ==="
"$PY" "$D/compare_runs.py" "$R/eval_expE06g_seg16f448_n2000" "$R/$TAG" --labels 16f 64f
echo "=== $(date '+%F %T') DONE ==="
