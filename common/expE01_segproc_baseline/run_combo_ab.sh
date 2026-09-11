#!/usr/bin/env bash
# expI04 — FO 検出索引 × アンカー工程区間の合成
#
#   1. time 形式          → FO 索引（アンカーがあればその工程区間と積）
#   2. 非 time × アンカー有 → アンカーの工程区間（位置 prior なし = expI02-A）
#   3. それ以外            → 一様のまま
#
# ★青（匿名）の明示除去は**入れない**。FO 索引だけで青率 9.2%→4.6% と半減し、
#   明示除去すると ±5s 到達が 48.6%→47.4% と下がる（expI03 実測）。
# ★1,027 問だけ推論し残りは ctrl から流用（greedy かつ問ごとに独立なので厳密）。
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"; D="$PWD/workspace/expE01_segproc_baseline"; R="$D/results"
AD="$R/expE03f_joint_all_full/fold0/adapter"
CTRL="$R/eval_expI01_uniform_64fproc448_kf5_n2000"
TAG="eval_expI04_combo_64fproc448_kf5_n2000"
step(){ echo "=== $(date '+%F %T') $* ==="; }
if [ ! -f "$R/${TAG}.part/responses.json" ]; then
  step "INFER combo (1027問)"
  stdbuf -oL -eL "$PY" "$D/run_infer.py" --track PROCEDURE --fold 0 --part val \
    --limit 2000 --n-frames 64 --size 448 --version v004 --grid 5 \
    --frame-select combo --qids "$D/qids_combo.txt" --adapter "$AD" --out-tag "${TAG}.part"
  rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && exit 1
fi
step "SPLICE"; "$PY" "$D/splice_responses.py" "$R/${TAG}.part" "$CTRL" "$R/$TAG"
step "JUDGE";  stdbuf -oL -eL "$PY" "$D/eval_seg.py" "$R/$TAG" --track PROCEDURE
step "COMPARE ctrl vs combo"
stdbuf -oL -eL "$PY" "$D/compare_runs.py" "$CTRL" "$R/$TAG" || true
step "COMPARE fo vs combo"
stdbuf -oL -eL "$PY" "$D/compare_runs.py" "$R/eval_expI03_fo_64fproc448_kf5_n2000" "$R/$TAG" || true
step "DONE"
