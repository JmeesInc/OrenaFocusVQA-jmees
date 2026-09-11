#!/usr/bin/env bash
# expI01 — PROCEDURE のフレーム推薦 A/B（工程索引 vs 一様）
#
#   ctrl  : --frame-select uniform  （現行）
#   phase : --frame-select phase    （expI00_phase_clf の推薦ルール）
#
# ★対照も測り直す。保存済みの 0.4660 は dl1 での実測で、GPU が違うと latency が
#   歪む（同一モデルが 1.90s→4.45s に化けた前例）。**同一 GPU・同一時刻**で取る。
# ★差は `--frame-select` の1つだけ。greedy(do_sample=False) なので同一入力なら
#   出力は決定的＝差はフレーム選択だけに帰着する。
#
# Usage: GPU=2 ./run_phase_ab.sh          （dl2 の 4090 は PCI 順 idx2）
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"          # dl2 は cu130 の共有 .venv
D="$PWD/workspace/expE01_segproc_baseline"
R="$D/results"
AD="$R/expE03f_joint_all_full/fold0/adapter"
step(){ echo "=== $(date '+%F %T') $* ==="; }

for ARM in uniform phase; do
  TAG="eval_expI01_${ARM}_64fproc448_kf5_n2000"
  if [ -f "$R/$TAG/responses.json" ]; then step "SKIP $ARM (既に responses.json あり)"; continue; fi
  step "INFER $ARM  PROCEDURE val 2000 @ 64f/448/grid5"
  stdbuf -oL -eL "$PY" "$D/run_infer.py" --track PROCEDURE --fold 0 --part val \
    --limit 2000 --n-frames 64 --size 448 --version v004 --grid 5 \
    --frame-select "$ARM" --adapter "$AD" --out-tag "$TAG"
  rc=$?; step "INFER $ARM rc=$rc"; [ "$rc" != "0" ] && exit 1
done

for ARM in uniform phase; do
  TAG="eval_expI01_${ARM}_64fproc448_kf5_n2000"
  step "JUDGE $ARM"
  stdbuf -oL -eL "$PY" "$D/eval_seg.py" "$R/$TAG" --track PROCEDURE
  step "JUDGE $ARM rc=$?"
done

step "COMPARE uniform vs phase (matched, McNemar)"
stdbuf -oL -eL "$PY" "$D/compare_runs.py" \
  "$R/eval_expI01_uniform_64fproc448_kf5_n2000" \
  "$R/eval_expI01_phase_64fproc448_kf5_n2000" || true
step "DONE"
