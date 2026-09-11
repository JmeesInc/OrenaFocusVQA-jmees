#!/usr/bin/env bash
# 重畳バリアントのスクリーニング: 1 GPU が渡されたアームを順に「推論 → judge」する。
#
#   GPU=0 ./run_screen.sh control r0
#
# ★アーム間の latency 比較を汚さないため、**同じ GPU で順に**回す
#   （[[gpu-contention-confounds-latency]]）。ただし今回見るのは SCORE なので
#   アームを GPU に散らしても採点自体は成立する。
# ★judge まで含めて 1 レーンにするのは、推論が終わったアームから採点を進めるため。
set -uo pipefail
cd "${REPO:-/data4/src/shunsuke/MICCAI2026/Orena}"
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="${PYBIN:-$PWD/.venv-dl1/bin/python}"
D="$PWD/workspace/expM00_frame_overlay"
ADAPTER="${ADAPTER:-$PWD/workspace/expE01_segproc_baseline/results/expK01F_frame_pseudo/fold0/adapter}"
LIMIT="${LIMIT:-}"

step() { echo "=== $(date '+%F %T') $* ==="; }

for arm in "$@"; do
  out="$D/results/screen_$arm"
  if [ -f "$out/results_judge.csv" ]; then step "SKIP $arm（採点済み）"; continue; fi
  # arm 名 → screen_infer の引数
  case "$arm" in
    control) A=(--variant control) ;;
    t1)      A=(--variant r0 --t1) ;;
    r0|r1|r2|r3) A=(--variant "$arm") ;;
    *) echo "unknown arm: $arm"; exit 2 ;;
  esac
  if [ ! -f "$out/responses.json" ]; then
    step "INFER $arm  (GPU $CUDA_VISIBLE_DEVICES)"
    stdbuf -oL -eL "$PY" "$D/screen_infer.py" --adapter "$ADAPTER" "${A[@]}" \
      --out-tag "screen_$arm" ${LIMIT:+--limit $LIMIT}
    rc=$?; step "INFER $arm rc=$rc"; [ "$rc" -ne 0 ] && continue
  else
    step "SKIP INFER $arm（responses.json あり）"
  fi
  step "JUDGE $arm"
  stdbuf -oL -eL "$PY" "$PWD/workspace/expE01_segproc_baseline/eval_seg.py" "$out" --track FRAME
  step "JUDGE $arm rc=$?"
done
step "LANE DONE: $*"
