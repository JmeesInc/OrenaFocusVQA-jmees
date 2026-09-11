#!/usr/bin/env bash
# expI02 — anchor 経路の2変種。expI01 で `position` が有害・`anchor` が有望と分かったので、
#          工程モデルを使う方策だけを残し、anchor の掛け方を2通り試す。
#
#   A_pureanchor : anchor 方策で位置 prior を掛けない（純粋にアンカーの phase 区間）
#   B_share025   : 一様に残す予算を 0.5 → 0.25 に下げて集中を強める
#
# ★どちらも**介入する 690 問だけ**推論し、残りは ctrl の応答を流用する
#   （greedy かつフレーム選択が問ごとに独立なので流用は厳密）。1本 ~55分。
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"
D="$PWD/workspace/expE01_segproc_baseline"
R="$D/results"
AD="$R/expE03f_joint_all_full/fold0/adapter"
CTRL="$R/eval_expI01_uniform_64fproc448_kf5_n2000"
step(){ echo "=== $(date '+%F %T') $* ==="; }

run_arm () {
  local TAG="$1"; shift
  local PART="$R/${TAG}.part"
  if [ ! -f "$PART/responses.json" ]; then
    step "INFER $TAG (690問のみ)"
    env "$@" stdbuf -oL -eL "$PY" "$D/run_infer.py" --track PROCEDURE --fold 0 --part val \
      --limit 2000 --n-frames 64 --size 448 --version v004 --grid 5 \
      --frame-select phase_anchor --qids "$D/qids_${TAG#eval_expI02_}.txt" \
      --adapter "$AD" --out-tag "${TAG}.part"
    rc=$?; step "INFER $TAG rc=$rc"; [ "$rc" != "0" ] && return 1
  fi
  step "SPLICE $TAG"
  "$PY" "$D/splice_responses.py" "$PART" "$CTRL" "$R/$TAG"
  step "JUDGE $TAG"
  stdbuf -oL -eL "$PY" "$D/eval_seg.py" "$R/$TAG" --track PROCEDURE
  step "COMPARE ctrl vs $TAG"
  stdbuf -oL -eL "$PY" "$D/compare_runs.py" "$CTRL" "$R/$TAG" || true
}

run_arm eval_expI02_A_pureanchor PHASE_ANCHOR_USE_POSITION=0
run_arm eval_expI02_B_share025   PHASE_UNIFORM_SHARE=0.25
step "DONE"
