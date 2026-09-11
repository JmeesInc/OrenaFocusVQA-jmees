#!/usr/bin/env bash
# SEGMENT のサンプリング戦略検証を無人で順に流す（dl2 4090）。
#
# ★先行ジョブ（刻み2s）の終了を待ってから開始する。GPU 空き容量で判定するので、
#   他ユーザが取った場合も待つ。
#
# 対照は全て `eval_expE06g_seg64f448_n2000`（64f 固定, SCORE 0.6855, latency 3.94s/問）。
# モデルは expE06g（F+S joint 全量）で固定＝**入力の作り方だけを変えた matched 比較**。
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"; D="$PWD/workspace/expE01_segproc_baseline"; R="$D/results"
AD="$R/expE06g_joint_frame_segment_FULL/fold0/adapter"
BASE="$R/eval_expE06g_seg64f448_n2000"
step(){ echo "########## $(date '+%F %T') $* ##########"; }

step "先行ジョブの終了待ち（GPU2 の空きが 20GB 超になるまで）"
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2)" -gt 20000 ]; do sleep 120; done
step "GPU2 空き確認"

run_one () {  # $1=タグ接尾 $2...=run_infer への追加引数
  local suf="$1"; shift
  local TAG="eval_expE06g_seg${suf}_n2000"
  if [ -f "$R/$TAG/results_merged.csv" ]; then step "SKIP $TAG（判定済み）"; return 0; fi
  step "INFER $TAG  [$*]"
  "$PY" "$D/run_infer.py" --track SEGMENT --fold 0 --part val --limit 2000 \
    --version v004 --adapter "$AD" --out-tag "$TAG" "$@"
  local rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && return 1
  "$PY" "$D/eval_seg.py" "$R/$TAG" --track SEGMENT; step "JUDGE rc=$?"
  "$PY" "$D/compare_runs.py" "$BASE" "$R/$TAG" --labels 64f "$suf"
  step "COMPARE rc=$?"
}

# ① アンカー: 問題文中の hh:mm:ss のフレームを必ず含める（枚数は 64 のまま）
run_one "64f448_anchor" --n-frames 64 --size 448 --anchor

# ② 可変解像度: 刻み2s で枚数を決め、枚数が少ない問だけ 768px に上げる
run_one "stride2_adasize" --n-frames 96 --size 448 --stride 2 --n-min 8 --adaptive-size 448 768

step "QUEUE DONE"
