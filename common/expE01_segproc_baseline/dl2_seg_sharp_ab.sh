#!/usr/bin/env bash
# SEGMENT: 「匿名回避 + 鮮明フレーム選択」の A/B（dl2 の RTX 4090 で実行）
#
# ★3アーム。sharp と anon_note は**別機構**なので寄与を分離する:
#     A  control     : uniform                （対照）
#     B  +anon_note  : uniform + 匿名区間を本文で明示（プロンプトだけ変わる）
#     C  +sharp      : B に加えて各スロットを非匿名・最鮮明な秒へ置換（入力画像が変わる）
#
# ★対照(A)も **同じ 4090・bf16 で取り直す**。既存の 0.7197 は dl1・fp16 の値で、
#   そのまま比べると「機材 + dtype + 施策」が交絡する（判定は greedy なので
#   同一機材同一 dtype なら再現する）。
#
# ★入力側は検証済み（2026-08-14）:
#     枚数保存 30.66 → 30.66 / アンカー消失 0問 / 鮮明度 median 775→931
#     ベタ塗りフレーム 3.08% → 2.45%（実画素で計数）
# ★smoke 実測: 6.38 s/q, 11,896 tok, latency max 6.91s（SEGMENT 上限 15s に余裕）
#   → 1アーム 2000問で約 3.7h。モデルロードは NFS 越しで ~11 分かかる。
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"; D="$PWD/workspace/expE01_segproc_baseline"; R="$D/results"
AD="$R/expE06g_joint_frame_segment_FULL/fold0/adapter"
step(){ echo "########## $(date '+%F %T') $* ##########"; }

run_one () {  # $1=タグ $2...=run_infer への追加引数
  local TAG="$1"; shift
  if [ -f "$R/$TAG/results_merged.csv" ]; then step "SKIP $TAG（判定済み）"; return 0; fi
  if [ ! -f "$R/$TAG/responses.json" ]; then
    step "INFER $TAG  [$*]"
    "$PY" "$D/run_infer.py" --track SEGMENT --fold 0 --part val --limit 2000 \
      --version v004 --n-frames 32 --size 768 --anchor \
      --adapter "$AD" --out-tag "$TAG" "$@"
    local rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && return 1
  else
    step "SKIP INFER $TAG（responses.json あり）"
  fi
  "$PY" "$D/eval_seg.py" "$R/$TAG" --track SEGMENT; step "JUDGE rc=$?"
}

# A: 対照（4090/bf16 で取り直す）
run_one "eval_dl2_seg32f768_ctrl_n2000"
# B: 匿名区間を本文で明示するだけ（入力画像は A と1枚も変わらない）
run_one "eval_dl2_seg32f768_anonnote_n2000"   --anon-note
# C: B + 非匿名・最鮮明な秒へ置換
run_one "eval_dl2_seg32f768_sharp_n2000"      --anon-note --frame-select sharp

# ★compare_runs.py は2 run 固定なので、寄与を分けるため3通りとも取る
step "COMPARE ctrl vs anon_note（プロンプトだけの寄与）"
"$PY" "$D/compare_runs.py" "$R/eval_dl2_seg32f768_ctrl_n2000" \
  "$R/eval_dl2_seg32f768_anonnote_n2000" --labels ctrl anon_note
step "COMPARE anon_note vs sharp（フレーム置換の上乗せ分）"
"$PY" "$D/compare_runs.py" "$R/eval_dl2_seg32f768_anonnote_n2000" \
  "$R/eval_dl2_seg32f768_sharp_n2000" --labels anon_note sharp
step "COMPARE ctrl vs sharp（提案の合計効果）"
"$PY" "$D/compare_runs.py" "$R/eval_dl2_seg32f768_ctrl_n2000" \
  "$R/eval_dl2_seg32f768_sharp_n2000" --labels ctrl sharp
step "QUEUE DONE"
