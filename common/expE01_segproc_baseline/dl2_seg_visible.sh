#!/usr/bin/env bash
# SEGMENT: 「匿名秒を候補から除去してから一様に選定」(`--frame-select visible`) の1本。
#
# ★対照は **既存の `eval_dl2_seg32f768_ctrl_n2000`（0.7197）をそのまま使う**。
#   greedy 推論は同一 config・同一機材で bit 単位に再現することを 2026-08-15 に実測済み
#   （再実行と content 2000/2000 一致）。対照の取り直しは 3.2h の無駄になる。
#
# ★arm C (`sharp`) との違い（2026-08-15 に実装し直した）:
#     sharp   = 一様に打ってから各スロットを近傍で置換 → 重複58.5%・寄り合い・穴7.6s → ❌−0.0222
#     visible = **先に匿名秒を候補から落とし**、候補列のインデックス上で等間隔に取る
#               → 重複0 / 最小間隔3.402s(uniform 3.450) / ベタ塗り **0.000%**
#
# ★入力側は検証済み（`len(set(...))` で確認。前回は `len()` で測って重複を見逃した）:
#     匿名ゼロの1622問は uniform と **完全一致（ズレ0.000s）** ＝ 介入が clean に分離される
#     差が出るのは匿名がある378問だけ。アンカー消失は5問（GT が匿名区間内で入れようがない）
# ★事前予測（GPU不要の指標。arm C の失敗を後付けで当てた）:
#     GT 許容内にフレームがある割合 uniform 96.4% → sharp 92.4%(❌) → **visible 96.6%**
#     匿名あり問では 97.9% → 88.3%(❌) → **99.3%**
#   → 安全だが**改善幅は小さい**。全体 SCORE ではなく **378問の部分集合で McNemar** を見る。
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"; D="$PWD/workspace/expE01_segproc_baseline"; R="$D/results"
AD="$R/expE06g_joint_frame_segment_FULL/fold0/adapter"
CTRL="$R/eval_dl2_seg32f768_ctrl_n2000"
TAG="eval_dl2_seg32f768_visible_n2000"
step(){ echo "########## $(date '+%F %T') $* ##########"; }

if [ ! -f "$R/$TAG/responses.json" ]; then
  step "INFER $TAG"
  "$PY" "$D/run_infer.py" --track SEGMENT --fold 0 --part val --limit 2000 \
    --version v004 --n-frames 32 --size 768 --anchor \
    --frame-select visible --adapter "$AD" --out-tag "$TAG"
  step "INFER rc=$?"
else
  step "SKIP INFER（responses.json あり）"
fi

if [ ! -f "$R/$TAG/results_merged.csv" ]; then
  "$PY" "$D/eval_seg.py" "$R/$TAG" --track SEGMENT; step "JUDGE rc=$?"
fi

step "COMPARE ctrl vs visible（全体）"
"$PY" "$D/compare_runs.py" "$CTRL" "$R/$TAG" --labels ctrl visible
step "COMPARE rc=$?"

# ★変化が起きうるのは匿名がある問だけ。全体では埋もれるので部分集合で必ず見る。
step "SUBSET 匿名あり問だけの matched 比較"
"$PY" "$D/subset_compare.py" "$CTRL" "$R/$TAG" --labels ctrl visible
step "SUBSET rc=$?"
step "DONE"
