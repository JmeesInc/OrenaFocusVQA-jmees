#!/usr/bin/env bash
# expI03 — FO 検出索引によるフレーム推薦の A/B
#
#   ctrl : eval_expI01_uniform_64fproc448_kf5_n2000（既取得、同一 GPU・同一設定）
#   fo   : --frame-select fo（15s 刻みの検出索引、thr0.3 の二値マスクへ予算 75%）
#
# ★`time` 形式にだけ適用する（expI01 で網羅が要る形式は集中させると壊れると確認済み）。
#   変化するのは 644問だけなので、そこだけ推論して残りは ctrl から流用（厳密に同一）。
# ★expI02 の完走を待ってから走る（4090 は 1 プロセスしか載らない）。
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
TAG="eval_expI03_fo_64fproc448_kf5_n2000"
step(){ echo "=== $(date '+%F %T') $* ==="; }

step "WAIT: expI02 の完走を待つ"
while ! grep -q '=== .* DONE ===' "$D/phase_anchor_ab.log" 2>/dev/null; do sleep 120; done
step "WAIT done"

if [ ! -f "$R/${TAG}.part/responses.json" ]; then
  step "INFER fo (644問のみ)"
  stdbuf -oL -eL "$PY" "$D/run_infer.py" --track PROCEDURE --fold 0 --part val \
    --limit 2000 --n-frames 64 --size 448 --version v004 --grid 5 \
    --frame-select fo --qids "$D/qids_fo.txt" --adapter "$AD" --out-tag "${TAG}.part"
  rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && exit 1
fi
step "SPLICE"
"$PY" "$D/splice_responses.py" "$R/${TAG}.part" "$CTRL" "$R/$TAG"
step "JUDGE"
stdbuf -oL -eL "$PY" "$D/eval_seg.py" "$R/$TAG" --track PROCEDURE
step "COMPARE ctrl vs fo (matched, McNemar)"
stdbuf -oL -eL "$PY" "$D/compare_runs.py" "$CTRL" "$R/$TAG" || true
step "DONE"
