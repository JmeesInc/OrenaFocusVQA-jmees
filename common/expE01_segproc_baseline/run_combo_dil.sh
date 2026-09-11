#!/usr/bin/env bash
# expI05 — combo（FO索引 × アンカー工程区間）に**膨張 ±30s** を足した版。
# 検出点の前後を候補に含めることで、粗い走査で落ちた分を拾い直す。
# 実測（±5s 到達, N=633）: 刻み15s 48.7% → **52.4%**（膨張 ±30s）
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-2}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 FO_DILATE_S=30
PY="$PWD/.venv/bin/python"; D="$PWD/workspace/expE01_segproc_baseline"; R="$D/results"
CTRL="$R/eval_expI01_uniform_64fproc448_kf5_n2000"
TAG="eval_expI05_combo_dil30_64fproc448_kf5_n2000"
step(){ echo "=== $(date '+%F %T') $* ==="; }
step "WAIT: expI04 の完走を待つ"
while ! grep -q '=== .* DONE ===' "$D/combo_ab.log" 2>/dev/null; do sleep 120; done
step "INFER combo+膨張30s"
stdbuf -oL -eL "$PY" "$D/run_infer.py" --track PROCEDURE --fold 0 --part val \
  --limit 2000 --n-frames 64 --size 448 --version v004 --grid 5 \
  --frame-select combo --qids "$D/qids_combo_dil.txt" \
  --adapter "$R/expE03f_joint_all_full/fold0/adapter" --out-tag "${TAG}.part"
rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && exit 1
step "SPLICE"; "$PY" "$D/splice_responses.py" "$R/${TAG}.part" "$CTRL" "$R/$TAG"
step "JUDGE";  stdbuf -oL -eL "$PY" "$D/eval_seg.py" "$R/$TAG" --track PROCEDURE
step "COMPARE ctrl vs combo+膨張"; stdbuf -oL -eL "$PY" "$D/compare_runs.py" "$CTRL" "$R/$TAG" || true
step "COMPARE combo vs combo+膨張"
stdbuf -oL -eL "$PY" "$D/compare_runs.py" "$R/eval_expI04_combo_64fproc448_kf5_n2000" "$R/$TAG" || true
step "DONE"
