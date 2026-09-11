#!/usr/bin/env bash
# GPU0: val を conf 0.25 で再レンダ → r1@0.25 アームを推論 → judge
# ★r3（2段conf）が r0 に負けたので、「conf を下げて拾った分は効いていない」可能性がある。
#   ただし r3 は描き方も変えているので、**同じ描き方(r1)でしきい値だけ 0.5→0.25** を測る。
set -uo pipefail
cd "$(dirname "$0")"
R=/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$R/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=$R/.venv-dl1/bin/python
AD=$R/workspace/expE01_segproc_baseline/results/expK01F_frame_pseudo/fold0/adapter
step(){ echo "=== $(date '+%F %T') $* ==="; }

step "val を conf 0.25 で再レンダ（cache25）"
$PY render_cache.py --part val --variants r0 r1 --conf 0.25 --conf-lo 0.25 \
    --cache-root "$PWD/cache25"
step "レンダ rc=$?"

step "INFER r1@conf0.25"
$PY screen_infer.py --adapter "$AD" --variant r1 --cache-root "$PWD/cache25" \
    --out-tag screen_r1c25
step "INFER rc=$?"

step "JUDGE r1@conf0.25"
$PY "$R/workspace/expE01_segproc_baseline/eval_seg.py" results/screen_r1c25 --track FRAME
step "JUDGE rc=$?  LANE DONE"
