#!/usr/bin/env bash
# GPU0（やり直し）: val を conf 0.25 で再レンダ → r1@0.25 推論 → judge
# ★前回は val と train を同時に shard なしで回して index.json を奪い合い、val 側が消えた。
#   render_cache.py の flush 名を index.<part>[.shard].json に直した上でやり直す。
set -uo pipefail
cd "$(dirname "$0")"
R=/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$R/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=$R/.venv-dl1/bin/python
AD=$R/workspace/expE01_segproc_baseline/results/expK01F_frame_pseudo/fold0/adapter
step(){ echo "=== $(date '+%F %T') $* ==="; }

step "val を conf 0.25 で再レンダ（cache25 / index.val.json）"
$PY render_cache.py --part val --variants r0 r1 --conf 0.25 --conf-lo 0.25 \
    --cache-root "$PWD/cache25"
rc=$?; step "レンダ rc=$rc"; [ "$rc" -ne 0 ] && exit 1

step "index の突合（val 3,087 + train 7,272 = 10,359 になっているか）"
$PY - <<'PY'
import json, glob
n = {}
for p in sorted(glob.glob('cache25/r1/index*.json')):
    d = json.load(open(p)); n[p] = len(d)
print(n, "合計", sum(n.values()))
assert sum(n.values()) == 10359, "index の件数が合わない"
PY
[ $? -ne 0 ] && { echo "★index が不整合。推論しない"; exit 1; }

step "INFER r1@conf0.25"
$PY screen_infer.py --adapter "$AD" --variant r1 --cache-root "$PWD/cache25" \
    --out-tag screen_r1c25
rc=$?; step "INFER rc=$rc"; [ "$rc" -ne 0 ] && exit 1

step "JUDGE r1@conf0.25"
$PY "$R/workspace/expE01_segproc_baseline/eval_seg.py" results/screen_r1c25 --track FRAME
step "JUDGE rc=$?  LANE DONE"
