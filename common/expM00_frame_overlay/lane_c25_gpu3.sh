#!/usr/bin/env bash
# GPU3: train DUAL arm を conf 0.25 で再レンダ（どちらのしきい値に決まっても使う）
set -uo pipefail
cd "$(dirname "$0")"
R=/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$R/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=$R/.venv-dl1/bin/python
echo "=== $(date '+%F %T') train を conf 0.25 で再レンダ（cache25）==="
$PY render_cache.py --part train --arm DUAL --variants r0 r1 --conf 0.25 --conf-lo 0.25 \
    --cache-root "$PWD/cache25"
echo "=== $(date '+%F %T') rc=$? LANE DONE ==="
