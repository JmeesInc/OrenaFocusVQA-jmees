#!/usr/bin/env bash
# GPU0 レーン: val 重畳(shard1) の完了を待つ → train DUAL arm の重畳(shard1/2) → スクリーニング3本
set -uo pipefail
cd "$(dirname "$0")"
WAIT_PID=${1:-}
[ -n "$WAIT_PID" ] && while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "=== $(date '+%F %T') GPU0: train 重畳レンダ開始 ==="
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/data4/src/shunsuke/MICCAI2026/Orena/reference/src \
  ../../.venv-dl1/bin/python render_cache.py --part train --arm DUAL \
    --variants r0 r1 r2 r3 --shard 1/2
echo "=== $(date '+%F %T') GPU0: train レンダ rc=$? → スクリーニング開始 ==="
GPU=0 bash run_screen.sh control r1 r3
echo "=== $(date '+%F %T') GPU0 レーン完了 ==="
