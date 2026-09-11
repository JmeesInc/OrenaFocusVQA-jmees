#!/usr/bin/env bash
# GPU3 レーン: val 重畳(shard2) の完了を待つ → train DUAL arm の重畳(shard0/2) → スクリーニング3本
# ★PID 直指定で待つ（pgrep -f は自分自身にマッチする）。
set -uo pipefail
cd "$(dirname "$0")"
WAIT_PID=${1:-}
[ -n "$WAIT_PID" ] && while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
echo "=== $(date '+%F %T') GPU3: train 重畳レンダ開始 ==="
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=/data4/src/shunsuke/MICCAI2026/Orena/reference/src \
  ../../.venv-dl1/bin/python render_cache.py --part train --arm DUAL \
    --variants r0 r1 r2 r3 --shard 0/2
echo "=== $(date '+%F %T') GPU3: train レンダ rc=$? → スクリーニング開始 ==="
GPU=3 bash run_screen.sh r0 r2 t1
echo "=== $(date '+%F %T') GPU3 レーン完了 ==="
