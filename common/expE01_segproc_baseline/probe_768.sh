#!/usr/bin/env bash
# 16フレーム@768px の joint 学習が 4090(24GB) に載るかを実測する。
# 日報の「448px で回す」判断は **時間** が理由で、メモリは測っていなかった。
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
D="$PWD/workspace/expE01_segproc_baseline"
echo "=== $(date '+%F %T') PROBE 768px joint ==="
"$PWD/.venv/bin/python" "$D/train_lora_seg.py" --config "$D/config_PROBE_joint768.yaml" \
  --epochs 1 --train-limit 6
echo "=== PROBE rc=$? ==="
