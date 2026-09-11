#!/usr/bin/env bash
# expE01 — SEGMENT / PROCEDURE の学習・推論・評価をこのスクリプト経由で回す。
#
# ★GPU は dl2 の 4090 のみ。dl1 は torch(cu130) に対しドライバが 560(CUDA12.6) で
#   `torch.cuda.is_available()==False` になる（nvidia-smi は GPU を出すので気づきにくい）。
#   dl1 は ffmpeg フレーム抽出（CPU/IO）専用。
#
# 使い方:
#   bash run.sh extract SEGMENT 0 val 16 448        # フレーム抽出（dl1 でよい）
#   bash run.sh train   config_seg_16f.yaml         # 学習（dl2）
#   bash run.sh infer   SEGMENT 0 16 448 <out-tag> [adapter]
#   bash run.sh eval    <run_dir> SEGMENT
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
export PYTHONPATH="$ROOT/reference/src"
PY="$ROOT/.venv/bin/python"
D="$ROOT/workspace/expE01_segproc_baseline"

# 4090 を指定。PCI_BUS_ID を付けないと nvidia-smi の index と一致しない。
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

cmd="${1:?extract|train|infer|eval}"; shift
case "$cmd" in
  extract)
    track="${1:-SEGMENT}"; fold="${2:-0}"; part="${3:-val}"
    nf="${4:-16}"; sz="${5:-448}"; lim="${6:-0}"
    "$PY" "$D/dataset_seg.py" --track "$track" --fold "$fold" --part "$part" \
      --n-frames "$nf" --size "$sz" --limit "$lim" --workers 32
    ;;
  train)
    cfg="${1:-$D/config_seg_16f.yaml}"; shift || true
    "$PY" "$D/train_lora_seg.py" --config "$cfg" "$@"
    ;;
  infer)
    track="${1:-SEGMENT}"; fold="${2:-0}"; nf="${3:-16}"; sz="${4:-448}"
    tag="${5:?out-tag}"; adapter="${6:-}"
    "$PY" "$D/run_infer.py" --track "$track" --fold "$fold" --n-frames "$nf" \
      --size "$sz" --out-tag "$tag" ${adapter:+--adapter "$adapter"}
    ;;
  eval)
    run_dir="${1:?run_dir}"; track="${2:-SEGMENT}"; shift 2 || true
    "$PY" "$D/eval_seg.py" "$run_dir" --track "$track" "$@"
    ;;
  *) echo "unknown: $cmd" >&2; exit 1 ;;
esac
