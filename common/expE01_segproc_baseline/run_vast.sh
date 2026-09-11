#!/usr/bin/env bash
# 貸しGPU（RTX 6000Ada x4）でアームを1本起動する。
# Usage: GPU=0 ARM=V01 bash run_vast.sh
# ★smoke を必ず先に通す（config の形は本番と同一のまま --train-limit で件数だけ絞る）。
set -uo pipefail
cd /workspace/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PY=/opt/conda/bin/python
D="$PWD/workspace/expE01_segproc_baseline"
step(){ echo "########## $(date '+%F %T') $* ##########"; }
case "${ARM:?ARM を指定}" in
  V01) CFG=$D/config_V01_fs_aug_seg.yaml ;;
  V02) CFG=$D/config_V02_fs_27b.yaml ;;
  V03) CFG=$D/config_V03_fs_fix64f560.yaml ;;
  V04) CFG=$D/config_V04_frame_visionrank.yaml ;;
  V05) CFG=$D/config_V05_frame_1024px.yaml ;;
  V06) CFG=$D/config_V06_frame_r32.yaml ;;
  V07) CFG=$D/config_V07_seg_ep2.yaml ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac
SMOKE=/tmp/smoke_$ARM.yaml
sed -e 's/^  name: \(.*\)$/  name: SMOKE_\1/' -e 's/eval_subset: [0-9]*/eval_subset: 4/' \
    -e 's/save_steps: [0-9]*/save_steps: 2/' -e 's/eval_steps: [0-9]*/eval_steps: 2/' \
    -e 's/grad_accum: [0-9]*/grad_accum: 2/' "$CFG" > "$SMOKE"
step "SMOKE $ARM"
stdbuf -oL -eL $PY "$D/train_lora_seg.py" --config "$SMOKE" --epochs 1 --train-limit 8
rc=$?; step "SMOKE $ARM rc=$rc"
[ "$rc" != "0" ] && { echo "!!! smoke 失敗。本番は流さない"; exit 1; }
step "TRAIN $ARM 本番"
stdbuf -oL -eL $PY "$D/train_lora_seg.py" --config "$CFG"
step "TRAIN $ARM rc=$?"
