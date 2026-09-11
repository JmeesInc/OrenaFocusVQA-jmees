#!/usr/bin/env bash
# expN04 本走（DDP 4GPU / ddp_wrap でランク分離 / flash-attn）
# ★bs=1 固定。bs=2 は 48GB でも OOM（2026-09-04 実測: 10.56GiB 要求/空き9.64GiB）。
#   系列長 p50 975 / max 5709 と 5.9倍ばらつき、左パディングで長い方に揃うため。
set -uo pipefail
cd /workspace/Orena
export PYTHONPATH="$PWD/reference/src"
export ORIG_CVD=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME="$PWD/.hf" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
LOG="$PWD/expN04_train.log"
[ -f "$LOG" ] && { echo "!!! $LOG が既にある。中止"; exit 1; }
setsid nohup bash -c "
  stdbuf -oL -eL \"$PWD/.venv/bin/python\" -m torch.distributed.run --nproc_per_node=4 --master_port=29601 \
    --no-python \"$PWD/workspace/expE01_segproc_baseline/ddp_wrap.sh\" \
    \"$PWD/.venv/bin/python\" \
    \"$PWD/workspace/expM03_round1_external/train_lora_ext.py\" \
    --config \"$PWD/workspace/expN04_seg_framerouter/config_N04_framerouter.yaml\" > \"$LOG\" 2>&1
  echo \"=== expN04 rc=\$? \$(date) ===\" >> \"$LOG\"
" > /dev/null 2>&1 < /dev/null &
echo "本走 起動 pid=$!  log=$LOG"
