#!/usr/bin/env bash
# 貸しGPU で **DDP 4GPU** で1アームを回す。
# ★単体GPU 実測 133 s/it（16サンプル/step）= 1 epoch 74時間・$146 で枠に入らないため。
#   DDP なら約 25時間・$50 の見込み。
# ★config 側で grad_accum を GPU 数で割ってあること（有効バッチを 16 に保つ）。
# Usage: ARM=V01 bash run_vast_ddp.sh
set -uo pipefail
cd /workspace/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_DATASETS_OFFLINE=1
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
step "TRAIN $ARM (DDP 4GPU)"
stdbuf -oL -eL $PY -m torch.distributed.run --nproc_per_node=4 --master_port=29531 \
  "$D/train_lora_seg.py" --config "$CFG"
step "TRAIN $ARM rc=$?"
