#!/usr/bin/env bash
# expQ00-B（rank 16）を **4×4090 DDP** で回す（機体側で実行）。
#   bash train_ddp_q00b.sh [nproc] [追加引数（--train-limit 160 など）]
# ★A 版（r=64）とは別ファイルにしてある（[[never-edit-a-running-bash-script]]）。
#
# ベース: workspace/expP00_stageC_s3/train_ddp_p00c.sh を逐語で流用
#   （[[port-inference-call-verbatim]]: 動いている起動をそのまま写す）。
#
# ★★[[ddp-ranks-squat-on-gpu0]] `ddp_wrap.sh` は必須。無いと他ランクの CUDA コンテキストが
#   rank0 の GPU に 386MiB ずつ居座り、24GB の 4090 では OOM に直結する。
#   ⚠️ torchrun は entrypoint を Python として実行しようとするので **`--no-python` が要る**。
# ★★grad_accum は config 側で 16→4 にしてある（有効バッチ 16 を単体実行と一致させる）。
set -uo pipefail
REPO=/workspace/Orena; cd "$REPO"
NP="${1:-4}"
export PYTHONPATH="$REPO/reference/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME="$REPO/.hf" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3
export ORIG_CVD=0,1,2,3          # ddp_wrap.sh が rank ごとに1枚だけ見せるのに使う
CFG="$REPO/workspace/expQ00_frame_alldata/config_Q00B_frame_alldata_r16_ddp4.yaml"
WRAP="$REPO/workspace/expE01_segproc_baseline/ddp_wrap.sh"
chmod +x "$WRAP" 2>/dev/null

echo "########## $(date '+%F %T') expQ00-A DDP 学習開始 nproc=$NP ##########"
grep -E "name:|train_part|expect_dual|lr:|grad_accum|per_device|eval_steps|r:|alpha:" "$CFG"
stdbuf -oL -eL "$REPO/.venv/bin/python" -m torch.distributed.run \
  --nproc_per_node="$NP" --master_port=29543 --no-python \
  "$WRAP" "$REPO/.venv/bin/python" \
  "$REPO/workspace/expM03_round1_external/train_lora_ext.py" --config "$CFG" "${@:2}"
echo "########## $(date '+%F %T') rc=$? ##########"
