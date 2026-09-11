#!/usr/bin/env bash
# expR00 を **4×RTX6000Ada DDP** で学習する（vast 側で実行）。
#   bash workspace/expR00_seg_s3/train_ddp_r00.sh [nproc] [追加引数...]
#   例) smoke: bash .../train_ddp_r00.sh 1 --train-limit 24
#       本走 : bash .../train_ddp_r00.sh 4
#
# ★★`grad_accum` は config で **4**。1 step のサンプル数 =
#     per_device_bs(1) × grad_accum(4) × GPU数(4) = 16 で単体実行(accum 16)と同一。
#   **16 のままだと有効バッチ 64 になって別物**（根拠 chain_v07_ddp.sh）。
#   ⚠️nproc を 4 以外にするなら config の grad_accum も合わせて変えること。
# ★★[[ddp-ranks-squat-on-gpu0]] **ddp_wrap.sh は必須**。無いと他ランクのコンテキストが
#   386MiB×3 rank0 の GPU に居座って OOM する。
#   ⚠️`torchrun ... ddp_wrap.sh python ...` は torchrun が entrypoint を Python として
#     実行しようとするので **`--no-python` が要る**。
# ★★[[detached-remote-training]] **setsid + nohup で切り離す**。素の ssh 実行は
#   セッション終了で道連れ kill される（train_ddp_p00c.sh には入っていない）。
set -uo pipefail
REPO=/workspace/Orena; cd "$REPO"
NP="${1:-4}"; shift || true
export PYTHONPATH="$REPO/reference/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME="$REPO/.hf" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3
export ORIG_CVD=0,1,2,3          # ddp_wrap.sh が rank ごとに1枚だけ見せるのに使う
CFG="$REPO/workspace/expR00_seg_s3/config_R00_seg_s3_r64.yaml"
WRAP="$REPO/workspace/expE01_segproc_baseline/ddp_wrap.sh"
chmod +x "$WRAP" 2>/dev/null

# smoke と本走でログを分ける。★同名ログが在れば中止（上書きで前回の証拠を消さない）
case " $* " in *" --train-limit "*) LOG="$REPO/expR00_smoke.log";; *) LOG="$REPO/expR00_train.log";; esac
[ -f "$LOG" ] && { echo "!!! $LOG が既にある。中止（消すか名前を変えてから）"; exit 1; }

echo "########## $(date '+%F %T') expR00 DDP 学習開始 nproc=$NP log=$LOG ##########"
grep -E "^  name:|train_part|n_frames|anchor|two_way|expect_drawn|  r:|alpha:|lr:|grad_accum|per_device|max_seq_len|eval_steps" "$CFG"

setsid nohup bash -c "
  stdbuf -oL -eL '$REPO/.venv/bin/python' -m torch.distributed.run \
    --nproc_per_node=$NP --master_port=29533 --no-python \
    '$WRAP' '$REPO/.venv/bin/python' \
    '$REPO/workspace/expM03_round1_external/train_lora_ext.py' --config '$CFG' $* > '$LOG' 2>&1
  echo \"########## expR00 rc=\$? \$(date '+%F %T') ##########\" >> '$LOG'
" > /dev/null 2>&1 < /dev/null &
echo "起動 pid=$!  log=$LOG"
