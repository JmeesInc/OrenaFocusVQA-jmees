#!/usr/bin/env bash
# torchrun の各ランクを「自分の GPU だけ見える」状態にして起動するラッパ。
#
# ★★DDP では **他ランクの CUDA コンテキストが rank0 の GPU に 386MiB ずつ居座る**。
#   4ランクなら rank0 だけ **~1.16GiB 損**をする（2026-09-02 expN01 Stage B で実測）。
#   居候は「物理GPU0」ではなく **「rank0 が使う GPU」** に付くので、
#   CUDA_VISIBLE_DEVICES の並べ替えでは避けられない（実測で確認済み）。
#   発生源は import 連鎖・モデルロード・all_reduce・NCCL P2P のいずれでもなく特定できなかったが、
#   **他の GPU を見せなければ物理的に作れない**ので、ここで塞ぐ。
#
# 使い方:
#   ORIG_CVD=0,1,2,3 torchrun --nproc_per_node=4 ... ddp_wrap.sh python train.py --config ...
#   （torchrun の entrypoint をこのスクリプトにして、後ろに本来のコマンドを置く）
#
# ★LOCAL_RANK を 0 に上書きするのが要点: 各プロセスから見える GPU は1枚なので index は必ず 0。
#   HF Trainer / accelerate は LOCAL_RANK でデバイスを決めるため、これを直さないと
#   rank3 が cuda:3 を掴もうとして落ちる。RANK / WORLD_SIZE は触らないので NCCL は無傷。
set -u
: "${ORIG_CVD:?ORIG_CVD に元の CUDA_VISIBLE_DEVICES を渡すこと}"
IFS=',' read -ra _DEVS <<< "$ORIG_CVD"
_LR="${LOCAL_RANK:-0}"
export CUDA_VISIBLE_DEVICES="${_DEVS[$_LR]}"
export LOCAL_RANK=0
echo "[ddp_wrap] RANK=${RANK:-?} 元LOCAL_RANK=$_LR → CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES LOCAL_RANK=0" >&2
exec "$@"
