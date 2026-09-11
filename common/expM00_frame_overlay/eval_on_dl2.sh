#!/usr/bin/env bash
# dl1 が別セッションで埋まっているときに dl2 で評価を回す。
#   NAME=... ADAPTER=... VARIANT=... CACHE=... bash eval_on_dl2.sh
# ★dl2 は /mnt/data/data4 経由で同じリポジトリを見る（跨マシン symlink は /mnt/data/data4）。
set -uo pipefail
R=/mnt/data/data4/src/shunsuke/MICCAI2026/Orena
NAME="${NAME:?}"; ADAPTER="${ADAPTER:?}"; VARIANT="${VARIANT:?}"; CACHE="${CACHE:-cache}"
GPU="${GPU:-2}"
ssh dl2 "cd $R/workspace/expM00_frame_overlay && REPO=$R \
  NAME=$NAME ADAPTER='$ADAPTER' VARIANT=$VARIANT CACHE=$CACHE GPUS='$GPU' \
  PYBIN=$R/.venv/bin/python bash run_eval_model.sh"
echo "=== dl2 eval rc=$? ==="
