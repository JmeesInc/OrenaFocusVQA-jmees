#!/usr/bin/env bash
# SEGMENT 全量（15,994問）の学習。方針変更（2026-08-08）により **6000/3000 の縮小版は今後作らない**。
#
#   A-full: scratch                    （config_A_full_seg_scratch.yaml）
#   B-full: expD11(FRAME 16,000) から warm start（config_B_full_seg_from_framelora.yaml）
#
# ★どちらも **qa_v005** に揃える。B は expD11（fold v001 学習）由来なので v005 以外だと汚染し、
#   A も同じ fold にしないと B と比較できない。
# ★smoke は `--train-limit` で件数だけ絞り、**config の形（train_limit: null）は本番と同一に保つ**
#   （2026-08-07 の実害: smoke が limit を書き換えて壊れた経路を隠し、本番が 1.5h 後に落ちた）。
#
# Usage: GPU=1 run_seg_full.sh <A|B>
set -uo pipefail
cd /data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv-dl1/bin/python"
D="$PWD/workspace/expE01_segproc_baseline"
step(){ echo "=== $(date '+%F %T') $* ==="; }

case "${1:-}" in
  A) CFG=$D/config_A_full_seg_scratch.yaml ;;
  B) CFG=$D/config_B_full_seg_from_framelora.yaml ;;
  *) echo "usage: GPU=N $0 <A|B>"; exit 2 ;;
esac

SMOKE=/tmp/smoke_segfull_$1.yaml
sed -e 's/^  name: \(.*\)$/  name: SMOKE_\1/' \
    -e 's/eval_subset: [0-9]*/eval_subset: 4/' \
    -e 's/save_steps: [0-9]*/save_steps: 2/' \
    -e 's/eval_steps: [0-9]*/eval_steps: 2/' \
    -e 's/grad_accum: [0-9]*/grad_accum: 2/' "$CFG" > "$SMOKE"
step "SMOKE $1 (train_limit で8件に絞る。config の形は本番と同一)"
stdbuf -oL -eL "$PY" "$D/train_lora_seg.py" --config "$SMOKE" --epochs 1 --train-limit 8
rc=$?; step "SMOKE $1 rc=$rc"
[ "$rc" != "0" ] && { echo "!!! smoke 失敗。本番は流さない"; exit 1; }

step "TRAIN $1 全量 15,994問（RTX8000 で推定 70〜80h）"
stdbuf -oL -eL "$PY" "$D/train_lora_seg.py" --config "$CFG"
step "TRAIN $1 rc=$?"
