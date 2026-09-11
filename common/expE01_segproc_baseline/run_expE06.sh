#!/usr/bin/env bash
# expE06: FRAME 全量 + SEGMENT 全量（**PROCEDURE なし**）。expE03f の唯一の正しい対照。
# 差は PROCEDURE の有無だけ（fold v004 / 448px / 全量 / 1ep は同一）。推定 41h。
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"; D="$PWD/workspace/expE01_segproc_baseline"
step(){ echo "=== $(date '+%F %T') $* ==="; }
CFG=$D/config_G_joint_frame_segment_FULL.yaml
SMOKE=/tmp/smoke_E06.yaml
# ★limit: null は書き換えない（本番と同じ経路を通す）。件数は --train-limit で絞る
sed -e 's/^  name: \(.*\)$/  name: SMOKE_\1/' -e 's/eval_subset: [0-9]*/eval_subset: 4/' \
    -e 's/save_steps: [0-9]*/save_steps: 2/' -e 's/eval_steps: [0-9]*/eval_steps: 2/' \
    -e 's/grad_accum: [0-9]*/grad_accum: 2/' "$CFG" > "$SMOKE"
step "SMOKE expE06"
"$PY" "$D/train_lora_seg.py" --config "$SMOKE" --epochs 1 --train-limit 8
rc=$?; step "SMOKE rc=$rc"; [ "$rc" != "0" ] && { echo "!!! smoke 失敗"; exit 1; }
step "TRAIN expE06 (FRAME 15,992 + SEGMENT 15,994, 推定41h)"
"$PY" "$D/train_lora_seg.py" --config "$CFG"
step "TRAIN expE06 rc=$?"
