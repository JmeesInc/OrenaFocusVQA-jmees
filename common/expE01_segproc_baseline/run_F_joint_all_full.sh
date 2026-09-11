#!/usr/bin/env bash
# expE03f: 使えるデータ全量の joint 学習（FRAME 15,992 + SEGMENT 15,994 + PROCEDURE 8,000）。
# ★59h の本番前に必ず smoke。`limit: null` と PROCEDURE 16f の経路は一度も通していない。
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2   # 4090
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True      # joint は系列長が可変で断片化しやすい
export PYTHONUNBUFFERED=1
PY="$PWD/.venv/bin/python"
D="$PWD/workspace/expE01_segproc_baseline"
step(){ echo "=== $(date '+%F %T') $* ==="; }

CFG=$D/config_F_joint_all_full.yaml
SMOKE=/tmp/smoke_F_joint_all_full.yaml
# ★`limit: null` を書き換えないこと。
#   2026-08-07 の実害: smoke で `limit: null` → `limit: 6` に置換していたため、
#   **壊れていた経路（null での比率計算）を smoke が 潰したまま通過**し、
#   本番が 1.5h かけて 39,986 件を構築した直後に TypeError で落ちた。
#   件数の削減は `--train-limit` で行い、**config の形は本番と同一に保つ**。
sed -e 's/^  name: \(.*\)$/  name: SMOKE_\1/' \
    -e 's/eval_subset: 300/eval_subset: 4/' \
    -e 's/save_steps: 200/save_steps: 2/' \
    -e 's/eval_steps: 100/eval_steps: 2/' \
    -e 's/grad_accum: 16/grad_accum: 2/' "$CFG" > "$SMOKE"
step "SMOKE (limit: null のまま = 本番と同じ経路。件数は --train-limit で絞る)"
"$PY" "$D/train_lora_seg.py" --config "$SMOKE" --epochs 1 --train-limit 8
rc=$?; step "SMOKE rc=$rc"
if [ "$rc" != "0" ]; then echo "!!! smoke 失敗。本番は流さない"; exit 1; fi

step "TRAIN expE03f (全量 39,986 問, 推定 59h)"
"$PY" "$D/train_lora_seg.py" --config "$CFG"
step "TRAIN expE03f rc=$?"
