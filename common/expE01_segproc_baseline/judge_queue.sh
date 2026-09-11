#!/usr/bin/env bash
# 採点キュー: results/ を巡回し、「推論は完走(meta.json あり)だが未採点(results_merged.csv 無し)」
# の run を見つけ次第 judge にかける。**A4000(idx0)** で回すので 4090 の推論・学習を止めない。
#
# 4090 側のチェーンが次々に run を吐くので、こちらは待ち受けにしておけば人手が要らない。
# 使い方: setsid nohup bash judge_queue.sh SEGMENT > log 2>&1 < /dev/null &
set -uo pipefail
cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
PY="$PWD/.venv/bin/python"
D="$PWD/workspace/expE01_segproc_baseline"
TRACK="${1:-SEGMENT}"
IDLE_LIMIT="${2:-240}"      # 何回空振りしたら終了するか（240×30s = 2時間）

idle=0
while [ "$idle" -lt "$IDLE_LIMIT" ]; do
  found=0
  for d in "$D"/results/*/; do
    [ -f "$d/meta.json" ] || continue                 # 推論が完走していない
    [ -f "$d/results_merged.csv" ] && continue        # 採点済み
    [ -f "$d/.judging" ] && continue                  # 別プロセスが処理中
    touch "$d/.judging"
    # ★track は **meta.json から読む**。固定の --track を渡すと、PROCEDURE の run を
    #   SEGMENT として採点して突合0件になる（qID が別トラックの集合なので静かに壊れる）。
    t=$(grep -o '"track"[^,]*' "$d/meta.json" | head -1 | grep -oE 'FRAME|SEGMENT|PROCEDURE')
    t="${t:-$TRACK}"
    echo "=== $(date +%T) judge $(basename "$d") [$t] ==="
    "$PY" "$D/eval_seg.py" "$d" --track "$t"
    rc=$?   # ★直後に単独行で捕まえる（$(date)/$(basename) が $? を上書きするため）
    echo "=== $(date +%T) judge $(basename "$d") rc=$rc ==="
    rm -f "$d/.judging"
    found=1
  done
  if [ "$found" = "1" ]; then idle=0; else idle=$((idle+1)); sleep 30; fi
done
echo "=== $(date +%T) judge queue: $((IDLE_LIMIT*30/60))分 新規なし。終了 ==="
