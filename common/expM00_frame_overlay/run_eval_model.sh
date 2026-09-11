#!/usr/bin/env bash
# 学習済み adapter を FRAME val 4,008 問で評価する（3分割 → 結合 → judge）。
#   NAME=eval_M00B ADAPTER=<path> VARIANT=control ./run_eval_model.sh
#   NAME=eval_M00A ADAPTER=<path> VARIANT=r1 CACHE=cache25 ./run_eval_model.sh
# ★推論は greedy なので分割しても結果は変わらない。結合後に**必ず 4,008 件を assert**する
#   （[[tar-over-ssh-truncates-silently]] と同じ発想: 件数を見るまで信じない）。
set -uo pipefail
cd "$(dirname "$0")"
# ★dl2 には /data4 が無い（跨マシンは /mnt/data/data4）。REPO / PYBIN で上書きできるようにする。
R="${REPO:-/data4/src/shunsuke/MICCAI2026/Orena}"
export PYTHONPATH="$R/reference/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="${PYBIN:-$R/.venv-dl1/bin/python}"
NAME="${NAME:?}"; ADAPTER="${ADAPTER:?}"; VARIANT="${VARIANT:?}"
CACHE="${CACHE:-cache25}"; GPUS="${GPUS:-0 1 3}"; SIZE="${SIZE:-768}"
step(){ echo "=== $(date '+%F %T') $* ==="; }

n=0; for g in $GPUS; do n=$((n+1)); done
step "$NAME variant=$VARIANT adapter=$ADAPTER を $n 分割（GPU: $GPUS）"
i=0
for g in $GPUS; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g \
  setsid nohup $PY screen_infer.py --adapter "$ADAPTER" --variant "$VARIANT" \
    --cache-root "$PWD/$CACHE" --out-tag "$NAME" --shard "$i/$n" --size "$SIZE" \
    > "infer_${NAME}_s$i.log" 2>&1 < /dev/null &
  echo "  shard $i/$n → GPU$g pid=$!"
  i=$((i+1))
done
wait
step "全シャード終了 → 結合"

NAME="$NAME" NSHARD="$n" $PY - <<'PY' || { echo "★結合に失敗"; exit 1; }
import json, os
from pathlib import Path
tag = os.environ["NAME"]; n = int(os.environ["NSHARD"])
rows, metas = [], []
for i in range(n):
    d = Path("results")/f"{tag}.shard{i}of{n}"
    rows += json.loads((d/"responses.json").read_text())
    metas.append(json.loads((d/"meta.json").read_text()))
uids = {r["uid"] for r in rows}
print(f"結合 {len(rows)} 問 / ユニーク {len(uids)}")
assert len(rows) == 4008 and len(uids) == 4008, "★件数が 4,008 でない"
out = Path("results")/tag; out.mkdir(parents=True, exist_ok=True)
(out/"responses.json").write_text(json.dumps(rows, indent=1))
m = dict(metas[0]); m["n"] = len(rows); m["shards"] = n
m["n_overlay"] = sum(1 for r in rows if r.get("overlay"))
m["latency_mean"] = sum(r["latency"] for r in rows)/len(rows)
m["latency_max"] = max(r["latency"] for r in rows)
(out/"meta.json").write_text(json.dumps(m, indent=1))
print("重畳あり", m["n_overlay"], "/ latency mean", round(m["latency_mean"],2),
      "/ max", round(m["latency_max"],2))
PY

step "JUDGE $NAME"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(echo $GPUS | cut -d' ' -f1) \
  $PY "$R/workspace/expE01_segproc_baseline/eval_seg.py" "results/$NAME" --track FRAME
step "JUDGE rc=$?  $NAME DONE"
