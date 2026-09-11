#!/usr/bin/env bash
# val レンダ(PID待ち) → r1@conf0.25 の推論を **GPU0/1/3 の3分割** → 結合 → judge
#   bash lane_c25_par.sh <render_pid>
# ★PID 直指定で待つ（pgrep -f は自分自身にマッチする）。
set -uo pipefail
cd "$(dirname "$0")"
R=/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$R/reference/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=$R/.venv-dl1/bin/python
AD=$R/workspace/expE01_segproc_baseline/results/expK01F_frame_pseudo/fold0/adapter
TAG=screen_r1c25
step(){ echo "=== $(date '+%F %T') $* ==="; }

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  step "val レンダ (pid $WAIT_PID) の完了を待つ"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
fi

step "★index の突合（val 3,087 + train 7,272 = 10,359）"
$PY - <<'PY' || { echo "★index が不整合。推論しない"; exit 1; }
import json, glob, sys
n = {p: len(json.load(open(p))) for p in sorted(glob.glob('cache25/r1/index*.json'))}
print(n, "合計", sum(n.values()))
sys.exit(0 if sum(n.values()) == 10359 else 1)
PY

step "推論を 3 分割で起動（GPU0 / GPU1 / GPU3）"
i=0
for g in 0 1 3; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g \
  setsid nohup $PY screen_infer.py --adapter "$AD" --variant r1 \
    --cache-root "$PWD/cache25" --out-tag "$TAG" --shard "$i/3" \
    > "infer_c25_s$i.log" 2>&1 < /dev/null &
  echo "  shard $i/3 → GPU$g pid=$!"
  i=$((i+1))
done
wait
step "3 シャード完了 → 結合"

$PY - <<'PY' || { echo "★結合に失敗"; exit 1; }
import json, sys
from pathlib import Path
tag = "screen_r1c25"
out = Path("results")/tag
rows, metas = [], []
for i in range(3):
    d = Path("results")/f"{tag}.shard{i}of3"
    rows += json.loads((d/"responses.json").read_text())
    metas.append(json.loads((d/"meta.json").read_text()))
uids = {r["uid"] for r in rows}
print(f"結合 {len(rows)} 問 / ユニーク {len(uids)}")
assert len(rows) == 4008 and len(uids) == 4008, "★件数が 4,008 でない"
out.mkdir(parents=True, exist_ok=True)
(out/"responses.json").write_text(json.dumps(rows, indent=1))
m = dict(metas[0]); m["n"] = len(rows); m["shards"] = 3
m["n_overlay"] = sum(1 for r in rows if r.get("overlay"))
m["latency_mean"] = sum(r["latency"] for r in rows)/len(rows)
m["latency_max"] = max(r["latency"] for r in rows)
(out/"meta.json").write_text(json.dumps(m, indent=1))
print("重畳あり", m["n_overlay"], "/ latency mean", round(m["latency_mean"],2))
PY

step "JUDGE"
$PY "$R/workspace/expE01_segproc_baseline/eval_seg.py" "results/$TAG" --track FRAME
step "JUDGE rc=$?  LANE DONE"
