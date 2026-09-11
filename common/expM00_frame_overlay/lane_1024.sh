#!/usr/bin/env bash
# A@1024-dual を測る: val の重畳を **1024px で再レンダ** → 1024 推論 → judge
# ★重畳は「モデルが見る幅」で描く規約なので、1024 推論なら 1024 で描いたキャッシュが要る。
set -uo pipefail
cd "$(dirname "$0")"
R=/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$R/reference/src" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=$R/.venv-dl1/bin/python
AD=$R/workspace/expE01_segproc_baseline/results/expM00A_frame_overlay/fold0/adapter
step(){ echo "=== $(date '+%F %T') $* ==="; }

step "val 重畳を 1024px でレンダ（4シャード）"
for i in 0 1 2 3; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$i \
  setsid nohup $PY render_cache.py --part val --variants r1 --conf 0.25 --conf-lo 0.25 \
    --size 1024 --out-width 1024 --cache-root "$PWD/cache25_1024" --shard "$i/4" \
    > "render1024_s$i.log" 2>&1 < /dev/null &
done
wait
step "レンダ完了: $(find cache25_1024/r1 -name '*.jpg' | wc -l) 枚 / index $(ls cache25_1024/r1/index*.json | wc -l) 本"

step "A@1024-dual を評価"
NAME=eval_M00A_1024 ADAPTER="$AD" VARIANT=r1 CACHE=cache25_1024 SIZE=1024 bash run_eval_model.sh
step "rc=$?  LANE DONE"
