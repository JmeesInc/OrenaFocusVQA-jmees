#!/usr/bin/env bash
# r4（クラス色 + 個体番号 + confidence）で FRAME 重畳キャッシュを作り直す。
# ★conf は A と同じ 0.25。train(DUAL arm) と val の両方。dl1 4GPU シャード。
set -uo pipefail
cd "$(dirname "$0")"
R=/data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$R/reference/src" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=$R/.venv-dl1/bin/python
step(){ echo "=== $(date '+%F %T') $* ==="; }

for part in val train; do
  step "$part を r4 でレンダ（4シャード）"
  ARM=""; [ "$part" = train ] && ARM="--arm DUAL"
  for i in 0 1 2 3; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$i \
    setsid nohup $PY render_cache.py --part "$part" $ARM --variants r4 \
      --conf 0.25 --conf-lo 0.25 --cache-root "$PWD/cache_r4" --shard "$i/4" \
      > "render_r4_${part}_s$i.log" 2>&1 < /dev/null &
  done
  wait
  step "$part 完了"
done
step "合計 $(find cache_r4/r4 -name '*.jpg' | wc -l) 枚 / index $(ls cache_r4/r4/index*.json | wc -l) 本"
