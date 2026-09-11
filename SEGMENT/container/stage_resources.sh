#!/usr/bin/env bash
# v019: adapter 3 本を resources/ に置き、md5 で照合する。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
stage(){ local name=$1 src=$2 want=$3
  [ -d "$src" ] || { echo "ERROR: $name が無い: $src" >&2; exit 1; }
  rm -rf "$SCRIPT_DIR/resources/$name"; cp -r "$src" "$SCRIPT_DIR/resources/$name"; chmod -R u+rw "$SCRIPT_DIR/resources/$name"
  got=$(md5sum "$SCRIPT_DIR/resources/$name/adapter_model.safetensors" | cut -d" " -f1)
  if [ "$want" = "TBD" ]; then echo "$name md5=$got（★未固定: 学習完了後にここへ書く）"; else
    [ "$got" = "$want" ] || { echo "ERROR: $name md5 不一致 got=$got want=$want" >&2; exit 1; }; echo "$name md5 OK: $got"; fi
}
mkdir -p "$SCRIPT_DIR/resources"
stage adapter_n04  "$ROOT/workspace/expN04_seg_framerouter/results/expN04_seg_framerouter/fold0/adapter" 71f7d92a6df032607f3d4a01a1e6ebdd
stage adapter_r00  "$ROOT/workspace/expR00_seg_s3/results/expR00_seg_s3_r16_alldata_5090/fold0/adapter" c88fde1d59a3716fd93271d2e6723005
stage adapter_r00c "$ROOT/workspace/expR00_seg_s3/results/expR00C_seg_focusft_alldata/fold0/adapter" 09a229b19b760ae3319ba7948f050a1a
du -sh "$SCRIPT_DIR/resources"/*
