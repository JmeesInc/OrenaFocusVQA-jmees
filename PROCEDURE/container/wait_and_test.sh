#!/usr/bin/env bash
# GPU が空き次第（free >= 25GB）コンテナ回帰テストを打つ。
# ★別レーンの評価が 4 GPU を占有しているので、奪わずに空きを待つ。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEED_MB=${NEED_MB:-25000}
while true; do
  G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F', ' -v n="$NEED_MB" '$2>n {print $1; exit}')
  if [ -n "$G" ]; then
    echo "$(date '+%F %T') GPU$G が空いた（free $(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader)）"
    TEST_GPU=$G bash "$SCRIPT_DIR/do_test_run_nocdi.sh"
    exit $?
  fi
  sleep 60
done
