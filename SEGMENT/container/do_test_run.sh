#!/usr/bin/env bash
# ローカル回帰テスト: 本番仕様のクリップでコンテナを回し answer.json を検証する。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_TAG="orena-focus-segment-conf2"
"$SCRIPT_DIR/do_build.sh"
rm -rf "$SCRIPT_DIR/test/output"; mkdir -p "$SCRIPT_DIR/test/output"; chmod -R o+rwX "$SCRIPT_DIR/test/output"
# ★本番は `ml.g6e.xlarge`（L40S / **4 vCPU**）を選ぶ。デコードが CPU 律速なので
#   ローカルでも --cpus で本番相当に絞らないと latency の見積もりが甘くなる。
#   （2026-08-14 まで 8 vCPU 前提だったのは `ml.g7e.2xlarge` を選ぶつもりだった名残。
#     そのインスタンスは Blackwell/sm_120 でこのイメージが動かず、提出が Failed になった）
docker run --rm --gpus "device=${TEST_GPU:-0}" --network none \
  --cpus "${TEST_CPUS:-4}" \
  -v "$SCRIPT_DIR/test/input/interface_1":/input:ro \
  -v "$SCRIPT_DIR/test/output":/output \
  "$DOCKER_TAG"
# ★system python3 だと `focus` が無く形式 verify チェックが黙ってスキップされる
#   （"(format check skipped: No module named 'focus')" が出ていたら venv を使えていない）。
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
PYTHONPATH="$ROOT/reference/src" "$PY" "$SCRIPT_DIR/validate_output.py" \
  "$SCRIPT_DIR/test/input/interface_1/request.json" "$SCRIPT_DIR/test/output/answer.json"
