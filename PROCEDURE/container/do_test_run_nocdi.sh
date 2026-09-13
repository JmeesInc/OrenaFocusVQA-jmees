#!/usr/bin/env bash
# ローカル回帰テスト（★nvidia-container-toolkit が無い dl1 用）。
#
# ★2026-09-04: dl1 から nvidia-container-toolkit が消えており `--gpus` が
#   `failed to discover GPU vendor from CDI` で落ちる（daemon.json は nvidia runtime を
#   参照しているのに実体が無い）。root が要る修復は避け、**デバイスノードとドライバ .so を
#   手で流し込む**ことで同じことをする。CUDA が見えることは実測で確認済み。
#   本番(grand-challenge)は正規の GPU 連携なので、この差は**ローカル検証だけの都合**。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_TAG="orena-focus-procedure-p00c"
GPU="${TEST_GPU:-3}"
V="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"

rm -rf "$SCRIPT_DIR/test/output"; mkdir -p "$SCRIPT_DIR/test/output"
chmod -R o+rwX "$SCRIPT_DIR/test/output"

# ★本番は ml.g6e.xlarge（L40S / 4 vCPU）。デコードが CPU 律速なので --cpus で本番相当に絞る
docker run --rm --runtime=runc --network none --cpus "${TEST_CPUS:-4}" \
  ${FOCUS_TWO_PATH:+-e FOCUS_TWO_PATH="$FOCUS_TWO_PATH"} \
  ${FOCUS_VIDEO_FRAMES:+-e FOCUS_VIDEO_FRAMES="$FOCUS_VIDEO_FRAMES"} \
  ${FOCUS_PER_Q_BUDGET_S:+-e FOCUS_PER_Q_BUDGET_S="$FOCUS_PER_Q_BUDGET_S"} \
  ${FOCUS_N_QUESTIONS:+-e FOCUS_N_QUESTIONS="$FOCUS_N_QUESTIONS"} \
  --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  --device "/dev/nvidia${GPU}" \
  -v "/usr/lib/x86_64-linux-gnu/libcuda.so.$V:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro" \
  -v "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.$V:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro" \
  -v "/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.$V:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro" \
  -v "/usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.$V:/usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.4:ro" \
  -v "$SCRIPT_DIR/test/input/${TEST_IFACE:-interface_1}":/input:ro \
  -v "$SCRIPT_DIR/test/output":/output \
  "$DOCKER_TAG"

ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
PYTHONPATH="$ROOT/reference/src" "$PY" "$SCRIPT_DIR/validate_output.py" \
  "$SCRIPT_DIR/test/input/${TEST_IFACE:-interface_1}/request.json" "$SCRIPT_DIR/test/output/answer.json"
