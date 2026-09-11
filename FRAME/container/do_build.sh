#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_TAG="orena-focus-frame-router"
BASE_IMAGE="orena-focus-frame-n16000"
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
  echo "ERROR: base image '$BASE_IMAGE' が無い。先に submit/v003_qlora_n16000/do_build.sh を実行。" >&2; exit 1; }
for m in m1_m03b_overlay m2_m03b_768 m3_v06_r32 b1_q02b_r1 b2_q00b_r4 b3_q03b_r1sa detector detector_clip; do
  [ -d "$SCRIPT_DIR/resources/$m" ] || { echo "ERROR: resources/$m が無い。stage_resources.sh を先に。" >&2; exit 1; }
done
docker build "$SCRIPT_DIR" --platform=linux/amd64 \
  --build-arg "BUILD_ID=${BUILD_ID:-r1}" --tag "$DOCKER_TAG" "$@"
echo "built: $DOCKER_TAG"
