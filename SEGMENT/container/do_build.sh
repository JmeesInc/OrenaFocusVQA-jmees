#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_TAG="orena-focus-segment-conf2"
BASE_IMAGE="orena-focus-frame-n16000"
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "ERROR: base image '$BASE_IMAGE' が無い。先に submit/v003_qlora_n16000/do_build.sh を実行してください。" >&2
  exit 1
fi
if [ ! -d "$SCRIPT_DIR/resources/adapter_n04" ]; then
  echo "ERROR: resources/adapter が無い。stage_resources.sh を先に実行してください。" >&2
  exit 1
fi
docker build "$SCRIPT_DIR" --platform=linux/amd64 \
  --build-arg "BUILD_ID=${BUILD_ID:-r1}" --tag "$DOCKER_TAG" "$@"
echo "built: $DOCKER_TAG"
