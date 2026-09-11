#!/usr/bin/env bash
# 提出物 (.tar.gz) を作る。アップロードは手動。
#
# ★grand-challenge は **イメージ digest でアカウント横断の重複判定**をする。
#   一度上げた digest は（**Failed になった提出のものでも**）二度と登録できないので、
#   上げ直すときは `BUILD_ID=v008r4 bash do_save.sh` のように **BUILD_ID を上げる**。
#   ファイル名にも BUILD_ID を入れて、どれを上げるべきか取り違えないようにする。
#   （2026-08-27: 旧版は BUILD_ID をファイル名に入れず検算も無かったため v009 に合わせた）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_TAG="orena-focus-frame-router"
BUILD_ID="${BUILD_ID:-r1}"
export BUILD_ID   # do_build.sh へ確実に伝える

echo "BUILD_ID=${BUILD_ID}"
"$SCRIPT_DIR/do_build.sh"

# ★ビルド後に**イメージの中身で**検算する。BUILD_ID を渡し忘れると
#   「テストしたのは r3 なのに保存したのは r2」という取り違えが起きる（無言で digest が戻る）。
built_id="$(docker run --rm --entrypoint cat "$DOCKER_TAG" /opt/app/resources/.build_id)"
if [ "$built_id" != "$BUILD_ID" ]; then
  echo "ERROR: image の .build_id が '${built_id}' で BUILD_ID='${BUILD_ID}' と一致しない" >&2
  exit 1
fi

OUT="${OUT_DIR:-$SCRIPT_DIR}/${DOCKER_TAG}-${BUILD_ID}.tar.gz"
if command -v pigz >/dev/null 2>&1; then
  docker save "$DOCKER_TAG" | pigz -p "${PIGZ_THREADS:-8}" -c > "$OUT"
else
  docker save "$DOCKER_TAG" | gzip -c > "$OUT"
fi
echo "saved: $OUT ($(du -h "$OUT" | cut -f1))"
echo "image digest: $(docker image inspect "$DOCKER_TAG" --format '{{.Id}}')"
