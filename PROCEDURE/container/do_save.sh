#!/usr/bin/env bash
# 提出物 (.tar.gz) を作る。アップロードは手動。
#
# ★grand-challenge は **イメージ digest でアカウント横断の重複判定**をする。
#   一度上げた digest は（Failed になった提出のものでも）二度と登録できないので、
#   上げ直すときは `BUILD_ID=r2 bash do_save.sh` のように **BUILD_ID を上げる**。
#   ファイル名にも BUILD_ID を入れて、どれを上げるべきか取り違えないようにする。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_TAG="orena-focus-procedure-p00c"   # v016: all-data r16 + 匿名フレーム置換
BUILD_ID="${BUILD_ID:-r1}"
export BUILD_ID   # do_build.sh へ確実に伝える

echo "BUILD_ID=${BUILD_ID}"
"$SCRIPT_DIR/do_build.sh"

# ★ビルド後に**イメージの中身で**検算する。BUILD_ID を渡し忘れると
#   「テストしたのは r2 なのに保存したのは r1」という取り違えが起きる（無言で digest が戻る）。
# ★dl1 は nvidia-container-toolkit が無く既定の nvidia runtime が使えない（2026-09-04）。
#   この検算は GPU 不要なので runc で回す。
built_id="$(docker run --rm --runtime=runc --entrypoint cat "$DOCKER_TAG" /opt/app/resources/.build_id)"
if [ "$built_id" != "$BUILD_ID" ]; then
  echo "ERROR: image の .build_id が '${built_id}' で BUILD_ID='${BUILD_ID}' と一致しない" >&2
  exit 1
fi

OUT="${OUT_DIR:-$SCRIPT_DIR}/${DOCKER_TAG}-${BUILD_ID}.tar.gz"
# pigz があれば使う（27GB の圧縮が単スレッド gzip だと 20分以上かかる）。出力は同じ gzip 形式。
if command -v pigz >/dev/null 2>&1; then
  docker save "$DOCKER_TAG" | pigz -p "${PIGZ_THREADS:-8}" -c > "$OUT"
else
  docker save "$DOCKER_TAG" | gzip -c > "$OUT"
fi
echo "saved: $OUT ($(du -h "$OUT" | cut -f1))"
echo "image digest: $(docker image inspect "$DOCKER_TAG" --format '{{.Id}}')"
