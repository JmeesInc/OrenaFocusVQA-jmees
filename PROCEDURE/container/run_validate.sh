#!/usr/bin/env bash
# validate_vs_cv.sh を**一時コピーから**実行する launcher。
#
# ★bash はスクリプトを**バイトオフセットで読み進める**ので、実行中に本体を編集すると
#   途中から別の行を実行して意味不明なエラーになる（2026-09-01: `line 36: ut: command not found`）。
#   検証は数十分かかり、その間に編集したくなるので、必ずコピーを走らせる。
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp /tmp/validate_vs_cv.XXXXXX.sh)"
cp "$D/validate_vs_cv.sh" "$TMP"
trap 'rm -f "$TMP"' EXIT
# 一時コピーからだとリポジトリルートを相対で辿れないので、実体の場所を渡す
export FOCUS_REPO_DIR="$(cd "$D/../.." && pwd)"
bash "$TMP" "$@"
