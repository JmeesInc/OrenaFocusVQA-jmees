#!/usr/bin/env bash
# 提出コンテナを **CV と同じ母集団・同じ採点器**で検証する。
#
# ★狙い: コンテナが CV 相当の SCORE を出せるかを確認する。
#   推論経路が違う（クリップからデコード vs 元動画から時刻指定抽出）ので、
#   同じ問・同じ採点で比べないと差の原因が切り分けられない。
#
# ★eval_seg.py は**移植しない**。`responses.json` を読むだけなので、
#   コンテナの `answer.json` を同じ形へ変換して食わせる（score_as_cv.py）。
#
# 使い方: validate_vs_cv.sh <バッチ入力dir> <出力タグ> [GPU]
set -uo pipefail
IN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"   # ★docker -v は絶対パスが要る
TAG=$2; GPU=${3:-2}
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ★`run_validate.sh` は一時コピーを走らせるので `$D` はリポジトリ内ではない。
#   ランチャーが渡す FOCUS_REPO_DIR を優先する（無ければ従来どおり相対で辿る）。
R="${FOCUS_REPO_DIR:-$(cd "$D/../.." && pwd)}"
OUT=/tmp/${TAG}_out
rm -rf "$OUT"; mkdir -p "$OUT"; chmod -R o+rwX "$OUT"

# ★VRAM が空くまで待つ。共有 GPU なので、前回の judge やコンテナが残っていると
#   モデルロードで OOM し、**全問が空回答**になる（2026-09-01 に実害）。
#   OOM は例外で落ちるだけなので、答えが空でも「動いたように見える」のが厄介。
NEED_MB=${NEED_MB:-18000}
for _ in $(seq 1 180); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
  [ "$FREE" -ge "$NEED_MB" ] && break
  echo "GPU$GPU の空き ${FREE}MB < ${NEED_MB}MB。待機..."; sleep 20
done
echo "GPU$GPU 空き ${FREE}MB で開始"

echo "=== (1) コンテナ実行（本番相当: --cpus 4 / --network none）==="
docker run --rm --gpus "device=$GPU" --network none --cpus 4 \
  -v "$IN":/input:ro -v "$OUT":/output \
  orena-focus-procedure-index > "$OUT/container.log" 2>&1
# ★コンテナのログは**全文残す**（tail -5 だと例外の先頭が消えて原因が追えない）
tail -5 "$OUT/container.log"
grep -nE "Traceback|Error:" "$OUT/container.log" | head -5

[ -f "$OUT/answer.json" ] || { echo "answer.json が無い"; exit 1; }

echo "=== (2) answer.json -> responses.json ==="
"$R/.venv/bin/python" "$D/score_as_cv.py" "$OUT/answer.json" "$OUT/scored" || exit 1

echo "=== (3) CV と同じ採点器で SCORE ==="
# time の後処理はコンテナ内で適用済みなので off（on だと二重適用）
PYTHONPATH="$R/reference/src" "$R/.venv/bin/python" \
  "$R/workspace/expE01_segproc_baseline/eval_seg.py" "$OUT/scored" \
  --track PROCEDURE --time-postproc off 2>&1 | grep -E "pooled|^\| (agg|comp|even|obje|temp)"

echo "=== (4) 同じ問での CV の値 ==="
"$R/.venv/bin/python" - "$OUT/answer.json" <<'PY'
import json,sys,pandas as pd
from pathlib import Path
R=Path(__file__).resolve() if False else Path("/mnt/data/data4/src/shunsuke/MICCAI2026/Orena")
if not R.exists(): R=Path("/data4/src/shunsuke/MICCAI2026/Orena")
ans=json.load(open(sys.argv[1])); qs={int(a["qID"]) for a in ans}
cv=pd.read_csv(R/"workspace/expE01_segproc_baseline/results/eval_expI06_c9w_win8/results_merged.csv")
s=cv[cv.qID.isin(qs)]
print(f"  対象 {len(s)}/{len(qs)} 問  CV の正答率 {s.correct.astype(int).mean():.4f}")
print("  ※ SCORE はバケット非加重平均。少数問では正答率で比べる方が安定する")
PY
