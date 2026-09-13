#!/usr/bin/env bash
# resources/ を配置する（git 管理外・サイズが大きいため）。
# ★base_model は既存イメージのレイヤーを再利用するのでコピー不要（Dockerfile 参照）。
#
# v010 で増えたもの:
#   m2f/            Mask2Former(expF40) の HF 重み  … TRT が使えない環境のフォールバック
#   m2f_sm89.trt    L40S(Ada, sm_89) 用の TensorRT エンジン … 本番はこれが使われる
#   m2f_classes.json 索引のクラス列（expF40 は Gallstone を落として 7 クラス）
#   phase_cholec/   胆摘用の工程モデル（ConvNeXtV2 + MS-TCN stride5）
#   phase_heico/    大腸用の工程モデル
#   phase_lib/      PhaseNet / load_tcn の定義（重みを読むのに要る）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
R="$SCRIPT_DIR/resources"
mkdir -p "$R"

# --- VLM アダプタ（★expP05 = expP02 の all-data 版。fold 版 CV 0.5934 / 対照 0.5386）---
#   ★CV は測れない（val が学習データに入っている）。判定は LB のみ。
#   箱（8×5090）から回収した adapter をこのパスに置いてから実行すること。
ADAPTER_SRC="${ADAPTER_SRC:-$ROOT/workspace/expM03_round1_external/results/expP05_warm_proc64f_alldata/fold0/adapter}"
[ -f "$ADAPTER_SRC/adapter_config.json" ] || {
  echo "★adapter が無い: $ADAPTER_SRC"; echo "  箱から回収してから実行する"; exit 1; }
rm -rf "$R/adapter"; cp -r "$ADAPTER_SRC" "$R/adapter"; chmod -R u+rw "$R/adapter"
python3 - "$R/adapter" <<'PY'
import json, sys
from pathlib import Path
c = json.load(open(Path(sys.argv[1]) / "adapter_config.json"))
print(f"  adapter: r={c.get('r')} alpha={c.get('lora_alpha')} "
      f"targets={len(c.get('target_modules') or [])}")
assert int(c.get("r", 0)) == 16, f"★r が 16 でない: {c.get('r')}"
PY

# --- 採点直前処理のテーブル（CV と揃えないと LB がズレる）---
cp "$SCRIPT_DIR/../v005_segment_hybrid/resources/group_templates.json" "$R/"
cp "$SCRIPT_DIR/../v005_segment_hybrid/time_count_prior.json" "$R/"

# --- FO 索引の検出器（expF40: segm_AP 0.4043, 7 クラス）---
M2F_SRC="$ROOT/workspace/expF00_fo_instseg/results/expF40_m2f_ps1sN_noGallstone_0828/fold0/best_model"
rm -rf "$R/m2f"; cp -r "$M2F_SRC" "$R/m2f"; chmod -R u+rw "$R/m2f"
python3 - "$M2F_SRC" "$R/m2f_classes.json" <<'PY'
import json, sys
from pathlib import Path
d = json.load(open(Path(sys.argv[1]) / "config.json"))["id2label"]
json.dump([d[str(i)] for i in range(len(d))], open(sys.argv[2], "w"))
print("classes ->", sys.argv[2])
PY
# TRT エンジン（Ada 用。無ければ PyTorch フォールバックで動く）
ENG="$ROOT/workspace/expI03_fo_index/m2f_expF40_ada.trt"
[ -f "$ENG" ] && cp "$ENG" "$R/m2f_sm89.trt" || echo "⚠️ Ada 用 TRT エンジンが無い（PyTorch で動作）"

# --- 工程モデル（アンカー時刻と同じ工程の区間へ絞るのに使う。外すと −0.0081）---
P="$ROOT/workspace/expI00_phase_clf"
for pair in "cholec:expI00b_cholec_convnextv2t_sn" "heico:expI00a_heico_convnextv2t_sn"; do
  name="${pair%%:*}"; dir="${pair##*:}"
  mkdir -p "$R/phase_$name/tcn"
  cp "$P/results/$dir/fold0/best_model.pt" "$R/phase_$name/"
  # ★stride 5/10/20 を全部入れる。フレームの刻みを変えたとき**最も近い重み**を選ぶため
  #   （1 つしか入れないと選択のしようがなく、刻みを変えても TCN が合わないまま動く）。
  cp "$P/results/$dir/fold0/tcn/"tcn_stride*.pt "$R/phase_$name/tcn/"
done
mkdir -p "$R/phase_lib"
# ★train_tcn.py も要る（evaluate.load_tcn が MSTCN をそこから import する）。
#   入れ忘れると「工程モデルを読めない」で**黙って工程なしに落ちる**（2026-09-01 に実際に踏んだ）。
cp "$P/model.py" "$P/evaluate.py" "$P/dataset.py" "$P/train_tcn.py" "$R/phase_lib/"
for f in model.py evaluate.py train_tcn.py; do
  [ -f "$R/phase_lib/$f" ] || { echo "★必須ファイルが無い: $f"; exit 1; }
done

du -sh "$R"/* | sort -h
