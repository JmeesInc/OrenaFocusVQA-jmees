#!/usr/bin/env bash
# ローカル回帰テスト。★打ち切り経路を実際に踏むため、予算を絞った run も回す。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCKER_TAG="orena-focus-frame-router"
"$SCRIPT_DIR/do_build.sh"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3

run_case () {   # $1=ラベル  $2=BUDGET_SCALE
  local tag="$1" scale="$2" out="$SCRIPT_DIR/test/output_$1"
  rm -rf "$out"; mkdir -p "$out"; chmod -R o+rwX "$out"
  echo "##### case=$tag BUDGET_SCALE=$scale #####"
  docker run --rm --gpus "device=${TEST_GPU:-0}" --network none \
    -e "FOCUS_BUDGET_SCALE=$scale" \
    -v "$SCRIPT_DIR/test/in/interface_1":/input:ro -v "$out":/output "$DOCKER_TAG" \
    2>&1 | tee "$out/container.log"
  # ★「✓」だけ見て通してはいけない。**機能が黙って無効化されていないか**を数字で確認する。
  #   2026-08-22: scipy 欠落で detector が fail-soft し、重畳なしで 3ケースとも ✓ が出た。
  # ★期待する views は **inference.py の MEMBERS から導出**する（固定文字列で書くと
  #   メンバー構成を変えるたびに嘘のアサートになる。2026-09-03 に実際に踏んだ）。
  local want; want=$("$PY" "$SCRIPT_DIR/expected_views.py")
  echo "  期待 views=$want"
  grep -q "overlay=True procedure=True hernia_mesh=.* views=$want" "$out/container.log" || {
    echo "✗ FAIL[$tag]: 重畳/procedure/視点構成が想定と違う" >&2
    grep -E "detector のロードに失敗|overlay=" "$out/container.log" >&2; exit 1; }
  local nfail
  nfail=$(grep -c "重畳に失敗" "$out/container.log" || true)
  [ "$nfail" -eq 0 ] || { echo "✗ FAIL[$tag]: 重畳が $nfail 問で失敗" >&2; exit 1; }
  # ★★「重畳が有効」だけでなく **実際に何問へ描かれたか** を数える。
  #   検出0件ゲートがあるので 0 問にはならないはず（val 実測で 78.6% に重畳が付く）。
  #   ここが 0 なら検出器が黙って何も出していない＝ v008 の scipy 事故と同じ型。
  local ndet
  ndet=$(grep -c "detector: all=" "$out/container.log" || true)
  [ "$ndet" -ge 1 ] || { echo "✗ FAIL[$tag]: detector のロードログが無い" >&2; exit 1; }
  echo "✓ overlay/procedure 有効 (重畳失敗 0 問 / detector ロード $ndet 回)"
  PYTHONPATH="$ROOT/reference/src" "$PY" "$SCRIPT_DIR/validate_output.py" \
    "$SCRIPT_DIR/test/in/interface_1/request.json" "$out/answer.json"
}

run_case full 1.0      # 3パス入るはず

# ★hernia sponge→mesh の経路を実際に踏む。procedure_type を書き換えたテスト入力を作り、
#   FOCUS_HERNIA_MESH=1 で置換が起きること、既定(0)では起きないことを両方確認する。
"$PY" - "$SCRIPT_DIR" <<'PYEOF'
import json, shutil, sys
from pathlib import Path
src = Path(sys.argv[1]) / "test/in/interface_1"
dst = src.parent / "interface_hernia"
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst)
f = dst / "request.json"
rs = json.loads(f.read_text())
for r in rs:
    r["procedure_type"] = "Laparoscopic Inguinal Hernia Repair (TAPP)"
f.write_text(json.dumps(rs, indent=1))
print(f"hernia テスト入力を作成: {dst} ({len(rs)}問)")
PYEOF

run_hernia () {   # $1=ラベル $2=FOCUS_HERNIA_MESH
  local tag="$1" flag="$2" out="$SCRIPT_DIR/test/output_$1"
  rm -rf "$out"; mkdir -p "$out"; chmod -R o+rwX "$out"
  echo "##### case=$tag FOCUS_HERNIA_MESH=$flag #####"
  docker run --rm --gpus "device=${TEST_GPU:-0}" --network none \
    -e "FOCUS_HERNIA_MESH=$flag" \
    -v "$SCRIPT_DIR/test/in/interface_hernia":/input:ro -v "$out":/output "$DOCKER_TAG" \
    2>&1 | tee "$out/container.log"
  grep -q "hernia_mesh=$([ "$flag" = 1 ] && echo True || echo False)" "$out/container.log" \
    || { echo "✗ FAIL[$tag]: hernia_mesh フラグが反映されていない" >&2; exit 1; }
  PYTHONPATH="$ROOT/reference/src" "$PY" "$SCRIPT_DIR/validate_output.py" \
    "$SCRIPT_DIR/test/in/interface_hernia/request.json" "$out/answer.json"
}
run_hernia hernia_off 0
run_hernia hernia_on 1
"$PY" - "$SCRIPT_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]) / "test"
def load(p):
    x = json.loads((d/p/"answer.json").read_text())
    x = x if isinstance(x, list) else x["answers"]
    return {a["qID"]: a["content"] for a in x}
off, on = load("output_hernia_off"), load("output_hernia_on")
ch = [(q, off[q], on[q]) for q in off if off[q] != on[q]]
sp = sum(1 for v in off.values() if "sponge" in v.lower())
print(f"OFF で sponge を含む回答 {sp}問 / ON で変化した回答 {len(ch)}問")
for q, a, b in ch[:5]:
    print(f"   {a!r} → {b!r}")
assert len(ch) == sp, f"★置換件数が sponge 件数と一致しない ({len(ch)} != {sp})"
print("✓ hernia sponge→mesh: OFF/ON の差分が sponge 件数と一致")
PYEOF
run_case tight 0.25    # ★pass 2/3 の途中で打ち切られるはず（打ち切り経路の検証）
run_case p1only 0.16   # ★pass 1 のみ。全問に回答が残ることの検証
