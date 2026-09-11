"""部分実行の応答を対照の応答で埋めて 2000 問に戻す.

フレーム選択は**問ごとに独立**で、生成は greedy（`do_sample=False`）なので、
介入しなかった問の応答は対照と厳密に同一になる。よって「介入した問だけ推論して
残りを対照から流用する」のは近似ではなく**厳密**。2000問 2.6h → 690問 0.9h に縮む。
"""
import json, shutil, sys
from pathlib import Path

part, ctrl, out = (Path(x) for x in sys.argv[1:4])
pr = json.loads((part / "responses.json").read_text())
cr = json.loads((ctrl / "responses.json").read_text())
key = "qID" if isinstance(pr, list) and pr and "qID" in pr[0] else None
assert key, f"responses.json の形式が想定外: {type(pr)}"
have = {str(r[key]) for r in pr}
merged = pr + [r for r in cr if str(r[key]) not in have]
out.mkdir(parents=True, exist_ok=True)
for f in ("meta.json",):
    if (part / f).exists():
        shutil.copy(part / f, out / f)
(out / "responses.json").write_text(json.dumps(merged))
print(f"spliced: 実行 {len(pr)} + 流用 {len(merged)-len(pr)} = {len(merged)} -> {out}")
