"""`inference.py` の ROUTES から、ログに出るはずの views 文字列を導出する。

★テスト側に固定文字列で書くと、メンバー構成を変えたときに**嘘のアサート**になる
  （2026-09-03: `views=D/C/D` を期待したまま構成を変えて FAIL した）。
★v017 で MEMBERS が ROUTES からの内包表記になったので、**ROUTES を読む**形に変えた
  （旧実装は `MEMBERS = [` のリテラルを正規表現で拾っており、AttributeError で落ちていた）。
"""
import re
from pathlib import Path

src = (Path(__file__).parent / "inference.py").read_text()
blk = re.search(r"ROUTES: dict\[str, list\[tuple\[str, int, str\]\]\] = \{(.*?)\n\}", src, re.S)
if blk is None:
    raise SystemExit("★inference.py から ROUTES を読めない。expected_views.py を直すこと")
out = []
for rk, body in re.findall(r'"(\w+)":\s*\[(.*?)\n    \]', blk.group(1), re.S):
    variants = re.findall(r'\(\s*"[^"]+"\s*,\s*\d+\s*,\s*"([^"]*)"\s*\)', body)
    out.append(f"{rk}:" + "/".join(v or "C" for v in variants))
print(" ".join(out))
