r"""回答を各形式の `verify()` が通る形へ正規化する（落とすと自動的に0点になる分を拾う）.

## なぜ必要か
公式の `AnswerFormat.verify()` は**厳格**で、少しでも余計な文字があると例外＝**自動的に不正解**になる:

- `Number.verify`: `text.strip().isdigit()` → **`'1.'` は落ちる**
- `Binary.verify` : `strip().lower() in ('yes','no')` → **`'Yes.'` は落ちる**
- `Percentage.verify`: `^\d+(\.\d+)?\s*%?$` → `'about 12%'` は落ちる
- `FOClass.verify`: カンマ分割した各要素がクラス名 or `none` → 末尾の `.` が付くと落ちる

FRAME LoRA(expD11) を SEGMENT に当てた実測（fold0 val 2000問）:

| 形式 | n | 正規化前 | 正規化後 | 差 |
|---|---|---|---|---|
| number | 148 | 0.2973 | **0.3581** | **+9問**（`'1.'` が51件）|
| binary | 153 | 0.5882 | **0.6797** | **+14問**（`'Yes.'` 等）|

**モデルは正解を言っているのに、句読点1つで0点になっていた。**

## 方針
- **意味を変えない範囲でだけ**直す（数値そのものや語彙は書き換えない）
- 判断が要るもの（open_ended / multiple_choice）は **judge が採点するので触らない**
- `time` は個数補正が絡むので `time_postproc.py` が担当（ここでは扱わない）
"""
from __future__ import annotations

import re

# FO クラスの正規名（`FOClass.verify` が受理する語彙。大小文字は不問）
FO_NAMES = {"sponge", "clip", "specimen bag", "silicone loop", "external drain",
            "needle", "gallstone", "specimen", "mesh", "none"}

_TRAIL = " \t\n.。、,；;:：!！?？'\"`*"


def normalize(fmt: str, text: str) -> str:
    """`fmt` の verify を通す形へ整える。無理なら原文をそのまま返す（判定は評価側に委ねる）。"""
    s = str(text).strip()
    if not s:
        return s

    if fmt == "number":
        # 先頭の整数だけ取る。`'1.'` / `'2 clips'` / `'約3'` → `1` / `2` / `3`
        m = re.search(r"\d+", s)
        return m.group(0) if m else s

    if fmt == "binary":
        t = s.strip(_TRAIL).lower()
        if t in ("yes", "no"):
            return t
        # 文頭が yes/no ならそれを採用（`'Yes, it re-appears'`）
        m = re.match(r"^(yes|no)\b", s.strip().lower())
        return m.group(1) if m else s

    if fmt == "percentage":
        m = re.search(r"\d+(?:\.\d+)?", s)
        return m.group(0) if m else s

    if fmt == "fo_class":
        parts = [p.strip(_TRAIL) for p in s.split(",")]
        parts = [p for p in parts if p]
        # 全要素が語彙内なら整形して返す。1つでも語彙外なら**触らない**
        # （勝手に落とすと集合が変わって正解を壊す）
        if parts and all(p.lower() in FO_NAMES for p in parts):
            if any(p.lower() == "none" for p in parts):
                return "none"          # `none` は単独でしか許されない
            return ", ".join(parts)
        return s

    return s        # open_ended / multiple_choice / time は触らない


def normalize_all(rows: list[dict], content_key: str = "content") -> int:
    """responses のリストを in-place で正規化し、変更件数を返す。"""
    n = 0
    for r in rows:
        new = normalize(r.get("fmt", ""), r.get(content_key, ""))
        if new != r.get(content_key):
            r[content_key] = new
            n += 1
    return n


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="正規化で何問拾えるかを測る（採点はしない）")
    ap.add_argument("run_dirs", nargs="+", type=Path)
    a = ap.parse_args()

    def ok(fmt: str, gt: str, pred: str) -> bool:
        g, p = str(gt).strip(), str(pred).strip()
        if fmt == "number":
            return p.isdigit() and g.isdigit() and int(p) == int(g)
        if fmt == "binary":
            return p.lower() in ("yes", "no") and p.lower() == g.lower()
        if fmt == "fo_class":
            nm = lambda s: frozenset(x.strip().lower() for x in s.split(",") if x.strip())
            return nm(p) == nm(g)
        if fmt == "percentage":
            try:
                return abs(float(p.rstrip("%")) - float(g)) < 1e-9
            except Exception:
                return False
        return False

    for d in a.run_dirs:
        rows = json.loads((d / "responses.json").read_text())
        print(f"\n### {d.name}  (n={len(rows)})")
        print("| 形式 | n | 正規化前 | 正規化後 | 差 |")
        print("|---|---|---|---|---|")
        tot = 0
        for fmt in ("number", "binary", "fo_class", "percentage"):
            sub = [r for r in rows if r.get("fmt") == fmt]
            if not sub:
                continue
            b = sum(ok(fmt, r["answer"], r["content"]) for r in sub)
            n2 = sum(ok(fmt, r["answer"], normalize(fmt, r["content"])) for r in sub)
            tot += n2 - b
            print(f"| {fmt} | {len(sub)} | {b/len(sub):.4f} | **{n2/len(sub):.4f}** | {n2-b:+d} |")
        print(f"\n**合計 {tot:+d} 問**")
