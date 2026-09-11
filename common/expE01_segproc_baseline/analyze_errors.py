"""judge 無しで採点できる形式に絞って、誤りの構造を出す（実行中の run でも使える）.

## なぜ judge 無しで見るか
`time / number / binary / fo_class / percentage` は**決定的に採点できる**（判定コードは公式と同じ規則）。
SEGMENT ではこの5形式で 全体の 77%、PROCEDURE で 87% を占めるので、
**judge を待たずに誤りの構造を掴める**。open_ended / multiple_choice だけは judge が要る。

CLAUDE.md「スコアの前に出力を見ろ」に対応する常設の診断。

Usage:
  .venv/bin/python workspace/expE01_segproc_baseline/analyze_errors.py \
     workspace/expE01_segproc_baseline/results/zs_seg_16f448
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

_TS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")
JUDGED = {"open_ended", "multiple_choice"}      # LLM judge が要る形式


def t_secs(text: str):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    out = []
    for p in parts:
        m = _TS.fullmatch(p)
        if not m:
            return None
        out.append(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]))
    return sorted(out) or None


def judge_free(fmt: str, gt: str, pred: str) -> bool | None:
    """公式規則で決定的に採点する。judge が要る形式は None。"""
    p, g = str(pred).strip(), str(gt).strip()
    if fmt in JUDGED:
        return None
    if fmt == "time":
        a, b = t_secs(g), t_secs(p)
        return b is not None and a is not None and len(a) == len(b) and \
            all(abs(x - y) <= 5 for x, y in zip(a, b))
    if fmt == "number":
        return p.isdigit() and g.isdigit() and int(p) == int(g)
    if fmt == "binary":
        return p.lower() in ("yes", "no") and p.lower() == g.lower()
    if fmt == "fo_class":
        norm = lambda s: frozenset(x.strip().lower() for x in s.split(",") if x.strip())
        return norm(p) == norm(g)
    if fmt == "percentage":
        try:
            return abs(float(p.rstrip("%")) - float(g)) < 1e-9
        except Exception:
            return False
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", default="analyze_errors.md")
    a = ap.parse_args()
    rows = json.loads((a.run_dir / "responses.json").read_text())
    for r in rows:
        r["ok"] = judge_free(r["fmt"], r["answer"], r["content"])

    det = [r for r in rows if r["ok"] is not None]
    L = [f"# 誤り構造（judge 不要形式のみ）— {a.run_dir.name}", "",
         f"- 全 {len(rows)} 問中 **{len(det)} 問（{len(det)/len(rows):.1%}）を決定的に採点**",
         f"- 残り {len(rows)-len(det)} 問（open_ended / multiple_choice）は judge 待ち", ""]

    L += ["## 形式別", "", "| 形式 | n | 正答率 |", "|---|---|---|"]
    for f, c in Counter(r["fmt"] for r in det).most_common():
        s = [r for r in det if r["fmt"] == f]
        L.append(f"| {f} | {len(s)} | **{np.mean([r['ok'] for r in s]):.4f}** |")
    L.append("")

    L += ["## バケット別（judge 不要形式のみなので実際の SCORE とは一致しない）", "",
          "| バケット | n | 正答率 |", "|---|---|---|"]
    for g, c in Counter(r["group"] for r in det).most_common():
        s = [r for r in det if r["group"] == g]
        L.append(f"| {g} | {len(s)} | **{np.mean([r['ok'] for r in s]):.4f}** |")
    L.append("")

    # ── number: 予測分布と GT 分布のズレ（FRAME では「過小分散」が主因だった）──
    num = [r for r in det if r["fmt"] == "number"]
    if num:
        gp = [(int(r["answer"]), int(r["content"])) for r in num
              if str(r["answer"]).isdigit() and str(r["content"]).strip().isdigit()]
        L += ["## number（計数）の分布", "",
              f"- 整数として読めた予測: {len(gp)}/{len(num)}"]
        if gp:
            g = np.array([x for x, _ in gp]); p = np.array([y for _, y in gp])
            L += [f"- GT   : mean {g.mean():.2f} / std {g.std():.2f} / "
                  f"分布 {Counter(g.tolist()).most_common(6)}",
                  f"- PRED : mean {p.mean():.2f} / std {p.std():.2f} / "
                  f"分布 {Counter(p.tolist()).most_common(6)}",
                  f"- 相関 corr(pred,gt) = **{np.corrcoef(g, p)[0,1]:.3f}**",
                  f"- **予測が 0 の割合 {np.mean(p==0):.3f}**（GT が 0 の割合 {np.mean(g==0):.3f}）"]
            L += ["", "| GT | n | 予測平均 | バイアス |", "|---|---|---|---|"]
            for v in sorted(set(g.tolist()))[:10]:
                m = g == v
                L.append(f"| {v} | {m.sum()} | {p[m].mean():.2f} | {p[m].mean()-v:+.2f} |")
        L.append("")

    # ── fo_class: 見落とし / 誤検出をクラス別に ──
    fo = [r for r in det if r["fmt"] == "fo_class"]
    if fo:
        miss, fp_ = Counter(), Counter()
        for r in fo:
            norm = lambda s: {x.strip().lower() for x in str(s).split(",") if x.strip()}
            G, P = norm(r["answer"]), norm(r["content"])
            for c in G - P:
                miss[c] += 1
            for c in P - G:
                fp_[c] += 1
        L += ["## fo_class の見落とし / 誤検出（クラス別）", "",
              f"- 完全一致 {np.mean([r['ok'] for r in fo]):.4f} (n={len(fo)})", "",
              "| クラス | 見落とし | 誤検出 |", "|---|---|---|"]
        for c in sorted(set(miss) | set(fp_), key=lambda x: -(miss[x] + fp_[x]))[:12]:
            L.append(f"| {c} | {miss[c]} | {fp_[c]} |")
        L.append("")

    # ── 形式違反（verify に落ちる＝確実に0点）──
    bad = [r for r in det if not r["ok"] and judge_free(r["fmt"], r["content"], r["content"]) is False]
    viol = Counter()
    for r in det:
        f, p = r["fmt"], str(r["content"]).strip()
        if f == "number" and not p.isdigit():
            viol["number が整数でない"] += 1
        elif f == "binary" and p.lower() not in ("yes", "no"):
            viol["binary が yes/no でない"] += 1
        elif f == "time" and t_secs(p) is None:
            viol["time が hh:mm:ss でない"] += 1
    if viol:
        L += ["## 形式違反（`verify` に落ちる = 確実に0点）", "",
              "| 症状 | n |", "|---|---|"]
        for k, v in viol.most_common():
            L.append(f"| {k} | {v} |")
        L.append("")

    # ── 誤答の実例 ──
    L += ["## 誤答の実例（形式ごとに3件）", ""]
    for f in sorted({r["fmt"] for r in det}):
        ex = [r for r in det if r["fmt"] == f and not r["ok"]][:3]
        if not ex:
            continue
        L.append(f"### {f}")
        for r in ex:
            L += [f"- Q: {r['question'][:130]}",
                  f"  - PRED `{r['content']}` / GT `{r['answer']}`"
                  f"  （clip {r['end_time']-r['start_time']:.0f}s）"]
        L.append("")

    txt = "\n".join(L)
    (a.run_dir / a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
