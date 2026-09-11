"""zero-shot VLM を「映像を一切見ない事前分布」と **同一問題で** 突き合わせる.

## なぜ必要か
expE00 で、映像を見ない prior だけで SEGMENT 0.3187 / PROCEDURE 0.2474 が出た
（FRAME の zero-shot 0.226 を上回る）。つまり **prior が採用の下限**であり、
prior に負けるバケットは VLM でなく prior を使うべき。

比較は必ず **同一 qID** で行う（形式構成が違うと平均が動くため）。
prior は **train fold だけ**から作る（`time` は区間中点、他はテンプレ→train 最頻回答）。

Usage:
  .venv/bin/python workspace/expE01_segproc_baseline/vs_prior.py \
     workspace/expE01_segproc_baseline/results/zs_seg_16f448 --track SEGMENT
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from analyze_errors import judge_free  # noqa: E402
from dataset_seg import ROOT, _qa_index  # noqa: E402
from time_postproc import hhmmss  # noqa: E402


def build_prior(track: str, val_fold: int, version: str = "v004"):
    """train fold の (質問テンプレ → 最頻回答) と (形式 → 最頻回答)。`time` は別扱い。"""
    idx = _qa_index(track)
    tmpl: dict[str, Counter] = {}
    fmt: dict[str, Counter] = {}
    with (ROOT / f"workspace/fold/qa_{version}/qa_split.csv").open() as f:
        for r in csv.DictReader(f):
            if r["track"] != track or int(r["fold"]) == val_fold:
                continue
            if r["answer_format"] == "time":
                continue                      # 絶対時刻なので最頻値は無意味
            got = idx.get((r["dataset"], r["qID"]))
            if got is None:
                continue
            q, ref = got[0].question, str(got[1].answer)
            tmpl.setdefault(" ".join(q.split()[:8]), Counter())[ref] += 1
            fmt.setdefault(r["answer_format"], Counter())[ref] += 1
    return ({k: c.most_common(1)[0][0] for k, c in tmpl.items()},
            {k: c.most_common(1)[0][0] for k, c in fmt.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--track", default="SEGMENT")
    ap.add_argument("--val-fold", type=int, default=0)
    a = ap.parse_args()

    rows = json.loads((a.run_dir / "responses.json").read_text())
    tmpl, fmt = build_prior(a.track, a.val_fold)

    def prior_answer(r):
        if r["fmt"] == "time":
            # 映像を見ない最良: 与えられた区間の中点（正解の96-100%が区間内）
            return hhmmss((r["start_time"] + r["end_time"]) / 2)
        return tmpl.get(" ".join(r["question"].split()[:8]),
                        fmt.get(r["fmt"], "none"))

    recs = []
    for r in rows:
        v = judge_free(r["fmt"], r["answer"], r["content"])
        if v is None:
            continue                          # judge が要る形式は除外（両者とも判定不能）
        p = judge_free(r["fmt"], r["answer"], prior_answer(r))
        recs.append((r["group"], r["fmt"], bool(v), bool(p)))
    if not recs:
        raise SystemExit("判定できる問題が無い")

    V = np.array([x[2] for x in recs]); P = np.array([x[3] for x in recs])
    L = [f"# zero-shot VLM vs 映像を見ない prior（同一問題）— {a.run_dir.name}", "",
         "judge が要る形式（open_ended / multiple_choice）は両者とも判定不能なので除外。",
         "prior は **train fold のみ**から構築（`time` は区間中点）。", "",
         f"- 比較対象 **N = {len(recs)}**", ""]

    def blk(mask, name):
        v, p = V[mask], P[mask]
        b = int((p & ~v).sum())   # prior 正・VLM 誤 = VLM の負け
        c = int((~p & v).sum())   # VLM 正・prior 誤 = VLM の勝ち
        pv = 1.0 if b + c == 0 else float(binomtest(c, b + c, 0.5).pvalue)
        star = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "n.s."
        win = "**VLM**" if (v.mean() > p.mean() and pv < 0.05) else \
              ("**prior**" if (p.mean() > v.mean() and pv < 0.05) else "—")
        return (f"| {name} | {mask.sum()} | {p.mean():.4f} | {v.mean():.4f} | "
                f"{v.mean()-p.mean():+.4f} | {c} | {b} | {pv:.4f} {star} | {win} |")

    hdr = ["| 集合 | N | prior | zero-shot | Δ | VLM勝 | VLM負 | McNemar p | 優位 |",
           "|---|---|---|---|---|---|---|---|---|"]
    L += ["## 全体", ""] + hdr + [blk(np.ones(len(recs), bool), "ALL")] + [""]
    for col, i in (("バケット", 0), ("形式", 1)):
        keys = sorted({x[i] for x in recs})
        L += [f"## {col}別", "",
              "⚠️ N が小さいバケットは McNemar p を見ること", ""] + hdr
        for k in keys:
            L.append(blk(np.array([x[i] == k for x in recs]), k))
        L.append("")

    # バケット非加重平均（公式 SCORE の構造）
    gs = sorted({x[0] for x in recs})
    sv = np.mean([V[[x[0] == g for x in recs]].mean() for g in gs])
    sp = np.mean([P[[x[0] == g for x in recs]].mean() for g in gs])
    L += ["## バケット非加重平均（公式 SCORE の構造）", "",
          f"- prior: **{sp:.4f}** / zero-shot: **{sv:.4f}** / Δ **{sv-sp:+.4f}**", ""]
    txt = "\n".join(L)
    (a.run_dir / "vs_prior.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
