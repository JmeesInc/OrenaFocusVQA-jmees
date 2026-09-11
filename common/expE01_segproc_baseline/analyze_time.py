"""`time` 形式の誤りを分解する — 「個数」で落ちているのか「精度」で落ちているのか.

## なぜこれを最初に見るか
`Time.compare` は **①タイムスタンプ個数の完全一致 ②各ペアが ±5s 以内** の両方を要求する
（`formats.py:298`）。つまり時刻が合っていても**個数が違えば即不正解**。
smoke で `PRED '00:13:28, 00:13:30'` vs `GT '00:13:27'` を観測した（00:13:28 は ±5s 以内なのに不正解）。

TEMPORAL_GROUNDING は SEGMENT / PROCEDURE の配点の **20%** で、その96%が `time`。
「個数で落ちている」なら**後処理だけで取れる**ので、モデルを触る前に切り分ける。

## GT 側の個数分布（expE00 実測）
- SEGMENT: 1個 7504 / 2個 61 / 3個 47 / …  → **98.2% が1個**
- PROCEDURE: 1個 3535 / …                  → **96.9% が1個**

Usage:
  .venv/bin/python workspace/expE01_segproc_baseline/analyze_time.py \
     workspace/expE01_segproc_baseline/results/zs_seg_16f448
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def secs(text: str) -> list[int] | None:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    out = []
    for p in parts:
        m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", p)
        if not m:
            return None                      # 形式不正（verify に落ちる = 不正解）
        out.append(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]))
    return sorted(out) or None


def ok(gt: list[int], pr: list[int] | None, tol: int = 5) -> bool:
    if pr is None or len(gt) != len(pr):
        return False
    return all(abs(a - b) <= tol for a, b in zip(gt, pr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tol", type=int, default=5)
    a = ap.parse_args()

    rows = [r for r in json.loads((a.run_dir / "responses.json").read_text())
            if r.get("fmt") == "time"]
    L = [f"# `time` 誤りの分解 — {a.run_dir.name}", "", f"- time 問題数 **N = {len(rows)}**", ""]
    if not rows:
        print("\n".join(L)); return

    n_ok = n_badfmt = 0
    cnt_pairs = Counter()
    errs = []                    # 個数が合っているときの |誤差| 秒
    gain_first = 0               # 「先頭1個だけ返す」に変えたら正解になる数
    lose_first = 0               # 逆に不正解になる数
    for r in rows:
        gt = secs(r["answer"])
        pr = secs(r["content"])
        if gt is None:
            continue
        if pr is None:
            n_badfmt += 1
        cnt_pairs[(len(gt), len(pr) if pr else 0)] += 1
        base = ok(gt, pr, a.tol)
        n_ok += base
        if pr and len(gt) == len(pr):
            errs += [abs(x - y) for x, y in zip(gt, pr)]
        # 後処理案: 予測を先頭1個に切り詰める
        pr1 = pr[:1] if pr else None
        alt = ok(gt, pr1, a.tol)
        gain_first += (alt and not base)
        lose_first += (base and not alt)

    N = len(rows)
    L += ["## 現状", "",
          f"- 正解 **{n_ok}/{N} = {n_ok/N:.4f}**",
          f"- 形式不正（hh:mm:ss で無い）: {n_badfmt} ({n_badfmt/N:.3f})", ""]

    L += ["## 落ちている原因の切り分け", "", "| GT個数 | PRED個数 | n | 割合 |", "|---|---|---|---|"]
    for (g, p), c in sorted(cnt_pairs.items(), key=lambda kv: -kv[1])[:12]:
        flag = "" if g == p else "  ← **個数不一致=即不正解**"
        L.append(f"| {g} | {p} | {c} | {c/N:.3f}{flag} |")
    mism = sum(c for (g, p), c in cnt_pairs.items() if g != p)
    L += ["", f"- **個数不一致だけで {mism}/{N} = {mism/N:.4f} を失っている**", ""]

    if errs:
        e = np.array(errs)
        L += ["## 個数が合っている場合の時刻誤差（秒）", "",
              f"- n={len(e)}  median **{np.median(e):.1f}s**  mean {e.mean():.1f}s",
              f"- ±5s 以内: **{(e<=5).mean():.4f}**  / ±10s: {(e<=10).mean():.4f}"
              f"  / ±20s: {(e<=20).mean():.4f}",
              f"- 分位: p25={np.percentile(e,25):.0f} p50={np.percentile(e,50):.0f} "
              f"p75={np.percentile(e,75):.0f} p90={np.percentile(e,90):.0f}", ""]

    # ── 後処理案の比較: 複数個返してきたものを1個に落とす規則を総当たり ──
    # ★GT の 98.2%(SEGMENT) / 96.9%(PROCEDURE) は**タイムスタンプ1個**なので、
    #   個数不一致は「多く返しすぎ」が支配的。どの1個を残すかで結果が変わる。
    def rule_first(p, r): return p[:1]
    def rule_last(p, r): return p[-1:]
    def rule_median(p, r): return [int(np.median(p))]
    def rule_mid(p, r):
        c = (r["start_time"] + r["end_time"]) / 2
        return [min(p, key=lambda x: abs(x - c))]

    RULES = {"先頭": rule_first, "末尾": rule_last, "中央値": rule_median,
             "区間中点に最も近い": rule_mid}
    L += ["## 後処理案: 複数返しを1個に落とす規則の比較", "",
          "★根拠: GT の 98.2%(SEGMENT) / 96.9%(PROCEDURE) は**タイムスタンプ1個**。",
          "適用条件は「GT の個数は不明なので、**予測が2個以上のときだけ**1個に落とす」。", "",
          "| 規則 | 改善 | 悪化 | 正味 | time 精度 |", "|---|---|---|---|---|"]
    for name, fn in RULES.items():
        g = l = 0
        for r in rows:
            gt, pr = secs(r["answer"]), secs(r["content"])
            if gt is None:
                continue
            base = ok(gt, pr, a.tol)
            alt = ok(gt, fn(pr, r) if (pr and len(pr) > 1) else pr, a.tol)
            g += (alt and not base); l += (base and not alt)
        L.append(f"| {name} | +{g} | −{l} | **{g-l:+d}** | "
                 f"{n_ok/N:.4f} → **{(n_ok+g-l)/N:.4f}** |")
    L.append("")

    # ── 個数を **train fold の質問テンプレ統計** から決める（val で調律しない）──
    # ★上の「先頭/末尾/中央値」は val 上で勝った規則を選んでおり、**val への過学習**。
    #   GT の個数は質問テンプレでほぼ決まる（"at what time ..." は1個、
    #   "at which time points ... for each individual clip instance" は複数）ので、
    #   **train fold の GT 個数の最頻値**を使えば val を見ずに個数を決められる。
    try:
        import csv as _csv
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from dataset_seg import ROOT as _R, _qa_index
        track = rows[0].get("track") or json.loads(
            (a.run_dir / "meta.json").read_text()).get("track", "SEGMENT") \
            if (a.run_dir / "meta.json").exists() else "SEGMENT"
        val_fold = 0
        tmpl_counts: dict[str, Counter] = {}
        idx = _qa_index(track)
        with (_R / "workspace/fold/qa_v004/qa_split.csv").open() as f:
            for r in _csv.DictReader(f):
                if r["track"] != track or r["answer_format"] != "time":
                    continue
                if int(r["fold"]) == val_fold:
                    continue                      # ★train fold だけで統計を取る
                got = idx.get((r["dataset"], r["qID"]))
                if got is None:
                    continue
                s = secs(str(got[1].answer))
                if s is None:
                    continue
                key = " ".join(got[0].question.split()[:8])
                tmpl_counts.setdefault(key, Counter())[len(s)] += 1
        prior = {k: c.most_common(1)[0][0] for k, c in tmpl_counts.items()}
        g = l = n_hit = 0
        for r in rows:
            gt, pr = secs(r["answer"]), secs(r["content"])
            if gt is None:
                continue
            key = " ".join(r["question"].split()[:8])
            k = prior.get(key)
            n_hit += (k is not None)
            base = ok(gt, pr, a.tol)
            if pr and k and len(pr) != k:
                alt_p = sorted(pr)[-k:] if len(pr) > k else pr    # 多い側だけ後ろから k 個
            else:
                alt_p = pr
            alt = ok(gt, alt_p, a.tol)
            g += (alt and not base); l += (base and not alt)
        L += ["## 後処理案(本命): 個数を **train fold のテンプレ統計**から決める", "",
              "★上の先頭/末尾/中央値は **val 上で勝った規則を選んでおり val への過学習**。",
              "テンプレ→GT個数の最頻値は train fold だけで作れるので、val を見ずに決まる。", "",
              f"- テンプレが train に存在した割合: **{n_hit}/{N} = {n_hit/N:.3f}**",
              f"- 改善 **+{g}** / 悪化 **−{l}** / 正味 **{g-l:+d}**",
              f"- time 精度 {n_ok/N:.4f} → **{(n_ok+g-l)/N:.4f}**", ""]
    except Exception as e:                        # 分析の失敗で全体を落とさない
        L += [f"（テンプレ統計版の評価に失敗: {e}）", ""]

    # ── サンプリング刻み仮説の検証 ──
    # 「±5s に届かないのはフレーム間隔で律速されているから」なら、
    # **クリップが長いほど（=刻みが粗いほど）誤差が大きい**はず。
    # ★meta.json は推論完走時にしか書かれない。実行中の run も分析したいので
    #   responses の `n_frames`（実際に渡した枚数の最大 = 要求フレーム数）から復元する。
    nf = 0
    try:
        nf = int(json.loads((a.run_dir / "meta.json").read_text())["n_frames"])
    except Exception:
        nf = max((int(r.get("n_frames", 0)) for r in rows), default=0)
    rec = []
    for r in rows:
        gt, pr = secs(r["answer"]), secs(r["content"])
        if gt is None or pr is None or len(gt) != len(pr):
            continue
        dur = r["end_time"] - r["start_time"]
        rec.append((dur, np.mean([abs(x - y) for x, y in zip(gt, pr)]),
                    ok(gt, pr, a.tol)))
    if rec and nf:
        L += [f"## サンプリング刻み仮説の検証（n_frames={nf}）", "",
              "刻み = クリップ長 /(n_frames−1)。刻みが ±5s より粗ければ、"
              "**サンプルした時刻の中に正解が無い**ので原理的に当たらない。", "",
              "| クリップ長 | 刻み(秒) | n | 誤差中央値(秒) | ±5s 正答率 |", "|---|---|---|---|---|"]
        for lo, hi in [(0, 40), (40, 130), (130, 210), (210, 301)]:
            sub = [(d, e, o) for d, e, o in rec if lo <= d < hi]
            if not sub:
                continue
            dm = np.mean([d for d, _, _ in sub])
            L.append(f"| {lo}–{hi}s (平均{dm:.0f}s) | **{dm/max(nf-1,1):.1f}** | {len(sub)} | "
                     f"{np.median([e for _, e, _ in sub]):.1f} | "
                     f"**{np.mean([o for _, _, o in sub]):.4f}** |")
        L.append("")
    txt = "\n".join(L)
    (a.run_dir / "analyze_time.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
