"""2つの run を **同一 qID で matched 比較**し、(改善数, 悪化数, McNemar p) を出す.

## なぜ Δ だけで判断してはいけないか
FRAME で FO 分類器ヒントを「バケット平均 +0.046」だけ見て採用し、LB で −0.020 を食った。
matched で再検算すると **改善51 / 悪化38 = 正味+13, 不一致89 → McNemar p=0.203** で
**最初から有意ではなかった**。以来「Δ ではなく (改善数, 悪化数, McNemar p) で判断する」が運用ルール。

さらに SEGMENT/PROCEDURE では **バケットが5つあり、小バケットは N が二桁**
（PROCEDURE の EVENT_UNDERSTANDING は val 推定 19問）。
バケット別の差を語るときは **必ず N を併記する**こと。

Usage:
  .venv/bin/python workspace/expE01_segproc_baseline/compare_runs.py \
     results/zs_seg_16f448 results/zs_seg_64f448_n2000 --labels 16f 64f
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def load(run_dir: Path) -> pd.DataFrame:
    f = run_dir / "results_merged.csv"
    if not f.exists():
        raise SystemExit(f"{f} が無い。先に eval_seg.py を通すこと")
    d = pd.read_csv(f)
    d["key"] = d["dataset"].astype(str) + ":" + d["qID"].astype(str)   # ★qID 単体は一意でない
    return d


def mcnemar(b: int, c: int) -> float:
    """不一致ペア (b=A正/B誤, c=A誤/B正) の二項検定（exact McNemar）。"""
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--by", nargs="*", default=["group", "fmt", "dataset"],
                    help="この列ごとにも内訳を出す")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    la, lb = a.labels

    A, B = load(a.run_a), load(a.run_b)
    m = A.merge(B, on="key", suffixes=("_a", "_b"))
    if m.empty:
        raise SystemExit("共通 qID が無い")
    ca, cb = m["correct_a"].astype(bool), m["correct_b"].astype(bool)

    lines = [f"# matched 比較: {la} ({a.run_a.name}) vs {lb} ({a.run_b.name})", "",
             f"- 共通問題数 **N = {len(m)}**"
             f"（{la} は {len(A)} 問 / {lb} は {len(B)} 問）", ""]

    def block(sub: pd.DataFrame, name: str) -> list[str]:
        x, y = sub["correct_a"].astype(bool), sub["correct_b"].astype(bool)
        b_ = int((x & ~y).sum())      # A だけ正解 = B で悪化
        c_ = int((~x & y).sum())      # B だけ正解 = B で改善
        p = mcnemar(b_, c_)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        return [f"| {name} | {len(sub)} | {x.mean():.4f} | {y.mean():.4f} | "
                f"{y.mean()-x.mean():+.4f} | {c_} | {b_} | {p:.4f} {star} |"]

    hdr = ["| 集合 | N | " + la + " | " + lb + " | Δ | 改善 | 悪化 | McNemar p |",
           "|---|---|---|---|---|---|---|---|"]
    lines += ["## 全体", ""] + hdr + block(m, "ALL") + [""]

    for col in a.by:
        ca_col = f"{col}_a"
        if ca_col not in m.columns:
            continue
        rows = []
        for v, sub in m.groupby(ca_col):
            rows += block(sub, str(v))
        lines += [f"## {col} 別", "",
                  "⚠️ N が小さいバケットの差は McNemar p を見ること（差が1〜2問でも Δ は大きく見える）", ""]
        lines += hdr + rows + [""]

    # バケット非加重平均（＝公式 SCORE の構造）でも見る
    if "group_a" in m.columns:
        sa = m.groupby("group_a")["correct_a"].mean().mean()
        sb = m.groupby("group_a")["correct_b"].mean().mean()
        lines += ["## バケット非加重平均（公式 SCORE の構造。ただし ID/OOD 分割は無し）", "",
                  f"- {la}: **{sa:.4f}**", f"- {lb}: **{sb:.4f}**", f"- Δ: **{sb-sa:+.4f}**", ""]

    txt = "\n".join(lines)
    out = Path(a.out) if a.out else a.run_b / f"compare_vs_{a.run_a.name}.md"
    out.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
