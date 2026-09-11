"""2 run を **入力が実際に変わった問だけ**に絞って matched 比較する.

## なぜ必要か
`visible` のように「一部の問でしか入力が変わらない」施策は、全体 2000 問で見ると
**変わらない 1600 問が分母に入って差が埋もれる**。McNemar は不一致ペアだけを見るので
検定自体は歪まないが、**バケット精度の Δ が薄まって読み違える**。
効果の有無は「変わった問の部分集合」で見るのが正しい。

区分:
  changed = uniform と入力フレーム時刻が1つでも違う問（= 匿名があった問）
  clean   = 入力が完全に同一の問（**ここに差が出たら実装バグ or 非決定性**）

Usage:
  .venv/bin/python workspace/expE01_segproc_baseline/subset_compare.py \
     results/eval_dl2_seg32f768_ctrl_n2000 results/eval_dl2_seg32f768_visible_n2000 \
     --labels ctrl visible
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
logging.basicConfig(level=logging.WARNING)
import dataset_seg as D  # noqa: E402


def mcnemar(b: int, c: int) -> float:
    """b=改善数, c=悪化数。不一致ペアのみの二項検定（exact）。"""
    n = b + c
    return 1.0 if n == 0 else binomtest(b, n, 0.5).pvalue


def block(df: pd.DataFrame, name: str, la: str, lb: str) -> None:
    if not len(df):
        print(f"\n### {name}: 該当なし")
        return
    b = int((~df.correct_a & df.correct_b).sum())
    c = int((df.correct_a & ~df.correct_b).sum())
    print(f"\n### {name}  N={len(df)}")
    print(f"| 集合 | N | {la} | {lb} | Δ | 改善 | 悪化 | McNemar p |")
    print("|---|---|---|---|---|---|---|---|")
    print(f"| ALL | {len(df)} | {df.correct_a.mean():.4f} | {df.correct_b.mean():.4f} | "
          f"{df.correct_b.mean()-df.correct_a.mean():+.4f} | {b} | {c} | {mcnemar(b, c):.4f} |")
    for key in ("group", "fmt"):
        for k, g in df.groupby(key):
            if len(g) < 10:
                continue
            bb = int((~g.correct_a & g.correct_b).sum())
            cc = int((g.correct_a & ~g.correct_b).sum())
            if bb + cc == 0:
                continue
            print(f"| {k} | {len(g)} | {g.correct_a.mean():.4f} | {g.correct_b.mean():.4f} | "
                  f"{g.correct_b.mean()-g.correct_a.mean():+.4f} | {bb} | {cc} | "
                  f"{mcnemar(bb, cc):.4f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--mode-b", default="visible", help="run_b のフレーム選定モード")
    a = ap.parse_args()

    da = pd.read_csv(a.run_a / "results_merged.csv")
    db = pd.read_csv(a.run_b / "results_merged.csv")
    m = da[["uid", "fmt", "group", "correct"]].merge(
        db[["uid", "correct"]], on="uid", suffixes=("_a", "_b"))

    kw = dict(track='SEGMENT', fold=0, part='val', n_frames=32, size=768, limit=2000,
              version='v004', extract=False, anchor=True, anon_note=True)
    U = {x.uid: x for x in D.build_samples(frame_select='uniform', **kw)}
    V = {x.uid: x for x in D.build_samples(frame_select=a.mode_b, **kw)}
    changed = {u for u in U if u in V
               and sorted(U[u].frame_times) != sorted(V[u].frame_times)}
    m["changed"] = m.uid.isin(changed)

    la, lb = a.labels
    print(f"# 部分集合 matched 比較: {la} vs {lb}")
    print(f"\n- 入力が変わった問 **{int(m.changed.sum())}/{len(m)}** "
          f"({m.changed.mean():.1%})")
    block(m[m.changed], "★入力が変わった問（ここが効果の本体）", la, lb)
    block(m[~m.changed], "入力が同一の問（差が出たら実装バグ/非決定性）", la, lb)


if __name__ == "__main__":
    main()
