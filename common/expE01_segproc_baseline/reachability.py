"""「一様サンプリングでは原理的に届かない問題」がどれだけあるかを数える.

## 動機
PROCEDURE の実例で `How many separate times does a Sponge completely leave the field of view?`
の **GT が 28** だった（90分の動画）。32フレームの一様サンプルで 28回の出入りは**数えられない**。
モデルを良くしても届かない領域がどれだけあるかを先に把握しないと、投資先を誤る。

## 判定する2種類の「原理的な上限」
1. **`time` の分解能**: 刻み = 文脈長/(n_frames−1)。**刻み > 5s** なら、
   サンプルした時刻の中に正解が無いので ±5s に入らない（端の偶然を除く）。
2. **イベント計数の分解能**: GT が N 回の出入りなら、
   最低でも **2N フレーム**（各イベントの前後）を見ないと数え上げられない（ナイキスト相当）。
   `n_frames < 2·GT` の問題は原理的に不可能。

Usage:
  .venv/bin/python workspace/expE01_segproc_baseline/reachability.py --n-frames 16 32 64 128
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "expE00_segproc_eda"


def secs(a: str):
    try:
        return [sum(int(x) * m for x, m in zip(p.strip().split(":"), (3600, 60, 1)))
                for p in str(a).split(",")]
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--tol", type=int, default=5)
    a = ap.parse_args()
    df = pd.read_csv(OUT / "all_qa.csv")

    L = ["# 一様サンプリングの原理的な到達可能性", "",
         "「モデルを良くしても届かない」問題の割合。投資先を決める前に確認する。", ""]

    # ── 1. time の分解能 ──
    L += ["## 1. `time`: 刻みが ±5s より粗い問題の割合", "",
          "刻み = 文脈長 /(n_frames−1)。粗いとサンプル時刻の中に正解が無い。", "",
          "| track | time問題数 | " + " | ".join(f"{n}f" for n in a.n_frames) + " |",
          "|---" * (2 + len(a.n_frames)) + "|"]
    for track in ("SEGMENT", "PROCEDURE"):
        sub = df[(df.track == track) & (df.answer_format == "time")]
        cells = []
        for n in a.n_frames:
            stride = sub["duration"] / max(n - 1, 1)
            cells.append(f"**{(stride > a.tol).mean():.3f}**")
        L.append(f"| {track} | {len(sub)} | " + " | ".join(cells) + " |")
    L.append("")
    L += ["※ SEGMENT は 64f でほぼ解消するが、**PROCEDURE は 128f でもほぼ全滅**",
          "（平均文脈 4405s ÷ 127 = 35s 刻み ≫ 5s）。",
          "→ **PROCEDURE の `time` は一様サンプリングでは解けない。粗→密の2段探索が必須**", ""]

    # ── 2. イベント計数の分解能 ──
    # 「何回 視野から出るか / 何回 現れるか」型の計数は、GT 回数の2倍のフレームが要る
    EVENT_Q = r"how many separate times|how many times|how often"
    L += ["## 2. イベント計数: `n_frames < 2×GT` で原理的に不可能な問題の割合", "",
          "N 回の出入りを数えるには最低 2N フレーム要る（ナイキスト相当）。", "",
          "| track | 該当問題数 | GT中央値 | GT最大 | " +
          " | ".join(f"{n}f" for n in a.n_frames) + " |",
          "|---" * (4 + len(a.n_frames)) + "|"]
    for track in ("SEGMENT", "PROCEDURE"):
        sub = df[(df.track == track) & (df.answer_format == "number") &
                 (df.question.str.contains(EVENT_Q, case=False, regex=True))].copy()
        if sub.empty:
            continue
        sub["gtv"] = pd.to_numeric(sub["answer"], errors="coerce")
        sub = sub[sub["gtv"].notna()]
        cells = [f"**{(sub["gtv"] * 2 > n).mean():.3f}**" for n in a.n_frames]
        L.append(f"| {track} | {len(sub)} | {sub["gtv"].median():.0f} | {sub["gtv"].max():.0f} | "
                 + " | ".join(cells) + " |")
    L.append("")

    # GT 分布（どこまで大きい値が出るか）
    for track in ("SEGMENT", "PROCEDURE"):
        sub = df[(df.track == track) & (df.answer_format == "number")].copy()
        sub["gtv"] = pd.to_numeric(sub["answer"], errors="coerce")
        sub = sub[sub["gtv"].notna()]
        L += [f"### {track} の number GT 分布 (n={len(sub)})", "",
              f"- 分位: p50={sub["gtv"].quantile(.5):.0f} p90={sub["gtv"].quantile(.9):.0f} "
              f"p99={sub["gtv"].quantile(.99):.0f} max={sub["gtv"].max():.0f}",
              f"- **GT > 10 の割合: {(sub["gtv"] > 10).mean():.3f}** / GT > 20: {(sub["gtv"] > 20).mean():.3f}", ""]

    txt = "\n".join(L)
    (Path(__file__).parent / "reachability.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
