r"""どの時刻のフレームを見るかを決める（`workspace/expE01_segproc_baseline/dataset_seg.py` と同一仕様）.

学習・CV とバイト単位で同じ入力を作るのが目的なので、**元実装の挙動を変えない**こと。
元実装との対応:
  `sample_times` / `question_anchor_times` / `hhmmss` … dataset_seg.py からそのまま移植

## 仕様のポイント
- `[start, end]` を n 点で一様サンプル（**両端を含む**。`first/last visible` 系が端に集中するため）
- **格子は start 原点**（`start + grid*k`）。本番クリップは窓に切り出し済みで先頭がキーフレーム
- 格子点数が n に満たなければ**格子の全点**を返す（水増ししない）
- アンカー: 問題文中の `hh:mm:ss` のフレームを必ず含める（最も近い一様点を置換、**総枚数は不変**）
  該当は SEGMENT の 17.1%
"""
from __future__ import annotations

import math
import re

SEGMENT_GRID_S = 1.0

_TS_RE = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")


def hhmmss(sec: float) -> str:
    s = int(round(max(sec, 0)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def question_anchor_times(question: str) -> list[float]:
    """問題文中の `hh:mm:ss` を秒（**絶対時刻**）にして返す。

    ★回答フォーマット指示の中の `hh:mm:ss` はリテラルの書式指定（数字でない）なので
      正規表現に一致せず誤検出しない。
    """
    return [float(int(h) * 3600 + int(m) * 60 + int(s))
            for h, m, s in _TS_RE.findall(str(question))]


def sample_times(start: float, end: float, n: int, grid: float = SEGMENT_GRID_S,
                 anchors: list[float] | None = None) -> list[float]:
    """[start, end] を n 点で一様サンプル → 格子へスナップして重複除去（**絶対時刻**を返す）。"""
    span = end - start
    if n <= 1 or span <= 0:
        pts = [(start + end) / 2.0]
    else:
        step = span / (n - 1)
        pts = [start + i * step for i in range(n)]

    if anchors:
        for a in sorted({min(max(float(x), start), end) for x in anchors}):
            if not pts:
                pts = [a]
                continue
            j = min(range(len(pts)), key=lambda i: abs(pts[i] - a))
            pts[j] = a

    if grid <= 0:
        return sorted({round(p, 3) for p in pts})
    # end を超える格子点は実体が無いので k を [0, floor(span/grid)] にクランプする
    k_max = int(math.floor(span / grid + 1e-9))
    snapped = {min(max(int(round((p - start) / grid)), 0), k_max) for p in pts}
    return sorted(round(start + k * grid, 3) for k in snapped)
