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


# ── ★SEGMENT 質問文ルール（expN00, 2026-09-03 採用）────────────────────────
# `dataset_seg.segrules_select_times` の**逐語移植**（コンテナと CV で同一挙動にする）。
# val 実測（fold v003, 4,006問）: 発火 367（between 149 / after-T 218）、
# 非発火 3,639 問は uniform+anchor と**ビット同一**（assert 済み）。
#   between : [T1,T2] へ予算を絞る（窓外は質問と論理的に無関係）
#   after-T : [T,end] へ密化（GT < T は scored 96 問中 0 件で論理保証）
# CV 効果: s0 で SCORE +0.0174（悪化バケット 0）、s1 で +0.0018。
_SEG_BETWEEN_RE = re.compile(
    r"[Ww]hat types of foreign objects are seen between "
    r"(\d{1,2}:\d{2}:\d{2}) and (\d{1,2}:\d{2}:\d{2})")
_SEG_AFTER_RE = re.compile(
    r"in the frame at (\d{1,2}:\d{2}:\d{2})\.? +When is it retrieved")


def _hms2s(t: str) -> float:
    h, m, s = t.split(":")
    return float(int(h) * 3600 + int(m) * 60 + int(s))


def segrules_times(question: str, start: float, end: float,
                   n: int, grid: float) -> list[float] | None:
    """発火したら窓内の時刻列、しなければ None（呼び出し側が uniform+anchor へ委譲）。"""
    q = str(question)
    lo, hi = float(start), float(end)
    m = _SEG_BETWEEN_RE.search(q)
    if m:
        t1 = max(lo, min(hi, _hms2s(m.group(1))))
        t2 = max(lo, min(hi, _hms2s(m.group(2))))
        if t2 > t1:
            return sample_times(t1, t2, n, grid, [t1, t2])
        return None
    m = _SEG_AFTER_RE.search(q)
    if m:
        t = max(lo, min(hi, _hms2s(m.group(1))))
        if hi > t:
            return sample_times(t, hi, n, grid, [t])
    return None
