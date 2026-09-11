"""SEGMENT / PROCEDURE の VQA データセット（複数フレーム入力）.

## 設計の理由

**公式 `FocusVideoDataset` を使わない**: あれは1サンプルごとに ffmpeg でクリップを再エンコードして
/tmp に MP4 を書く。`[0, 17780]s` の PROCEDURE で毎回それをやると学習が回らない。
FRAME と同じく **ffmpeg で必要な時刻のフレームだけ JPEG 抽出してキャッシュ**する。

**キャッシュキーは (videoID, 時刻) — qID ではない**。同じ動画の重なったクリップに対する質問が
大量にあるので、時刻を **1秒グリッドにスナップ**して共有すると抽出コストが激減する
（FRAME の `frames_cache` が qID キーだったのは1問1フレームだったから）。

**タイムスタンプは焼き込まず、各フレームの直前にテキストで置く**。
- 公式 `VideoTimestampOverlayPreprocessor` は全動画を再エンコードする（数百GB・数十時間）
- `Request.start_time` は推論時にも与えられるので、時刻は**こちらが知っている情報**であり、
  モデルに OCR させる必要がない。テキストで渡す方が正確でコストもゼロ
- ⚠️ 提出コンテナでも同じ渡し方をすること（学習と推論で入力形式を揃える）
"""
from __future__ import annotations

import csv
import logging
import math
import re
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/focus"
HERE = Path(__file__).parent
CACHE_ROOT = HERE / "frames_cache"

sys.path.insert(0, str(ROOT / "reference/src"))
sys.path.insert(0, str(HERE))

from focus import FocusConfig, FocusDataset, set_config  # noqa: E402
from focus.enums import DatasetSplit, Track  # noqa: E402

from prompts_seg import build_system_prompt  # noqa: E402

log = logging.getLogger("expE01.data")


@dataclass
class VideoVQASample:
    uid: str
    qID: str
    dataset: str
    videoID: str
    track: str
    start_time: float
    end_time: float
    frame_times: list[float]          # 絶対秒（昇順）
    frame_paths: list[Path]
    system_prompt: str
    question: str
    answer: str
    fmt: str
    primary: str = ""
    group: str = ""
    # ★`frame_paths` の実体解像度と、モデルに渡したい解像度が違うことがある。
    #   貸しGPU では転送量を抑えるため **768px だけをキャッシュ**し、448/560 は
    #   読み込み時に縮小する運用。None なら実体のまま使う。
    target_size: int | None = None
    # ★[start,end] と重なる匿名化区間。**本文でモデルに明示する**ために持つ。
    anon_ranges: list = field(default_factory=list)
    # ★★`Request.procedure_type`（例 "Laparoscopic Cholecystectomy"）。
    #   **推論時にも公式から与えられる**（コンテナの request.json に実在）のに、
    #   これまで一切使っていなかった。学習は4術式のみ（Proctocolectomy /
    #   Rectal Resection / Sigmoid Resection / Laparoscopic Cholecystectomy）なので、
    #   テストでこれ以外が来たら **文字列比較だけで OOD を確定できる**。
    procedure_type: str = ""
    # ★★instseg 重畳（expM00）。`overlay_path` があるときだけ 2枚目の画像として渡す。
    #   None = 単画像（CONTROL arm、または検出0件で重畳を作らなかった問）。
    #   ⚠️ **検出0件でも2枚目を付ける旧実装は禁止**（expG00 §39: 空の重畳は −0.0429）。
    #   `overlay_note` は user ターンに置く説明文（system は LoRA がほぼ無視する）。
    overlay_path: str | None = None
    overlay_note: str = ""
    meta: dict = field(default_factory=dict)


def hhmmss(sec: float) -> str:
    s = int(round(max(sec, 0)))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


# ── サンプリンググリッド ──────────────────────────────────────────────
# 公式提出テンプレート（IMSY-DKFZ/orena-focus-submission-template, procedure-algorithm/
# inference.py の docstring）が配布クリップの符号化を明記している:
#
#   "Clips are H.264 MP4 at exactly 5 fps (one frame every 0.2 s), height-normalised
#    to at most 576 px. ... A keyframe every 5 s keeps seeking cheap."
#
# ★**キーフレーム格子（5秒）より細かい時刻は本番でタダでは取れない**。非キーフレームを
#   読むには前のキーフレームからデコードし直す必要があり、その時間は latency 予算に入る
#   （同 docstring: "decoding counts towards your latency budget"）。
#   実測（4 vCPU 相当・仕様どおりの 1024x576/5fps/-g 25 クリップ, 1200s）:
#       ffmpeg -skip_frame nokey 全走査 (241枚) ...  2.7 s
#       ffmpeg 全フレーム走査      (6000枚) ...  9.9 s
#       decord get_batch 241枚 num_threads=1 ... 29.1 s / =4 ... 9.0 s
#   PROCEDURE は平均 4691s のクリップを 1問30秒で処理するので、**探索はキーフレーム上に
#   限る**のが唯一まともに収まる設計。学習側も同じ格子に載せて入力分布を揃える。
#
# ★`time` 回答の許容誤差はちょうど ±5s なので、5秒格子は精度上の損失にもならない。
KEYFRAME_GRID_S = 5.0

# トラック別の既定グリッド。**コスト制約が効くのは PROCEDURE だけ**なので、
# SEGMENT（<=5min = 全フレーム読んでも 1s 未満）と FRAME は 1 秒格子のままにする。
# 揃えたい場合は grid=KEYFRAME_GRID_S を明示的に渡す。
TRACK_GRID: dict[str, float] = {"FRAME": 1.0, "SEGMENT": 1.0, "PROCEDURE": KEYFRAME_GRID_S}


def grid_for(track: str) -> float:
    """トラックの既定サンプリンググリッド（秒）。"""
    return TRACK_GRID.get(str(track).upper(), 1.0)


def sample_times(start: float, end: float, n: int, grid: float = 1.0,
                 anchors: list[float] | None = None) -> list[float]:
    """[start, end] を n 点で一様サンプル（両端含む）→ グリッドにスナップして重複除去。

    両端を含めるのは `first visible` / `last visible` 系の質問が区間端に集中するため。

    ★格子は **start を原点に張る**（`start + grid*k`）。本番クリップは
    `[start_time, end_time]` に切り出し済みで、その先頭がキーフレームになるため、
    絶対時刻の 5 秒倍数ではなく **クリップ相対**が正しい原点になる。
    PROCEDURE は全 10,000 問が `start_time == 0` なので両者は一致し、
    フレームキャッシュの共有も失われない（SEGMENT は start が問ごとに違う）。

    ★格子点数が n に満たない場合は**格子の全点を返す**（n より少なくなる）。
    「5秒より細かく見られない」という本番の制約をそのまま表現するのが目的なので、
    近傍点で水増しして n を保つことはしない。
    """
    span = end - start
    if n <= 1 or span <= 0:
        pts = [(start + end) / 2.0]
    else:
        step = span / (n - 1)
        pts = [start + i * step for i in range(n)]

    # ── アンカー時刻の割り込み ──────────────────────────────────
    # ★問題文が時刻を明示している問（SEGMENT 17.1% / PROCEDURE 27.8%）では、
    #   **その時刻のフレームを必ず含める**。一様サンプリングだと 64f でも刻みが
    #   数秒あり、`the Sponge visible in the frame at 00:19:49` のような問で
    #   肝心のフレームを外しうる。
    # ★**総枚数は変えない**（アンカーに最も近い一様点を置き換える）。枚数を増やすと
    #   「アンカーの効果」と「枚数の効果」が混ざって切り分けられなくなる。
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


_TS_RE = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")


def frames_for_stride(duration: float, stride: float,
                      n_min: int = 8, n_max: int = 96) -> int:
    r"""目標刻み `stride` 秒を満たす枚数を返す（`[n_min, n_max]` にクランプ）.

    ## なぜ枚数固定より刻み固定か（expE01 2026-08-05 の実測）
    SEGMENT の clip 長は **29s / 119s / 299s の3つの塊**に固まっており、
    **枚数を固定すると刻みが 10倍ばらつく**（64枚固定: 0-40s で 0.46s / 210-301s で 4.75s）。
    zero-shot 実測でも 0-40s は **16f 0.4869 → 64f 0.3979 と悪化**（過剰サンプルで
    干し草の山が増えるだけ）、40-130s は 0.2249 → 0.3252 と改善、と**向きが逆**だった。
    刻み適応 `min(64, dur/2+1)` は 16f 比 +0.062\*\*\* / 64f 比 +0.022\* で両者に勝っている。

    ★`n_max` は解像度の容量上限で決まる（実測: 448px=96枚 / 768px=32枚。
      `probe_capacity.py`。おおむね `frames × (size/448)^2 <= 96`）。
    ★`n_min` は短すぎるクリップで文脈が消えるのを防ぐ下限。
    """
    if duration <= 0 or stride <= 0:
        return n_min
    return int(min(max(int(duration // stride) + 1, n_min), n_max))


# 解像度ごとの最大フレーム数（`probe_capacity.py` の実測, RTX8000 48GB / 4bit）。
# おおむね `frames × (size/448)^2 <= 96` の予算線に乗る。
CAPACITY_FRAMES = {336: 160, 448: 96, 560: 64, 672: 48, 768: 32, 896: 32, 1024: 16}


def size_for_frames(n_frames: int, sizes: tuple[int, ...] = (448, 768)) -> int:
    r"""枚数から解像度を決める（**予算線 `frames × (size/448)^2 <= 96` の範囲で最大**）.

    ## 狙い
    SEGMENT の clip 長は 29s / 119s / 299s の3塊で、刻み 2s だと枚数が 15 / 60 / 96 になる。
    **短いクリップは枚数が少ない＝予算が余る**ので、そのぶんを解像度に回す。
    根拠: SEGMENT では 448px 学習のモデルを **768px で推論すると +0.0131 (p=0.049\*)**、
    内訳は **fo_class +0.038\*\*** のみ（expE01, 2026-08-05）。物体識別は解像度で効く。

    ★`sizes` は昇順の候補。**乗る中で最大**を返す（乗らなければ最小を返す）。
    """
    best = min(sizes)
    for sz in sorted(sizes):
        if n_frames <= CAPACITY_FRAMES.get(sz, 0):
            best = sz
    return best


# ── 学習時の入力構成 augmentation ────────────────────────────────────
# ★推論時にどの構成を使うかが未確定（予算線上の 96f@448 / 64f@560 / 32f@768 が拮抗）で、
#   しかも **学習は 16f@448 なのに推論は 64f@560 が最良**という乖離が起きている。
#   → **学習時に構成をランダムに振って、どの構成でも強いモデルにする**。
#   実測(SEGMENT, expE06g を各構成で評価):
#       16f@448 0.6496 / 96f@448 0.6958 / 64f@560 0.7113 / 32f@768 0.7197
#   いずれも `frames × (size/448)^2 ≒ 96` の予算線上（= 同じ vision token 数）。
#
# `(n_frames, size, stride)` の組。stride が None なら枚数固定、数値なら刻み固定。
# ★★2026-08-13 訂正: `time` の許容誤差は **一律 ±5s ではなくクリップ長依存**
#   （`base_dataset.py:180`: `min(5.0, 1 + duration * 4/360)`）。SEGMENT では
#       29s → ±1.32s / 119s → ±2.32s / 299s → ±4.32s
#   これで到達可能性を計算し直すと:
#       clip長  16f      64f              96f
#        29s   ❌1.93s  ✅0.46s          ✅0.31s
#       119s   ❌7.93s  ✅1.89s          ✅1.25s
#       299s   ❌19.9s  **❌4.75s**      ✅3.15s   ← 4,041問(20%)は 64f では届かない
#   実測でも 64f→96f で temporal が +0.014 伸びており整合する。
#   → **96f 系の比重を上げる**（3/6）。32f@768 は SCORE 最高だが temporal を有意に落とす
#     （64f@560 比 −0.054, p=0.0006）ので 1/6 に留める。
# ★★2026-08-13 【方針変更】**96枚構成は廃止**。
#   理由: 96枚は「299秒クリップ(20%)の time に届く唯一の構成」として入れたが、
#   vision encoder はフレーム数に完全比例するため **96枚が学習コストを支配**していた。
#   実測: AUG 平均50枚で 11.6 s/sample → 1 epoch 101時間（単体GPU）。
#   ★FlashAttention2 は attention の O(L^2) しか削らず、**vision tower の
#     フレーム数比例コストには一切効かない**。ここが今回の最大の学び。
#   → 上限を 64枚に下げ、平均を ~40枚に落とす（コスト約 1/1.5）。
#
# ★★2026-08-13 実測: **RTX 6000Ada 48GB では ~13k token の backward が通らない**。
#   ⚠️当初「96枚@448 が原因」と診断したが**誤り**。96f@448 を外しても max が 13,058→13,153 と
#     ほぼ変わらず、真犯人は **64f@560 (13,089 tok)** と **32f@768 (11,169 tok)** だった。
#   ＝ 構成の組み方ではなく **トークン数そのもの**が制約。実測の分岐点は約 10k token。
#   FlashAttention2 は効いている（attention の O(L^2) は消えた）が、13k token × 32層の
#   MLP 活性化と vision tower の中間表現が残り、gradient checkpointing でも収まらない。
#   → **96枚は 336px に落とす**。予算線 `frames × (size/448)^2` で 96→54 相当、
#     token は 12,705 → 約 9,345（−26%）。
#   ★96枚を残す理由: **299秒クリップ(20%)の `time` に届く唯一の構成**だから
#     （刻み 3.15s < 許容 4.32s）。336px でも刻みは変わらないので目的は保たれる。
#   ⚠️336px の知覚精度は未検証。だから比重は 2/6 に留め、他は検証済みの構成にする。
SEG_AUG_CONFIGS: list[tuple[int, int, float | None]] = [
    (64, 448, None),   # 8,737 tok。検証済みの中核
    (64, 448, None),   # ↑ 比重2倍
    (64, 448, 2.0),    # 刻み 2s（clip 長に応じて 8..64 枚。短clipの過剰サンプルを避ける）
    (48, 560, None),   # 10,017 tok。中間（解像度寄り）
    (32, 672, None),   # 9,249 tok。解像度に振る
    (16, 448, None),   # 2,785 tok。現行の学習条件（分布から外さない）
]


def sample_aug_config(rng, configs=None) -> tuple[int, int, float | None]:
    """学習1サンプルごとに入力構成を引く。`rng` は `random.Random`。"""
    return rng.choice(list(configs or SEG_AUG_CONFIGS))


# ── 匿名化区間 & 鮮明度によるフレーム選択 ──────────────────────────
# ★匿名化: 動画の一部が全画面ベタ塗り（heico BLUE(3,0,252) / lapchole BLUE(0,14,255) / BLACK）。
#   heico 23.8% / lapchole 11.9% の時間が該当し、**匿名ゼロの動画は200本中3本だけ**。
#   SEGMENT val では**サンプルの 3.49% が匿名フレーム、19.6% の問が1枚以上含む**（実測）。
#   区間表は `workspace/expE00_segproc_eda/anon_intervals.csv`（1fps 全走査で作成、秒単位）。
#
# ★方針: **黙って除外せず、隠されている範囲を本文で明示する**。
#   除外だけだと「そこに何も無かった」のか「見えなかった」のかをモデルが区別できない。
#   `first visible` 系の問いでは、観測できない窓があること自体が重要な情報になる。
_ANON: dict | None = None


def anon_intervals(dataset: str, video_id: str) -> list[tuple[float, float]]:
    """(start, end) 秒のリスト。無ければ空。"""
    global _ANON
    if _ANON is None:
        import csv as _csv
        f = ROOT / "workspace/expE00_segproc_eda/anon_intervals.csv"
        _ANON = {}
        if f.exists():
            with open(f) as fh:
                for r in _csv.DictReader(fh):
                    _ANON.setdefault((r["dataset"], r["video"]), []).append(
                        (float(r["start"]), float(r["end"])))
    return _ANON.get((dataset, Path(video_id).stem), [])


def is_anon(t: float, ivs: list[tuple[float, float]], pad: float = 0.5) -> bool:
    return any(a - pad <= t <= b + pad for a, b in ivs)


def anon_ranges_in(start: float, end: float,
                   ivs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """[start,end] と重なる匿名区間をクリップして返す（本文で提示する用）。"""
    out = []
    for a, b in ivs:
        lo, hi = max(a, start), min(b, end)
        if hi > lo:
            out.append((lo, hi))
    return sorted(out)


_SHARP: dict | None = None


def sharpness_table(size: int) -> dict:
    """(dataset, video, ms) → Laplacian 分散。`precompute_sharpness.py` の出力を読む。"""
    global _SHARP
    if _SHARP is None:
        import pandas as _pd
        f = HERE / f"sharpness_{size}.parquet"
        _SHARP = {}
        if f.exists():
            d = _pd.read_parquet(f)
            _SHARP = {(r.dataset, r.video, int(r.ms)): float(r.sharp)
                      for r in d.itertuples()}
            log.info(f"鮮明度テーブル {len(_SHARP):,} 件を読み込み（{f.name}）")
        else:
            log.warning(f"{f} が無い。鮮明度選択は無効になる")
    return _SHARP


def pick_sharpest(dataset: str, video_id: str, t: float, half: float,
                  ivs: list[tuple[float, float]],
                  src_size: int = 768) -> float | None:
    """`t` の ±half 秒から **匿名でなく最も鮮明な** 時刻を選ぶ。候補が無ければ None。

    ★候補は 1 秒グリッド（キャッシュ済みの粒度）。追加抽出は発生しない。
    ★鮮明度は `src_size` のキャッシュで測る（縮小しても順位は変わらないため、
      解像度ごとに測り直さない）。
    """
    tbl = sharpness_table(src_size)
    stem = Path(video_id).stem
    best, best_s = None, -1.0
    for off in range(-int(half), int(half) + 1):
        c = t + off
        if is_anon(c, ivs):
            continue
        sc = tbl.get((dataset, stem, int(round(c * 1000))))
        if sc is None or sc != sc:      # 欠損 / NaN
            continue
        if sc > best_s:
            best, best_s = c, sc
    return best


# ★expP00(2026-09-04): 匿名(単色)フレームを**全経路で**選択から外す opt-in。
#   実測: combo2(c9w) が選ぶ 18,560 枚のうち **1,412 枚(7.61%) が単色**で、
#   その 99.6% は `anon_intervals.csv` に正しく載っている（表は正確で、使っていないだけ）。
#   ⚠️既定 off。on にすると選択が変わるので、**重畳キャッシュの差分レンダが要る**。
C2_SKIP_ANON = os.environ.get("C2_SKIP_ANON", "0") == "1"


def drop_anon_times(dataset: str, video_id: str, ts: list[float],
                    lo: float, hi: float, grid: float) -> list[float]:
    """匿名時刻を**同じ格子上の最も近い可視時刻**へ置き換える（枚数を減らさない）.

    ★枚数を保つのが要点。減らすと「匿名を外した効果」と「枚数の効果」が混ざる
      （`sample_times` の anchor 置換と同じ設計）。
    ★置換先が見つからない（可視な格子点が尽きた）ときだけ落とす。
    """
    ivs = anon_intervals(dataset, video_id)
    if not ivs:
        return ts
    taken = {round(t, 3) for t in ts}
    k_max = int(math.floor((hi - lo) / grid + 1e-9))
    out: list[float] = []
    for t in ts:
        if not is_anon(float(t), ivs):
            out.append(t)
            continue
        k0 = int(round((float(t) - lo) / grid))
        best = None
        for d in range(1, k_max + 1):          # 近い方から外側へ探す
            for k in (k0 - d, k0 + d):
                if not (0 <= k <= k_max):
                    continue
                c = round(lo + k * grid, 3)
                if c in taken or is_anon(c, ivs):
                    continue
                best = c
                break
            if best is not None:
                break
        if best is not None:
            taken.discard(round(float(t), 3))
            taken.add(best)
            out.append(best)
        # 見つからなければ落とす（全区間が匿名のような極端なクリップ）
    return sorted(out)


def visible_candidates(start: float, end: float, grid: float,
                       ivs: list[tuple[float, float]]) -> list[float]:
    """[start,end] の格子点から **匿名秒を除いた候補列**を返す（昇順）。

    ★これが「匿名化フレームを除去した状態からフレーム選定をする」の土台。
      `sample_times` が全区間に一様に打ってから個別に置換するのとは**別物**で、
      こちらは**先に候補集合を絞ってから**そこに一様に打つ。
      匿名に潰される枠が無くなるので、その分がまるごと可視フレームに回る。
    """
    k_max = int(math.floor((end - start) / grid + 1e-9))
    return [round(start + k * grid, 3) for k in range(k_max + 1)
            if not is_anon(start + k * grid, ivs)]


def pick_from_candidates(cands: list[float], n: int, keep: list[float] | None = None,
                         grid: float = 1.0) -> list[float]:
    """候補列から **n 点を等間隔（インデックス上）で**取る。両端を必ず含む。

    ★インデックスで等間隔に取るので **重複は原理的に発生しない**（n <= len(cands) のとき）。
      実時間では匿名区間をまたぐ所だけ間隔が開くが、それは**映像が無いのだから正しい**。
    ★`keep`（問題文が名指しした時刻）は最も近い選択点を置き換えて必ず含める。
    """
    if not cands:
        return []
    if n >= len(cands):
        out = list(cands)
    else:
        idx = sorted({int(round(i * (len(cands) - 1) / (n - 1))) for i in range(n)}) \
            if n > 1 else [len(cands) // 2]
        out = [cands[i] for i in idx]
    for a in sorted({float(x) for x in (keep or [])}):
        if not out or is_anon_like(a, cands, grid):
            continue
        j = min(range(len(out)), key=lambda i: abs(out[i] - a))
        out[j] = a
    return sorted(set(out))


def is_anon_like(a: float, cands: list[float], grid: float) -> bool:
    """アンカー時刻が候補集合（＝可視秒）に無い＝そこは匿名なので入れられない。"""
    return not any(abs(c - a) <= grid / 2 + 1e-6 for c in cands)


def sharpen_times(dataset: str, video_id: str, ts: list[float],
                  ivs: list[tuple[float, float]],
                  keep: list[float] | None = None, grid: float = 1.0,
                  src_size: int = 768) -> list[float]:
    """各スロットを **自分のビン内で** 最も鮮明・非匿名な時刻に置き換える。

    ★**相異なる時刻の数を減らさない**のが要件。素朴に「±(間隔/2) の窓で最良を採って
      重複除去」にすると隣接スロットが同じ時刻に吸い寄せられて潰れる（実測 −17.6%）。
    → スロット間の中点で区切ったビンから選ぶ。ただし **ビンは半開区間 [lo, hi)** にする。
      ★★閉区間にすると**隣り合うビンが境界 `mids[i]` を共有**し、間隔が偶数のとき
        両方が同じ整数秒を選べてしまう。2026-08-15 の実害: リスト長は 30.66 のままなのに
        **相異なる時刻が 29.23（40-130s 帯では 32.00 → 29.48 = −7.9%）**、
        重複を含む問 58.5%、最大の穴が 4.0s → 7.6s に拡大して time が壊れた。
      ⚠️検証は **`len(ts)` ではなく `len(set(ts))`** で行うこと。長さは重複があっても変わらない。
    ★ビン内に格子点が1つしか無い（= 間隔 1s の密サンプル）場合は uniform と同一。
    ★ビン内が全部匿名なら元の時刻を残す。`anon_note=True` なら本文で明示される。
    ★`keep`（= `anchor=True` で入れた**問題文が名指しした時刻**）は絶対に動かさない。
      「00:19:49 のフレームに写っている Sponge」を聞かれているのに、隣の秒のほうが
      鮮明だからとずらしたら**別のフレームについて答えることになる**。
    """
    if len(ts) < 2:
        return ts
    tbl = sharpness_table(src_size)
    stem = Path(video_id).stem
    mids = [(ts[i] + ts[i + 1]) / 2 for i in range(len(ts) - 1)]
    pinned = [any(abs(t - a) <= grid / 2 + 1e-6 for a in keep) for t in ts] if keep \
        else [False] * len(ts)
    out: list[float] = []
    taken: set[float] = set()
    for i, t in enumerate(ts):
        if pinned[i]:
            out.append(t); taken.add(t)
            continue
        lo = mids[i - 1] if i > 0 else t - (mids[0] - ts[0])
        hi = mids[i] if i < len(mids) else t + (ts[-1] - mids[-1])
        best, best_s = None, -1.0
        # ★半開 [lo, hi): 最後のビンだけ上端を含める（そこは隣と共有しないので安全）
        c_hi = int(math.floor(hi)) if i == len(ts) - 1 else int(math.ceil(hi)) - 1
        for c in range(int(math.ceil(lo)), c_hi + 1):
            if is_anon(float(c), ivs) or float(c) in taken:
                continue
            sc = tbl.get((dataset, stem, c * 1000))
            if sc is None or sc != sc:      # 欠損 / NaN
                continue
            if sc > best_s:
                best, best_s = float(c), sc
        out.append(best if best is not None else t)
    return out


def question_anchor_times(question: str) -> list[float]:
    """問題文中の `hh:mm:ss` を秒に変換して返す（**絶対時刻**）.

    出現例: `the Sponge visible in the frame at 00:19:49` / `There is one Sponge in
    the frame at 02:43:20. When is it retrieved ...`。
    ★回答フォーマット指示の中の `hh:mm:ss` は**リテラルの書式指定**（数字ではない）なので
      正規表現に一致せず、誤検出しない。
    """
    return [float(int(h) * 3600 + int(m) * 60 + int(s))
            for h, m, s in _TS_RE.findall(str(question))]


# ── SEGMENT 質問文ルール（expN00, 2026-09-02 ユーザ判断で検証なし採用）─────────
# val 実測（fold v003, N=1958）で発火と論理保証を確認済み:
#   between: 62問（全て fo_class/OBJECT, 窓は 119s クリップ内の ~61s）→ 窓外は無関係
#   after-T: 96問（time/TEMPORAL, "When is it retrieved"）→ GT<T は 0/96（論理保証）
# どちらも **[窓] にフレーム予算を絞って密度を上げる**だけ。非発火は必ず呼び出し側で
# 現行経路（uniform+anchor）へ委譲する（v2-selector-must-delegate-to-v1）。
_SEG_BETWEEN_RE = re.compile(
    r"[Ww]hat types of foreign objects are seen between "
    r"(\d{1,2}:\d{2}:\d{2}) and (\d{1,2}:\d{2}:\d{2})")
_SEG_AFTER_RE = re.compile(
    r"in the frame at (\d{1,2}:\d{2}:\d{2})\.? +When is it retrieved")


def _hms2s(t: str) -> float:
    h, m, s = t.split(":")
    return float(int(h) * 3600 + int(m) * 60 + int(s))


def segrules_select_times(req, n: int, grid: float) -> list[float] | None:
    """SEGMENT の質問文ルール。発火しなければ None（呼び出し側が uniform+anchor へ委譲）."""
    q = str(req.question)
    lo, hi = float(req.start_time), float(req.end_time)
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
            # 答えは T 以後（GT<T は val 0/96）。[T, end] に密化、T は必ず含める
            return sample_times(t, hi, n, grid, [t])
    return None


def _find_video(dataset: str, video_id: str) -> Path:
    p = DATA_ROOT / dataset / "videos" / video_id
    if p.exists():
        return p
    cands = list((DATA_ROOT / dataset / "videos").glob(f"{Path(video_id).stem}*"))
    if not cands:
        raise FileNotFoundError(video_id)
    return cands[0]


def frame_path(dataset: str, video_id: str, t: float, size: int) -> Path:
    stem = Path(video_id).stem
    return CACHE_ROOT / f"{size}" / dataset / stem / f"{int(round(t*1000)):09d}.jpg"


# ★縮小元にできるキャッシュ解像度（大きい順に探す）。
#   貸しGPU へは 768 だけ転送し、448/560 はここから縮小して作る（転送量が 1/3 で済む）。
FALLBACK_SIZES = (768,)


def resolve_frame(dataset: str, video_id: str, t: float, size: int) -> tuple[Path, int | None]:
    """(実在するパス, 縮小先サイズ or None) を返す.

    要求サイズのキャッシュがあればそれをそのまま使う（`None`）。無ければ
    `FALLBACK_SIZES` の中で**実在する大きい解像度**を返し、縮小先として `size` を付ける。
    どれも無ければ要求サイズのパスを返す（呼び出し側が抽出を試みる）。
    """
    p = frame_path(dataset, video_id, t, size)
    if p.exists():
        return p, None
    for src in FALLBACK_SIZES:
        if src <= size:
            continue
        q = frame_path(dataset, video_id, t, src)
        if q.exists():
            return q, size
    return p, None


def _extract_one(args) -> bool:
    dataset, video_id, t, out, size = args
    if out.exists():
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    # ★tmp 名に pid+tid を必ず入れる。固定名にすると**同じ抽出を2プロセスで走らせたとき**
    #   互いの tmp を rename し合って FileNotFoundError で落ちる（2026-08-05 に実害）。
    #   キャッシュ埋めは並列に流したくなるので、衝突しない名前が前提条件。
    tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp.jpg")
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(_find_video(dataset, video_id)),
         "-frames:v", "1", "-vf", f"scale={size}:-2", "-q:v", "3", str(tmp)],
        capture_output=True)
    if r.returncode == 0 and tmp.exists():
        tmp.rename(out)
        return True
    tmp.unlink(missing_ok=True)
    return False


def extract_all(jobs: list[tuple], workers: int = 16) -> int:
    """(dataset, videoID, t, out, size) のリストを並列抽出。既存はスキップ。"""
    todo = [j for j in jobs if not j[3].exists()]
    if not todo:
        return 0
    log.info(f"extracting {len(todo)} frames with {workers} workers "
             f"({len(jobs)-len(todo)} already cached)")
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_extract_one, todo), 1):
            ok += bool(r)
            if i % 2000 == 0:
                log.info(f"  {i}/{len(todo)}")
    log.info(f"extracted {ok}/{len(todo)}")
    return ok


def load_qa_rows(version: str, track: str, fold: int, part: str,
                 limit: int | None = None) -> list[dict]:
    """qa_{version}/qa_split.csv から (track, fold, part) の行を order 昇順で。

    part='val' は fold==fold、'train' は fold!=fold。`order` 昇順で取るので
    limit を増やすと必ず入れ子になる。
    """
    path = ROOT / f"workspace/fold/qa_{version}/qa_split.csv"
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["track"] != track:
                continue
            in_fold = int(r["fold"]) == fold
            if (part == "val") != in_fold:
                continue
            r["order"] = int(r["order"])
            rows.append(r)
    rows.sort(key=lambda r: r["order"])
    return rows[:limit] if limit else rows


_QA_INDEX: dict = {}


# ★LapChole-FOCUS の revision（2026-09-04 の公式アナウンス）
#   「重複が見つかったので 4 本（labeled 2 / unlabeled 2）を足した v2 を**別リビジョン**で出す。
#     現行パイプラインを壊さないため main はそのまま。締切後に main へマージする」
#   → 既定は **main のまま**（v2 に切り替えると fold・対照基準・全キャッシュが作り直しになる）。
#   採用するときは `FOCUS_LAPCHOLE_REVISION=v2` を立てる。**その場合は最低限:
#     ① workspace/fold の qa_split を作り直す ② 対照 SCORE を測り直す
#     ③ 追加2本のフレーム/fo_timeline/phase_timeline/重畳を作る** の3点が必要。
#   ⚠️2026-09-04 08:36 時点では repo に v2 ブランチもタグも**存在しない**（main のみ）。
_LAPCHOLE_REVISION = os.environ.get("FOCUS_LAPCHOLE_REVISION") or None


def _qa_index(track: str) -> dict:
    if track not in _QA_INDEX:
        set_config(FocusConfig(root_dir=str(DATA_ROOT)))
        idx = {}
        for dsn in ("heico", "lapchole"):
            rev = _LAPCHOLE_REVISION if dsn == "lapchole" else None
            d = (FocusDataset(dsn, DatasetSplit.ALL, Track[track], revision=rev)
                 if rev else FocusDataset(dsn, DatasetSplit.ALL, Track[track]))
            for req, ref in zip(d.requests, d.references):
                idx[(dsn, req.qID)] = (req, ref)   # ★qID は dataset 間で衝突する
        _QA_INDEX[track] = idx
    return _QA_INDEX[track]


# --------------------------------------------------------------------------- #
# 工程(phase)索引による推薦（expI00_phase_clf）
# --------------------------------------------------------------------------- #
_PHASE_DIR = HERE.parents[0] / "expI00_phase_clf" / "results" / "phase_timeline"
_PHASE_CACHE: dict = {}


def phase_timeline(dataset: str, video_id: str):
    """秒ごとの工程 ID 配列。無ければ None（＝呼び出し側は uniform に戻す）。

    ★`FOCUS_NO_PHASE=1` で強制的に None を返す。提出コンテナに工程分類モデルを
      積まない構成を評価するため（2026-09-01）。
    """
    if os.environ.get("FOCUS_NO_PHASE") == "1":
        return None
    key = (dataset, Path(video_id).stem)
    if key not in _PHASE_CACHE:
        p = _PHASE_DIR / dataset / f"{Path(video_id).stem}.npy"
        _PHASE_CACHE[key] = np.load(p) if p.exists() else None
    return _PHASE_CACHE[key]


_PRIOR = None


def phase_select_times(dataset: str, req, answer_format: str | None,
                       n: int, grid: float, only_phase_policies: bool = False):
    """`expI00_phase_clf/phase_search.py` の推薦ルールで時刻を選ぶ。

    ★索引が無い動画・ルールが uniform を返す問では **既存の一様サンプリングに完全に倒す**
      （余計な交絡を作らないため。`--frame-select visible` で同じ設計にして
      「差が出た問＝介入が効いた問」に絞れたのと同じ考え方）。
    返り値が None なら呼び出し側が uniform を使う。
    """
    global _PRIOR
    tl = phase_timeline(dataset, req.videoID)
    if tl is None:
        return None
    if _PRIOR is None:
        sys.path.insert(0, str(HERE.parents[0] / "expI00_phase_clf"))
        from phase_search import PhasePrior
        _PRIOR = PhasePrior.load(HERE.parents[0] / "expI00_phase_clf"
                                 / "results" / "phase_prior.json")
    proc = "colorectal" if dataset == "heico" else "cholecystectomy"
    lo, hi = int(req.start_time), int(min(req.end_time, len(tl) - 1))
    if hi <= lo:
        return None
    ts, policy = _PRIOR.select_times(tl[lo:hi + 1], proc, req.question,
                                     answer_format=answer_format, budget=n, grid=grid,
                                     anchor_sec=None)
    if policy == "uniform":
        return None                       # 既存経路に完全に倒す
    # ★`phase_anchor` モード: 工程モデルを実際に使う方策だけ適用し、
    #   位置 prior だけの `position` は既存の一様経路へ倒す
    #   （expI01 で `position` 単独 −0.0108 と有害だったため）
    if only_phase_policies and policy == "position":
        return None
    return sorted({float(min(max(lo + x, lo), hi)) for x in ts})


# --------------------------------------------------------------------------- #
# FO 検出索引による推薦（expI03_fo_index）
# --------------------------------------------------------------------------- #
_FO_DIR = HERE.parents[0] / "expI03_fo_index" / "results" / "fo_timeline"
_FO_CACHE: dict = {}
# ★クラス列は**索引 npz が自己申告するもの**に合わせる（`fo_timeline` が初回読み込み時に上書き）。
#   検出器を差し替えるとクラス構成が変わる: expF26 は 8 クラス、**expF40 は Gallstone を落として 7 クラス**。
#   ここを固定にすると `Specimen` を `Gallstone` の列で読むことになり、**全ルールが黙って誤作動する**
#   （2026-09-01 に気付いた。索引と読み手のクラス列が一致していることを assert する）。
_FO_CLASSES = ["sponge", "clip", "Specimen_Bag", "Silicon_Loop",
               "External_Drain", "Needle", "Gallstone", "Specimen"]
_FO_CLASSES_LOCKED = False
# QA の表記 → 検出器のクラス名
_QA2DET = {"Sponge": "sponge", "Clip": "clip", "Specimen Bag": "Specimen_Bag",
           "Silicone Loop": "Silicon_Loop", "External Drain": "External_Drain",
           "Needle": "Needle", "Gallstone": "Gallstone", "Specimen": "Specimen"}
FO_STRIDE_S = int(os.environ.get("FO_STRIDE_S", "15"))        # 索引の走査刻み（5s 索引を間引く）。15s は 5s と同性能で本番に載る
FO_SCORE_THR = float(os.environ.get("FO_SCORE_THR", "0.30"))     # ★連続 score でなく**二値マスク**にするのが要点（生 score は ≈uniform）
FO_UNIFORM_SHARE = float(os.environ.get("FO_UNIFORM_SHARE", "0.25")) # 予算の 25% は一様に残す
# ★検出点の前後を候補に含める（膨張）。粗い走査で落ちた分を拾い直す。
#   実測（±5s 到達, N=633）: 刻み15s で 48.7%→**52.4%**(±30s膨張) /
#   刻み30s+±60s膨張 48.0% は「刻み15s 膨張なし 48.7%」に**走査枚数半分で並ぶ**。
import os as _os
FO_DILATE_S = int(os.environ.get("FO_DILATE_S", "30"))


def _dilate(w: np.ndarray, grid: float) -> np.ndarray:
    """検出マスクを前後 `FO_DILATE_S` 秒へ広げる（0 なら何もしない）。"""
    if FO_DILATE_S <= 0:
        return w
    k = int(round(FO_DILATE_S / grid)) * 2 + 1
    # ⚠️np.convolve(..., "same") は入力がカーネルより短いと **max(M,N) 長**を返す。
    #   そのまま使うと短いクリップで w と w_anc の shape が食い違って例外→コンテナが落ちる。
    return (np.convolve(w, np.ones(k), "same")[:len(w)] > 0).astype(np.float32)


def fo_timeline(dataset: str, video_id: str):
    """FO 検出索引を読む。**npz が自己申告するクラス列に `_FO_CLASSES` を合わせる。**"""
    global _FO_CLASSES, _FO_CLASSES_LOCKED
    key = (dataset, Path(video_id).stem)
    if key not in _FO_CACHE:
        p = _FO_DIR / dataset / f"{Path(video_id).stem}.npz"
        z = dict(np.load(p)) if p.exists() else None
        if z is not None and "classes" in z:
            cls = [str(c) for c in z["classes"]]
            if not _FO_CLASSES_LOCKED:
                if cls != _FO_CLASSES:
                    log.warning("索引のクラス列に合わせる: %d→%d クラス %s",
                                len(_FO_CLASSES), len(cls), cls)
                _FO_CLASSES, _FO_CLASSES_LOCKED = cls, True
            elif cls != _FO_CLASSES:
                raise RuntimeError(
                    f"索引ごとにクラス列が違う: {p} は {cls} だが既に {_FO_CLASSES} を使用中。"
                    "検出器を混ぜて索引を作ると列がズレて全ルールが誤作動する")
        elif z is not None and not _FO_CLASSES_LOCKED:
            _FO_CLASSES_LOCKED = True      # 旧形式（classes 無し）＝ 8 クラス固定として確定
        _FO_CACHE[key] = z
    return _FO_CACHE[key]


def fo_class_of(question: str):
    """質問文から FO クラスを一意に決める。決まらなければ None。

    ★**長いキーから順に照合し、一致した箇所を消してから次を見る**。
      素朴に部分一致を取ると `"Specimen Bag"` の質問が `"Specimen"` にも当たり
      「2クラス＝決められない」で None に落ちる（2026-08-30 に実害: PROCEDURE val の
      time 問 103 件が FO 索引を使えず一様サンプリングになっていた。うち 55 件が Specimen Bag）。
    """
    ql = str(question).lower()
    hit = []
    for k in sorted(_QA2DET, key=len, reverse=True):
        kl = k.lower()
        if kl in ql:
            hit.append(_QA2DET[k])
            ql = ql.replace(kl, " ")      # ★消してから次のキーを見る
    hit = [h for h in hit if h in _FO_CLASSES]   # ★索引に無いクラスは落とす
    return hit[0] if len(set(hit)) == 1 else None


# --- 複数クラスの取り扱い（expI06 / 2026-08-31）--------------------------- #
# `fo_class_of` は 2 クラス以上だと None を返して**索引ごと放棄**していた。
# PROCEDURE 全 10,000 問のうち **1,241 問（12.4%）がクラスを名指ししているのに捨てられて**いた
# （2クラス 1,215 / 3クラス 26。AND 577 / OR 664。number 341 / time 330 / open 248 / binary 245）。
_AND_RE = re.compile(r"at the same time|co-?occur|simultaneous|both", re.I)


def fo_classes_of(question: str) -> list[str]:
    """質問文が名指しする FO クラスを**すべて**返す（長いキーから消し込む）。"""
    ql = str(question).lower()
    out: list[str] = []
    for k in sorted(_QA2DET, key=len, reverse=True):
        kl = k.lower()
        if kl in ql:
            out.append(_QA2DET[k])
            ql = ql.replace(kl, " ")
    seen: set[str] = set()
    out = [c for c in out if not (c in seen or seen.add(c))]
    # ★索引に無いクラス（例: expF40 では Gallstone）は落とす。
    #   残らなければ「クラス不明」として全クラスの和で扱われる（呼び出し側の既定動作）。
    return [c for c in out if c in _FO_CLASSES]


def fo_mask_mode(question: str) -> str:
    """`at the same time` / `co-occur` は積(all)、それ以外は和(any)。"""
    return "all" if _AND_RE.search(str(question)) else "any"


def fo_class_mask(z, classes, mode: str, thr: float, k: int, lo: float, hi: float):
    """指定クラス群の検出マスクを合成して返す。classes が空なら全クラスの和。"""
    t = z["t"].astype(np.float32)
    sel = (t >= lo) & (t <= hi)
    t = t[sel]
    if len(t) == 0:
        return t, np.zeros(0, dtype=bool)
    cols = [_FO_CLASSES.index(c) for c in classes if c in _FO_CLASSES] or list(range(len(_FO_CLASSES)))
    sc = z["score"][sel][:, cols].astype(np.float32) >= thr
    m = sc.all(axis=1) if (mode == "all" and len(cols) > 1) else sc.any(axis=1)
    return t, _opening(m, k)


def fo_select_times(dataset: str, req, answer_format: str | None, n: int, grid: float):
    """FO 検出区間へ予算を寄せる。適用外なら None（＝既存の一様経路に倒す）。

    ★**`time` 形式にだけ適用する**。expI01 で「網羅が要る形式（fo_class/binary/number）で
      集中させると壊れる」ことを実 SCORE で確認済み。
    ★**閾値で二値化した検出マスク**を重みにする。生 score を重みにすると
      到達率が uniform と変わらない（expI03 実測 18.7% vs 17.1%）。
    """
    if answer_format != "time":
        return None
    z = fo_timeline(dataset, req.videoID)
    cls = fo_class_of(req.question)
    if z is None or cls is None:
        return None
    step = max(1, FO_STRIDE_S // 5)
    ci = _FO_CLASSES.index(cls)
    tt = z["t"][::step].astype(np.float64)
    sc = z["score"][::step, ci].astype(np.float32)
    lo, hi = float(req.start_time), float(min(req.end_time, tt[-1] if len(tt) else 0))
    if hi <= lo:
        return None
    secs = np.arange(lo, hi + 1e-6, grid)
    w = np.interp(secs, tt, (sc >= FO_SCORE_THR).astype(np.float32))
    w = _dilate(w, grid)
    if w.sum() <= 0:
        return None                       # 一度も検出されない → 一様に倒す
    n_uni = int(round(n * FO_UNIFORM_SHARE))
    pts = list(np.linspace(lo, hi, n_uni)) if n_uni else []
    n_tgt = n - n_uni
    if n_tgt > 0:
        cdf = np.cumsum(w) / w.sum()
        q = (np.arange(n_tgt) + 0.5) / n_tgt
        pts += list(secs[np.searchsorted(cdf, q).clip(0, len(secs) - 1)])
    got = np.unique(np.round((np.array(pts) - lo) / grid) * grid + lo)
    # ★予算を捨てない（expI01 で実効枚数が 64→61.3 に落ちた実害があった）
    if len(got) < n:
        rest = np.setdiff1d(secs, got)
        if len(rest):
            take = min(n - len(got), len(rest))
            got = np.unique(np.concatenate(
                [got, rest[np.round(np.linspace(0, len(rest) - 1, take)).astype(int)]]))
    return sorted(float(x) for x in got)


def combo_select_times(dataset: str, req, answer_format: str | None, n: int, grid: float):
    """expI03(FO 索引) + expI02-A(純アンカー) の合成. 対象問が排他なので足し合わせられる.

    ```
    1. time 形式          → FO 検出索引（アンカーがあれば **その工程区間と積**を取る）
    2. 非 time × アンカー有 → アンカーの工程区間（位置 prior は掛けない = expI02-A）
    3. それ以外            → None（既存の一様経路へ倒す）
    ```

    ★**青（匿名）フレームの明示除去は入れない**。FO 索引を使うだけで選ばれた枚の青率は
      9.2% → 4.6% と半減する（青いフレームには FO 検出が無いため自動的に避ける）。
      明示除去すると ±5s 到達が 48.6% → 47.4% と**下がる**（expI03 実測）。
    """
    z = fo_timeline(dataset, req.videoID)
    if z is None:
        return None
    lo, hi = float(req.start_time), float(min(req.end_time, z["t"][-1]))
    if hi <= lo:
        return None
    secs = np.arange(lo, hi + 1e-6, grid)

    # アンカー（質問文の hh:mm:ss）の工程区間
    w_anc = None
    anc = question_anchor_times(req.question)
    if anc:
        ph = phase_timeline(dataset, req.videoID)
        if ph is not None:
            a = ph[int(np.clip(anc[0], 0, len(ph) - 1))]
            w_anc = (ph[np.clip(secs.astype(int), 0, len(ph) - 1)] == a).astype(np.float32)
            if w_anc.sum() <= 0:
                w_anc = None

    if answer_format == "time":
        cls = fo_class_of(req.question)
        if cls is None:
            return None
        step = max(1, FO_STRIDE_S // 5)
        ci = _FO_CLASSES.index(cls)
        w = np.interp(secs, z["t"][::step].astype(np.float64),
                      (z["score"][::step, ci].astype(np.float32) >= FO_SCORE_THR).astype(np.float32))
        w = _dilate(w, grid)
        if w.sum() <= 0:
            return None
        if w_anc is not None and float((w * w_anc).sum()) > 0:
            w = w * w_anc                      # ★両信号が一致した区間だけ（±5s 到達 48.6→50.0%）
    else:
        if w_anc is None:
            return None                        # 非 time でアンカーが無ければ触らない
        w = w_anc

    n_uni = int(round(n * FO_UNIFORM_SHARE))
    pts = list(np.linspace(lo, hi, n_uni)) if n_uni else []
    n_tgt = n - n_uni
    if n_tgt > 0:
        cdf = np.cumsum(w) / w.sum()
        q = (np.arange(n_tgt) + 0.5) / n_tgt
        pts += list(secs[np.searchsorted(cdf, q).clip(0, len(secs) - 1)])
    got = np.unique(np.round((np.array(pts) - lo) / grid) * grid + lo)
    if len(got) < n:                            # ★予算を捨てない
        rest = np.setdiff1d(secs, got)
        if len(rest):
            take = min(n - len(got), len(rest))
            got = np.unique(np.concatenate(
                [got, rest[np.round(np.linspace(0, len(rest) - 1, take)).astype(int)]]))
    return sorted(float(x) for x in got)


# --------------------------------------------------------------------------- #
# expI06: combo2 — 質問テンプレート別のフレーム選択ルール（全軸を env で振れる）
#
# combo（expI04, 0.4871）に**未実装だった3軸**を足したもの。既存の combo は壊さない。
#   ① A2 区間限定  : `between <T1> and <T2>` は**区間内に予算を寄せる**（704問）
#   ② A3 中間補間  : 2時刻の照合問は**両端に厚く・間を薄く繋ぐ**（265問, binary の同一性判定）
#   ③ D 系に粗く寄せ: 非time かつアンカー無し（計数/列挙）にも FO 索引を**弱く**当てる
#      ⚠️expI01 の教訓「網羅が要る形式で集中させると壊れる」があるので **share を高く**（=弱い集中）。
#         壊れる境界を測るのが目的。既定 0.70（＝30%だけ寄せる）。
#
# 全パラメータは env で上書きできる（推論を回すだけで A/B できる）。
# --------------------------------------------------------------------------- #
C2_INTERVAL   = os.environ.get("C2_INTERVAL", "1") == "1"    # ① between 区間限定
C2_BRIDGE     = float(os.environ.get("C2_BRIDGE", "0.30"))   # ② 2時刻問で中間に回す割合
C2_NONTIME    = float(os.environ.get("C2_NONTIME", "1.0"))   # ③ 非time無アンカーの uniform share
                                                              #    1.0=触らない（=combo と同じ）
C2_ANCHOR_SD  = float(os.environ.get("C2_ANCHOR_SD", "30"))  # アンカーの広がり(秒)
_BETWEEN_RE = re.compile(r"\bbetween\b.*?\d{1,2}:\d{2}:\d{2}.*?\band\b.*?\d{1,2}:\d{2}:\d{2}",
                         re.I | re.S)


# --- ④ edge 配分（expI06）------------------------------------------------- #
# 「最初に見えたのはいつ」「最後に見えたのはいつ」型は、検出マスク**全体**に配るのではなく
# **平滑化したマスクの端 ±win** に予算を集中させる方が到達率が高い（CPU 解析: FIRST 58.9%→79.3%）。
# 生マスクの端は誤検出でガタガタなので、連続 k コマ成立を要求して平滑化してから端を取る。
C2_EDGE       = os.environ.get("C2_EDGE", "0")               # "0"=無効 / "first"=FIRST のみ(既定推奨)
                                                              # / "flt"=FIRST+LAST / "all"=全 time 問
                                                              # ★LAST は win 240/120/90/60 の**4設定すべてで負**
                                                              #   (4/12, 7/10, 6/10, 5/10) ＝ 端の推定が効かない
C2_EDGE_WIN   = float(os.environ.get("C2_EDGE_WIN", "240"))  # 端の周り ±win 秒に集中
C2_EDGE_THR   = float(os.environ.get("C2_EDGE_THR", "0.70")) # 端検出用のスコア閾値（本体より高め）
C2_EDGE_K     = int(os.environ.get("C2_EDGE_K", "5"))        # 連続 k コマ(=k*5s)成立で初めて「見えた」
C2_EDGE_SHARE = float(os.environ.get("C2_EDGE_SHARE", "0.25"))  # 予算のうち一様に残す割合
C2_EDGE_RESCAN = float(os.environ.get("C2_EDGE_RESCAN", "0"))   # >0: 端±win 内を2段目として更に絞る幅(秒)
# ★マスクが空のときに緩める梯子 "thr:k,thr:k,..."（既定 off）。thr0.7/k5 は 79 問で検出0だが、
#   thr0.3/k3 まで緩めると 57/79 が救える。k=1 まで落とすと平滑化が効かなくなるので入れない。
C2_EDGE_LADDER = os.environ.get("C2_EDGE_LADDER", "")
# ⑤ 「答えはアンカー時刻より後」型 → 探索範囲を [anchor, end] に切る（既定 off）。
#    `When is it retrieved from the surgical site?` は val 98 問すべてで GT ≥ anchor（100%）。
#    ①interval と同じ「範囲を切って密度を上げる」手。到達率 53.1% → 67.3%、
#    探索幅 3645s → 1458s、サンプル間隔 17.5s → 10.0s（CPU 実測）。
C2_AFTER = os.environ.get("C2_AFTER", "0") == "1"
# ⑧ 複数クラス条件づけ（既定 off）。クラスが書いてあるのに現行が捨てている 1,241 問を救う。
C2_MULTICLASS = os.environ.get("C2_MULTICLASS", "0") == "1"
_RETRIEVAL_RE = re.compile(r"when is it retrieved|retrieved from the surgical site", re.I)


class _ClipReq:
    """start_time だけ差し替えた Request の薄いラッパ（元の Request は不変のまま使う）。"""

    def __init__(self, req, lo):
        self._req, self.start_time = req, lo

    def __getattr__(self, k):
        return getattr(self._req, k)

_FIRST_RE = re.compile(r"first visible|first inserted|for the first time", re.I)
_LAST_RE = re.compile(r"last visible|last seen|final and last", re.I)


def _edge_kind(q: str) -> str:
    s = str(q).lower()
    if _LAST_RE.search(s):
        return "LAST"
    if _FIRST_RE.search(s) and "last" not in s:
        return "FIRST"
    return "OTHER"


def _opening(mask, k):
    """連続 k コマ成立している所だけ True にする（誤検出1発で端が前倒しになるのを防ぐ）。"""
    if k <= 1:
        return mask
    c = np.convolve(mask.astype(np.int32), np.ones(k, dtype=np.int32), "same")
    return c[:len(mask)] >= k        # ★"same" は短い入力で max(M,N) 長を返す（上の _dilate 参照）


def _runs_of(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _gauss(secs, centers, sd):
    w = np.zeros_like(secs, dtype=np.float32)
    for c in centers:
        w += np.exp(-0.5 * ((secs - c) / max(sd, 1e-6)) ** 2)
    return w


def _edge_alloc(z, ci, kind, lo, hi, n, grid, classes=None, mode="any"):
    """平滑化した検出マスクの端(FIRST=立ち上がり / LAST=立ち下がり)± win に予算を寄せる。

    `classes` を渡すと**複数クラスを合成したマスク**の端を取る（`mode="all"` で積）。
    渡さなければ従来どおり単一クラス `ci` のマスク。
    """
    sc = None
    if classes:
        tt, m = fo_class_mask(z, classes, mode, C2_EDGE_THR, C2_EDGE_K, lo, hi)
    else:
        tt = z["t"].astype(np.float32)
        sc = z["score"][:, ci].astype(np.float32)
        sel = (tt >= lo) & (tt <= hi)
        tt, sc = tt[sel], sc[sel]
        m = _opening(sc >= C2_EDGE_THR, C2_EDGE_K) if len(tt) else np.zeros(0, dtype=bool)
    if len(tt) == 0:
        return None
    rs = _runs_of(m)
    for step_ in (C2_EDGE_LADDER.split(",") if (not rs and C2_EDGE_LADDER) else []):
        thr_, k_ = step_.split(":")
        if classes:
            _, m2 = fo_class_mask(z, classes, mode, float(thr_), int(k_), lo, hi)
        else:
            m2 = _opening(sc >= float(thr_), int(k_))
        rs = _runs_of(m2)
        if rs:
            break
    if not rs:
        return None
    e = float(tt[rs[-1][1] - 1] if kind == "LAST" else tt[rs[0][0]])
    win = C2_EDGE_WIN
    # ★ユーザ設計 last_visible からの移植: **1〜2 本先のイベントも見る**。
    #   検出器の誤検出で 1 本目が偽陽性だった場合の保険。予算は等分する。
    if C2_EDGE_MULTI > 1 and len(rs) > 1:
        k = min(C2_EDGE_MULTI, len(rs))
        es = ([float(tt[rs[i][0]]) for i in range(k)] if kind != "LAST"
              else [float(tt[rs[-1 - i][1] - 1]) for i in range(k)])
        hi2 = min(hi, float(tt[-1]))
        n_uni = int(round(n * C2_EDGE_SHARE))
        pts = list(np.linspace(lo, hi2, n_uni)) if n_uni else []
        per = max(2, (n - n_uni) // len(es))
        for x in es:
            pts += list(np.linspace(max(lo, x - win), min(hi2, x + win), per))
        return _take(pts, lo, hi2, n, grid)
    if C2_EDGE_RESCAN > 0:
        # 2段目: 1段目の窓の中だけ**より低い閾値**で見直し、端を精密化してから窓を狭める
        sub = (tt >= e - win) & (tt <= e + win)
        if sub.any() and sc is not None:
            m2 = _opening(sc[sub] >= FO_SCORE_THR, max(1, C2_EDGE_K // 2))
            rs2 = _runs_of(m2)
            if rs2:
                t2 = tt[sub]
                e = float(t2[rs2[-1][1] - 1] if kind == "LAST" else t2[rs2[0][0]])
                win = C2_EDGE_RESCAN
    hi2 = min(hi, float(tt[-1]))
    n_uni = int(round(n * C2_EDGE_SHARE))
    pts = list(np.linspace(lo, hi2, n_uni)) if n_uni else []
    pts += list(np.linspace(max(lo, e - win), min(hi2, e + win), n - n_uni))
    got = np.unique(np.round((np.array(pts) - lo) / grid) * grid + lo)
    if len(got) < n:
        rest = np.setdiff1d(np.arange(lo, hi2 + grid, grid, dtype=np.float32), got)
        if len(rest):
            take = min(n - len(got), len(rest))
            got = np.unique(np.concatenate(
                [got, rest[np.round(np.linspace(0, len(rest) - 1, take)).astype(int)]]))
    return sorted(float(x) for x in got)


# =========================================================================== #
# expI06 ⑫ 質問テンプレート別ルール（ユーザ設計 2026-08-31）
#   `C2_RULES` にカンマ区切りで有効化するルール名を並べる（既定 off）。
#   「密」の最大密度は **5 秒**（= PROCEDURE の time 許容枠）。それ以上細かくしても無意味。
# =========================================================================== #
C2_RULES = set(x for x in os.environ.get("C2_RULES", "").split(",") if x)
# --- ユーザ設計から既存ルールへ移植したアイデア（すべて既定 off で A/B できる）------- #
C2_EDGE_MULTI  = int(os.environ.get("C2_EDGE_MULTI", "0"))   # ③edge: 2,3 本目の立ち上がりも見る
C2_AFTER_ANON  = os.environ.get("C2_AFTER_ANON", "0") == "1" # ⑤after: 匿名区間±10s を必ず入れ、近いほど密
C2_DSYS_SEG    = os.environ.get("C2_DSYS_SEG", "0") == "1"   # ④dsys: FO 増減で分割し各区間に同枚数
C2_SHARPEN     = float(os.environ.get("C2_SHARPEN", "0"))    # 非 time 形式を ±秒 で鮮明フレームへ寄せる
C2_MC_DILATE   = float(os.environ.get("C2_MC_DILATE", "0"))  # ⑧multiclass: ±秒 ならしてから AND/OR
# ⑬ ④dsys の配分を「区間ごとに均等」に変える（既定 off）。
#   現行の CDF サンプリングは**長く写っている区間ほど厚く**なるので、
#   「何個あるか / どのクラスが出るか」という**列挙**の問いには最悪の配分になっている。
#   実測（train+val 4,289問）: CDF 配分が触れている検出区間は**中央 40.0%**（6割は 1 枚も見ていない）。
#   検出区間数の中央値は 8（97% の問が 48 以下）なので、**1 区間 1 枚なら全部カバーできる**。
C2_DSYS_PERRUN = float(os.environ.get("C2_DSYS_PERRUN", "0"))  # >0: 各区間へ回す割合
def _take(times, lo, hi, n, grid):
    """時刻列を [lo,hi] にクリップ→5s 格子へスナップ→重複除去→n 枚に整える。
    足りなければ一様に埋め戻す（予算を捨てない）。多ければ等間隔に間引く。"""
    a = np.asarray(list(times), dtype=np.float64)
    if a.size == 0:
        return None
    a = np.clip(a, lo, hi)
    got = np.unique(np.round((a - lo) / grid) * grid + lo)
    if len(got) > n:                                  # 多すぎたら等間隔に間引く
        got = got[np.round(np.linspace(0, len(got) - 1, n)).astype(int)]
    elif len(got) < n:
        full = np.arange(lo, hi + grid, grid)
        rest = np.setdiff1d(full, got)
        if len(rest):
            take = min(n - len(got), len(rest))
            got = np.unique(np.concatenate(
                [got, rest[np.round(np.linspace(0, len(rest) - 1, take)).astype(int)]]))
    return sorted(float(x) for x in got)


def _dense(a, b, grid=5.0):
    """[a,b] を最大密度（5 秒刻み）で埋める。"""
    if b < a:
        a, b = b, a
    return list(np.arange(a, b + grid, grid))


def _pad_runs(t, runs, frac=0.075):
    """検出区間を前後 frac 分だけ広げて (start, end) の秒で返す（既定 ±7.5%）。"""
    out = []
    for i, j in runs:
        a, b = float(t[i]), float(t[max(i, j - 1)])
        m = (b - a) * frac
        out.append((a - m, b + m))
    return out


# --- 質問テンプレ別ルール本体（ユーザ設計 2026-08-31）--------------------- #
# 各関数は「時刻リスト」か None（＝適用外、既存経路へ倒す）を返す。
_R: list = []                      # [(name, 正規表現, 関数)]  — 先頭から順に判定


def _reg(name, pat):
    def deco(fn):
        _R.append((name, re.compile(pat, re.I), fn))
        return fn
    return deco


def _cls_runs(z, classes, lo, hi, thr=0.30, k=3, mode="any"):
    t, m = fo_class_mask(z, classes, mode, thr, k, lo, hi)
    return t, m, _runs_of(m)


@_reg("leave_fov", r"how many separate times.*leave the field of view")
def _r_leave_fov(ds, z, req, q, cls, lo, hi, n, grid):
    """出入り回数: 対象 FO の検出区間を ±7.5% 広げて等間隔に。境目を見せるのが目的。"""
    t, m, rs = _cls_runs(z, cls, lo, hi)
    if not rs:
        return None
    segs = _pad_runs(t, rs)
    span = sum(b - a for a, b in segs) or 1.0
    pts = []
    for a, b in segs:
        pts += list(np.linspace(a, b, max(2, int(round(n * (b - a) / span)))))
    return _take(pts, lo, hi, n, grid)


@_reg("lastseen_pair", r"with which other foreign object classes")
def _r_lastseen_pair(ds, z, req, q, cls, lo, hi, n, grid):
    """最後に見えたクラスの前後 2 分。そのタイミングで同時に写る相手を探させる。"""
    best = -1.0
    for c in _FO_CLASSES:
        t, m, rs = _cls_runs(z, [c], lo, hi)
        if rs:
            best = max(best, float(t[rs[-1][1] - 1]))
    if best < 0:
        return None
    return _take(_dense(best - 120, best + 120, grid), lo, hi, n, grid)


@_reg("longest_cooc", r"longest co-?occurrence")
def _r_longest_cooc(ds, z, req, q, cls, lo, hi, n, grid):
    """どの 2 クラスかを問われるので**全クラスの検出和**を疎に等間隔で全部見せる。"""
    t, m, rs = _cls_runs(z, [], lo, hi)
    if not rs:
        return None
    cand = t[m]
    return _take(cand[np.round(np.linspace(0, len(cand) - 1, n)).astype(int)], lo, hi, n, grid)


@_reg("retrieval_exists", r"does a retrieval of this object exist")
def _r_retrieval_exists(ds, z, req, q, cls, lo, hi, n, grid):
    """T から末尾へ**近いほど密・遠いほど疎**。さらに匿名区間の前後 10s を必ず入れる。
    ★匿名区間はカメラ/物体が体外へ出るシーンで、retrieval がそこで起きやすい（ユーザ知見）。"""
    anc = question_anchor_times(q)
    if not anc:
        return None
    T = float(np.clip(anc[0], lo, hi))
    rest = hi - T
    if rest <= 0:
        return None
    must = []
    for a, b in _anon_for(req):
        if b >= T and a <= hi:
            must += _dense(max(lo, a - 10), min(hi, b + 10), grid)
    must = sorted(set(must))
    n_log = max(8, n - len(must))
    d = np.geomspace(15.0, max(30.0, rest), n_log)      # 15s,30s,60s,... 近いほど密
    pts = list(T + np.cumsum(d) / np.cumsum(d)[-1] * rest)
    return _take(must + pts + [T], lo, hi, n, grid)


@_reg("nth_inserted", r"\d+(?:st|nd|rd|th) visible .*inserted in the abdomen")
def _r_nth_inserted(ds, z, req, q, cls, lo, hi, n, grid):
    """個数 `count` を平滑化し、**N 回目の増加**を密に。前後の増加(N-1,N+1)も密に見る。
    insseg の見落とし/過検出に備え、各段の区間も疎に押さえる。"""
    m = re.search(r"(\d+)(?:st|nd|rd|th)", q)
    N = int(m.group(1)) if m else 1
    # ★N=1 は既存の ③edge/FIRST（マスクの立ち上がり）が 0.5924 で機能している。
    #   そこを count ベースに置き換えたら 0.2930 へ崩れた（4勝51敗, p<0.0001, 2026-08-31）。
    #   このルールが直したいのは「常に 1 本目を取ってしまう」N>=2 だけ（全体の 13.7%）。
    if N < 2:
        return None
    cols = [_FO_CLASSES.index(c) for c in cls] if cls else list(range(len(_FO_CLASSES)))
    t = z["t"].astype(np.float32)
    sel = (t >= lo) & (t <= hi)
    t = t[sel]
    if len(t) < 3:
        return None
    cnt = z["count"][sel][:, cols].sum(axis=1).astype(np.float32)
    kk = max(1, int(round(15.0 / grid)))
    cnt = np.convolve(cnt, np.ones(kk) / kk, "same")[:len(t)]
    ups = [float(t[i]) for i in range(1, len(t)) if np.floor(cnt[i]) > np.floor(cnt[i - 1])]
    if not ups:
        return None
    pts = []
    for i in (N - 2, N - 1, N):                          # N-1, N, N+1 回目の増加
        if 0 <= i < len(ups):
            pts += _dense(ups[i] - 60, ups[i] + 60, grid)
    if not pts:
        pts += _dense(ups[-1] - 60, ups[-1] + 60, grid)
    pts += list(np.linspace(lo, hi, max(8, n // 4)))     # 各段を疎に押さえる
    return _take(pts, lo, hi, n, grid)


@_reg("retrieval_when", r"when is it retrieved from the surgical site")
def _r_retrieval_when(ds, z, req, q, cls, lo, hi, n, grid):
    """T から対象 FO が消えるまで +7.5%。T 側と消失側を密に、間は等間隔。
    誤検出なら T に近いほど該当シーンがある。"""
    anc = question_anchor_times(q)
    if not anc:
        return None
    T = float(np.clip(anc[0], lo, hi))
    t, m, rs = _cls_runs(z, cls, T, hi)
    end = float(t[rs[0][1] - 1]) if rs else hi
    end = min(hi, end + max(0.0, end - T) * 0.075)
    pts = _dense(T, min(hi, T + 90), grid) + _dense(max(lo, end - 90), end, grid)
    pts += list(np.linspace(T, max(T, end), max(8, n // 3)))
    return _take(pts, lo, hi, n, grid)


@_reg("last_visible", r"at what time was a .*last visible in the video")
def _r_last_visible(ds, z, req, q, cls, lo, hi, n, grid):
    """消失タイミングの前後 30s を 5 秒刻み。**1〜2 個前の消失**も入れて誤検出に備える。"""
    t, m, rs = _cls_runs(z, cls, lo, hi)
    if not rs:
        return None
    pts = []
    for _, j in rs[-3:]:
        e = float(t[j - 1])
        pts += _dense(e - 30, e + 30, grid)
    pts += list(np.linspace(lo, hi, max(6, n // 6)))
    return _take(pts, lo, hi, n, grid)


@_reg("after_first_inserted", r"after the .*was inserted in the abdomen")
def _r_after_first(ds, z, req, q, cls, lo, hi, n, grid):
    """対象 FO の初出 −60s から最終検出 +60s までを等間隔（疎でよいが全体を見る）。"""
    t, m, rs = _cls_runs(z, cls, lo, hi)
    if not rs:
        return None
    a = max(lo, float(t[rs[0][0]]) - 60)
    b = min(hi, float(t[rs[-1][1] - 1]) + 60)
    return _take(np.linspace(a, b, n), lo, hi, n, grid)


@_reg("and_same_time", r"at the same time in this video")
def _r_and_same_time(ds, z, req, q, cls, lo, hi, n, grid):
    """**時刻方向にならしてから** AND を取る（A の 30 秒前に B が写れば同時扱い）。
    AND 区間は密、OR 区間は疎。"""
    if len(cls) < 2:
        return None
    t = z["t"].astype(np.float32)
    sel = (t >= lo) & (t <= hi)
    t = t[sel]
    if len(t) == 0:
        return None
    kd = int(round(30.0 / grid)) * 2 + 1
    ms = []
    for c in cls:
        mm = z["score"][sel][:, _FO_CLASSES.index(c)].astype(np.float32) >= FO_SCORE_THR
        ms.append(np.convolve(mm.astype(np.int32), np.ones(kd, np.int32), "same")[:len(t)] > 0)
    both = np.logical_and.reduce(ms)
    either = np.logical_or.reduce(ms)
    pts = []
    for i, j in _runs_of(both):
        pts += _dense(float(t[i]), float(t[max(i, j - 1)]), grid)
    if either.any():
        cand = t[either]
        pts += list(cand[np.round(np.linspace(0, len(cand) - 1, max(8, n // 4))).astype(int)])
    return _take(pts, lo, hi, n, grid) if pts else None


@_reg("max_count", r"maximum number of .*at once in a single frame")
def _r_max_count(ds, z, req, q, cls, lo, hi, n, grid):
    """検出インスタンス数が最大の区間を密に。2 番目・3 番目の区間も押さえる。"""
    cols = [_FO_CLASSES.index(c) for c in cls] if cls else list(range(len(_FO_CLASSES)))
    t = z["t"].astype(np.float32)
    sel = (t >= lo) & (t <= hi)
    t = t[sel]
    if len(t) == 0:
        return None
    cnt = z["count"][sel][:, cols].sum(axis=1)
    if cnt.max() <= 0:
        return None
    pts, used = [], np.zeros(len(t), bool)
    for _ in range(3):
        c2 = np.where(used, -1, cnt)
        i = int(np.argmax(c2))
        if c2[i] <= 0:
            break
        a, b = max(0, i - 12), min(len(t), i + 13)       # ±60s
        pts += _dense(float(t[a]), float(t[b - 1]), grid)
        used[a:b] = True
    pts += list(np.linspace(lo, hi, max(8, n // 6)))
    return _take(pts, lo, hi, n, grid)


def _count_changes(z, cls, lo, hi, smooth_s=15.0, grid=5.0):
    """FO 個数 `count` を平滑化し、**増減が起きた時刻**を返す（(時刻, 差分) のリスト）。
    ★動画を「同じ状態が続く区間」に割るための境界。取り逃しを防ぐのが目的。"""
    cols = [_FO_CLASSES.index(c) for c in cls] if cls else list(range(len(_FO_CLASSES)))
    t = z["t"].astype(np.float32)
    sel = (t >= lo) & (t <= hi)
    t = t[sel]
    if len(t) < 3:
        return t, []
    cnt = z["count"][sel][:, cols].sum(axis=1).astype(np.float32)
    k = max(1, int(round(smooth_s / grid)))
    cnt = np.floor(np.convolve(cnt, np.ones(k) / k, "same")[:len(t)])
    ch = [(float(t[i]), float(cnt[i] - cnt[i - 1]))
          for i in range(1, len(t)) if cnt[i] != cnt[i - 1]]
    return t, ch


def _equal_per_segment(bounds, n, grid):
    """区間ごとに**同じ枚数**を等間隔で割り当てる（長い区間で取り逃さないため）。"""
    bounds = [b for b in bounds if b[1] > b[0]]
    if not bounds:
        return []
    per = max(2, n // len(bounds))
    pts = []
    for a, b in bounds:
        pts += list(np.linspace(a, b, per))
    return pts


def _sharpen(dataset, req, times, half=2.0):
    """各時刻を ±half 秒の範囲で**匿名でなく最も鮮明な**時刻へ寄せる（無ければ元のまま）。
    ★`fo_class` など時刻を答えない形式では許容誤差の制約が無いので自由にずらせる。"""
    ivs = anon_intervals(dataset, req.videoID)
    out = []
    for t in times:
        c = pick_sharpest(dataset, req.videoID, float(t), half, ivs)
        out.append(c if c is not None else float(t))
    return out


@_reg("interval_seg", r"\bbetween\b.*\d{1,2}:\d{2}:\d{2}.*\band\b.*\d{1,2}:\d{2}:\d{2}")
def _r_interval_seg(ds, z, req, q, cls, lo, hi, n, grid):
    """`between T1 and T2`: 等間隔が基本。ただし**区間が長いときは FO 増減で分割**し、
    各小区間へ同じ枚数を配る（取り逃しを減らす）。さらに鮮明なフレームへ寄せる。"""
    anc = question_anchor_times(q)
    if len(anc) < 2:
        return None
    a, b = max(min(anc[:2]), lo), min(max(anc[:2]), hi)
    if b <= a:
        return None
    _, ch = _count_changes(z, [], a, b)
    cuts = [a] + [c for c, _ in ch if a < c < b] + [b]
    pts = (_equal_per_segment(list(zip(cuts[:-1], cuts[1:])), n, grid)
           if (b - a) > 300 and len(cuts) > 2 else list(np.linspace(a, b, n)))
    return _take(_sharpen(ds, req, pts), lo, hi, n, grid)


@_reg("all_classes", r"which foreign object classes appear in this video")
def _r_all_classes(ds, z, req, q, cls, lo, hi, n, grid):
    """全クラス列挙: FO が写っている区間 ±60s を等間隔。長ければ個数変化で分割。"""
    t, m, rs = _cls_runs(z, [], lo, hi)
    if not rs:
        return None
    a = max(lo, float(t[rs[0][0]]) - 60)
    b = min(hi, float(t[rs[-1][1] - 1]) + 60)
    _, ch = _count_changes(z, [], a, b)
    cuts = [a] + [c for c, _ in ch if a < c < b] + [b]
    pts = (_equal_per_segment(list(zip(cuts[:-1], cuts[1:])), n, grid)
           if (b - a) > 600 and len(cuts) > 2 else list(np.linspace(a, b, n)))
    return _take(pts, lo, hi, n, grid)


@_reg("applied_times", r"at which time points were .*applied")
def _r_applied_times(ds, z, req, q, cls, lo, hi, n, grid):
    """複数時刻を返す型: **対象 FO の個数が増える瞬間**を密に見せる。"""
    t, ch = _count_changes(z, cls, lo, hi)
    ups = [c for c, d in ch if d > 0]
    if not ups:
        return None
    per = max(3, (n - max(6, n // 6)) // len(ups))
    pts = []
    for u in ups:
        half = (per - 1) / 2 * grid
        pts += _dense(u - half, u + half, grid)
    pts += list(np.linspace(lo, hi, max(6, n // 6)))
    return _take(pts, lo, hi, n, grid)


@_reg("nth_unique_class", r"\d+(?:st|nd|rd|th) unique class")
def _r_nth_unique(ds, z, req, q, cls, lo, hi, n, grid):
    """N 番目に登場する FO クラス: **「見えたクラス数」が N になる瞬間**を密に見せる。
    ★各クラスの初出を並べて N 番目を取り、その前後を密に。前後 1 個の遷移も押さえる。"""
    firsts = []
    for c in _FO_CLASSES:
        t, m, rs = _cls_runs(z, [c], lo, hi)
        if rs:
            firsts.append(float(t[rs[0][0]]))
    if not firsts:
        return None
    firsts.sort()
    m = re.search(r"(\d+)(?:st|nd|rd|th)", q)
    N = int(m.group(1)) if m else 1
    pts = []
    for i in (N - 2, N - 1, N):                       # N-1, N, N+1 番目の登場
        if 0 <= i < len(firsts):
            pts += _dense(firsts[i] - 60, firsts[i] + 60, grid)
    if not pts:
        pts += _dense(firsts[-1] - 60, firsts[-1] + 60, grid)
    pts += list(np.linspace(lo, hi, max(8, n // 4)))   # 取りこぼし用に全体を疎に
    return _take(pts, lo, hi, n, grid)


@_reg("count_inserted", r"how many .*(are|is) inserted in the abdomen")
def _r_count_inserted(ds, z, req, q, cls, lo, hi, n, grid):
    """挿入個数: 対象 FO の検出区間 ±60s を等間隔。**匿名区間は除く**（そこには写っていない）。"""
    t, m, rs = _cls_runs(z, cls, lo, hi)
    if not rs:
        return None
    a = max(lo, float(t[rs[0][0]]) - 60)
    b = min(hi, float(t[rs[-1][1] - 1]) + 60)
    ivs = anon_intervals(ds, req.videoID)
    pts = [x for x in np.linspace(a, b, int(n * 1.6)) if not is_anon(float(x), ivs)]
    return _take(pts or list(np.linspace(a, b, n)), lo, hi, n, grid)


@_reg("present_at_t", r"how many .*present in the surgical site")
def _r_present_at_t(ds, z, req, q, cls, lo, hi, n, grid):
    """時刻 T における在数（見えていないものも含む）: **FO の初出から T へ向かって
    だんだん密になる**ように配る。T に近いほど在数の手掛かりが濃い。"""
    anc = question_anchor_times(q)
    if not anc:
        return None
    T = float(np.clip(anc[0], lo, hi))
    t, m, rs = _cls_runs(z, cls, lo, T)
    a = float(t[rs[0][0]]) if rs else lo
    span = T - a
    if span <= 0:
        return None
    u = np.linspace(0.0, 1.0, n) ** 2                  # T 側ほど密（2 乗で寄せる）
    return _take(T - span * u, lo, hi, n, grid)


@_reg("nth_unique_late", r"\d+(?:st|nd|rd|th) unique class")
def _r_nth_unique_late(ds, z, req, q, cls, lo, hi, n, grid):
    """N 番目のクラス（**N>=3 限定**）: FO の写っている区間を **前半 3 割 / 後半 7 割**で配る。
    ★N=1 は 0.880 取れているので触らない（N>=3 が 0.286）。番号が大きいほど答えは後ろにある。"""
    m = re.search(r"(\d+)(?:st|nd|rd|th)", q)
    if not m or int(m.group(1)) < 3:
        return None
    t, mm, rs = _cls_runs(z, [], lo, hi)
    if not rs:
        return None
    a, b = float(t[rs[0][0]]), float(t[rs[-1][1] - 1])
    mid = (a + b) / 2
    n_first = int(round(n * 0.3))
    pts = list(np.linspace(a, mid, n_first)) + list(np.linspace(mid, b, n - n_first))
    return _take(pts, lo, hi, n, grid)


@_reg("max_count_many", r"maximum number of .*at once in a single frame")
def _r_max_count_many(ds, z, req, q, cls, lo, hi, n, grid):
    """最大同時個数（**検出器の instance 数が 5 以上の動画に限定**）:
    count の 1 位 / 2 位 / 3 位の区間**だけ**から渡す。
    ⚠️GT>=5 で分岐するとリークになるので、**推論時に得られる検出器の値**で分岐する。"""
    cols = [_FO_CLASSES.index(c) for c in cls] if cls else list(range(len(_FO_CLASSES)))
    t = z["t"].astype(np.float32)
    sel = (t >= lo) & (t <= hi)
    t = t[sel]
    if len(t) == 0:
        return None
    cnt = z["count"][sel][:, cols].sum(axis=1)
    if cnt.max() < 5:                      # ★ここが分岐条件（GT ではなく検出器の値）
        return None
    pts, used = [], np.zeros(len(t), bool)
    for _ in range(3):
        c2 = np.where(used, -1, cnt)
        i = int(np.argmax(c2))
        if c2[i] <= 0:
            break
        a, b = max(0, i - 12), min(len(t), i + 13)     # ±60s
        pts += _dense(float(t[a]), float(t[b - 1]), grid)
        used[a:b] = True
    return _take(pts, lo, hi, n, grid) if pts else None


@_reg("interval_long", r"\bbetween\b.*\d{1,2}:\d{2}:\d{2}.*\band\b.*\d{1,2}:\d{2}:\d{2}")
def _r_interval_long(ds, z, req, q, cls, lo, hi, n, grid):
    """`between T1 and T2`（**区間 10 分超のみ**）: FO の**個数ごとに区間を割り**、
    各区間へ同じ枚数を配って 64 枚を埋める。★短い区間は等間隔が最適なので触らない。"""
    anc = question_anchor_times(q)
    if len(anc) < 2:
        return None
    a, b = max(min(anc[:2]), lo), min(max(anc[:2]), hi)
    if b - a <= 600:                       # 10 分以下は既存の等間隔に任せる
        return None
    _, ch = _count_changes(z, [], a, b)
    cuts = [a] + [c for c, _ in ch if a < c < b] + [b]
    if len(cuts) <= 2:
        return None
    return _take(_equal_per_segment(list(zip(cuts[:-1], cuts[1:])), n, grid), lo, hi, n, grid)


def _anon_for(req):
    for ds in ("heico", "lapchole"):
        a = anon_intervals(ds, req.videoID)
        if a:
            return a
    return []


def apply_template_rule(dataset, req, answer_format, n, grid):
    """`C2_RULES` で有効化されたテンプレルールを順に試す。発火しなければ None。"""
    if not C2_RULES:
        return None
    q = str(req.question)
    z = fo_timeline(dataset, req.videoID)
    if z is None:
        return None
    lo, hi = float(req.start_time), float(req.end_time)
    tt = z["t"].astype(np.float32)
    hi = min(hi, float(tt[-1]) if len(tt) else lo)
    if hi <= lo:
        return None
    cls = fo_classes_of(q)
    for name, pat, fn in _R:
        if name in C2_RULES and pat.search(q):
            try:
                got = fn(dataset, z, req, q, cls, lo, hi, n, grid)
            except Exception:
                log.exception("テンプレルール %s が失敗（適用外に倒す）", name)
                return None
            if got:
                return got
            return None
    return None


def _spread(secs, w, lo, hi, n, share, grid):
    """重み `w` に従って予算 n を配る（一様に share を残す）。取りこぼした枠は埋め戻す。"""
    n_uni = int(round(n * share))
    pts = list(np.linspace(lo, hi, n_uni)) if n_uni else []
    n_tgt = n - n_uni
    if n_tgt > 0:
        cdf = np.cumsum(w) / w.sum()
        qq = (np.arange(n_tgt) + 0.5) / n_tgt
        pts += list(secs[np.searchsorted(cdf, qq).clip(0, len(secs) - 1)])
    got = np.unique(np.round((np.array(pts) - lo) / grid) * grid + lo)
    if len(got) < n:
        rest = np.setdiff1d(secs, got)
        if len(rest):
            take = min(n - len(got), len(rest))
            got = np.unique(np.concatenate(
                [got, rest[np.round(np.linspace(0, len(rest) - 1, take)).astype(int)]]))
    return sorted(float(x) for x in got)


def combo2_select_times(dataset: str, req, answer_format: str | None, n: int, grid: float):
    """combo + 区間限定 + 中間補間 + 非time への弱い寄せ。"""
    z = fo_timeline(dataset, req.videoID)
    lo, hi = float(req.start_time), float(req.end_time)
    if hi <= lo:
        return None
    anc = question_anchor_times(req.question)
    q = str(req.question)

    # ⑫ 質問テンプレ別ルール（ユーザ設計）— 発火すればここで決まる。既定は off。
    got = apply_template_rule(dataset, req, answer_format, n, grid)
    if got:
        return got

    # ⑤ retrieval 型は答えが必ずアンカー以降 → 手前を捨てて密度を倍にする
    _after_extra: list[float] = []
    if (C2_AFTER and answer_format == "time" and len(anc) == 1
            and _RETRIEVAL_RE.search(q) and lo < anc[0] < hi):
        lo = float(anc[0])
        req = _ClipReq(req, lo)          # ★委譲先(combo)にも切った範囲を渡す
        # ★ユーザ設計 retrieval_exists からの移植: 匿名区間はカメラ/物体が体外へ出るシーンで
        #   retrieval がそこで起きやすい。前後 10s を必ず入れ、さらに近いほど密に配る。
        if C2_AFTER_ANON:
            for a, b in anon_intervals(dataset, req.videoID):
                if b >= lo and a <= hi:
                    _after_extra += _dense(max(lo, a - 10), min(hi, b + 10), grid)
            rest = hi - lo
            if rest > 0:
                d = np.geomspace(15.0, max(30.0, rest), max(8, n // 2))
                _after_extra += list(lo + np.cumsum(d) / np.cumsum(d)[-1] * rest)

    secs = np.arange(lo, hi + grid, grid, dtype=np.float32)

    # ① `between T1 and T2` → 区間の外を捨てる（検出器も phase も要らない）
    if C2_INTERVAL and len(anc) >= 2 and _BETWEEN_RE.search(q):
        a, b = min(anc[:2]), max(anc[:2])
        a, b = max(a, lo), min(b, hi)
        if b > a:
            pts = np.linspace(a, b, n)
            got = np.unique(np.round((pts - lo) / grid) * grid + lo)
            return sorted(float(x) for x in got)

    # ② 2時刻の照合（between ではない）→ 両端に厚く、C2_BRIDGE の割合で中間を繋ぐ
    if len(anc) >= 2:
        a, b = min(anc[:2]), max(anc[:2])
        n_mid = int(round(n * C2_BRIDGE))
        n_end = n - n_mid
        pts = list(np.linspace(a - C2_ANCHOR_SD, a + C2_ANCHOR_SD, n_end // 2))
        pts += list(np.linspace(b - C2_ANCHOR_SD, b + C2_ANCHOR_SD, n_end - n_end // 2))
        if n_mid > 0 and b > a:
            pts += list(np.linspace(a, b, n_mid + 2)[1:-1])
        pts = [min(max(x, lo), hi) for x in pts]
        got = np.unique(np.round((np.array(pts) - lo) / grid) * grid + lo)
        if len(got) >= max(4, n // 4):
            return sorted(float(x) for x in got)

    if z is None:
        return None
    tt = z["t"].astype(np.float32)
    hi = min(hi, float(tt[-1]) if len(tt) else lo)
    if hi <= lo:
        return None
    secs = np.arange(lo, hi + grid, grid, dtype=np.float32)

    step = max(1, FO_STRIDE_S // 5)
    share = FO_UNIFORM_SHARE

    # ⑧ 複数クラス条件づけ: 質問が名指しするクラス**全部**でマスクを作る
    #    （`at the same time` / `co-occur` は積、それ以外は和）。off なら従来の単一クラス。
    mcls = fo_classes_of(q) if C2_MULTICLASS else []
    mmode = fo_mask_mode(q)

    # ④ first/last visible → 平滑化した検出マスクの端 ±win に集中
    if answer_format == "time":
        cls = fo_class_of(q)
        kind = _edge_kind(q)
        ok = {"0": (), "first": ("FIRST",), "flt": ("FIRST", "LAST"),
              "all": ("FIRST", "LAST", "OTHER")}.get(C2_EDGE, ())
        if kind in ok and (cls is not None or len(mcls) >= 2):
            got = (_edge_alloc(z, None, kind, lo, hi, n, grid, classes=mcls, mode=mmode)
                   if len(mcls) >= 2 else
                   _edge_alloc(z, _FO_CLASSES.index(cls), kind, lo, hi, n, grid))
            if got is not None:
                return got

    # ⑨ time で①③⑤が発火しない問でも、クラスが 2 個以上書いてあるなら合成マスクへ寄せる
    #    （従来は fo_class_of=None → combo も None → **一様**だった）
    #    ★ここは必ず **OR(any)** で撒く。AND マスクは検出漏れで断片的なので、
    #      そこへ予算を配ると**一様より悪くなる**（train n=160 の LAST 2クラス同時で
    #      一様 26.2% に対し AND全体 19.4% / OR全体 46.2%）。AND は「端を取る」時だけ使う。
    if answer_format == "time" and len(mcls) >= 2:
        # ★ユーザ設計 and_same_time からの移植: **時刻方向にならしてから**判定する
        #   （A の 30 秒前に B が写っていれば「同時」として扱いたい）。
        kd = int(round(C2_MC_DILATE / grid)) * 2 + 1 if C2_MC_DILATE > 0 else 1
        if kd > 1:
            t2 = z["t"].astype(np.float32)
            s2 = (t2 >= lo) & (t2 <= hi)
            ms = [np.convolve(
                (z["score"][s2][:, _FO_CLASSES.index(c)].astype(np.float32) >= FO_SCORE_THR
                 ).astype(np.int32), np.ones(kd, np.int32), "same")[:int(s2.sum())] > 0
                for c in mcls]
            m = np.logical_or.reduce(ms)
        else:
            _, m = fo_class_mask(z, mcls, "any", FO_SCORE_THR, 1, lo, hi)
        if m.any():
            t2 = z["t"].astype(np.float32)
            t2 = t2[(t2 >= lo) & (t2 <= hi)]
            w = np.interp(secs, t2, _dilate(m.astype(np.float32), grid))
            if w.sum() > 0:
                share = FO_UNIFORM_SHARE
                return _spread(secs, w, lo, hi, n, share, grid)

    # ③ D 系（計数・列挙 = 非 time かつアンカー無し）— FO が写っている区間へ**弱く**寄せる
    if answer_format != "time" and not anc and C2_NONTIME < 1.0:
        # ⑬ 列挙・計数向け: **検出区間を1つも取りこぼさない**配分。
        #    予算の C2_DSYS_PERRUN を区間へ均等に配り、残りは一様に撒く。
        if C2_DSYS_PERRUN > 0:
            t2, m2 = fo_class_mask(z, mcls, "any", FO_SCORE_THR, 3, lo, hi)
            rs2 = _runs_of(m2)
            if rs2:
                n_run = int(round(n * C2_DSYS_PERRUN))
                per = max(1, n_run // len(rs2))
                pts = []
                for i, j in rs2:                       # 各区間から等間隔に per 枚
                    pts += list(np.linspace(float(t2[i]), float(t2[max(i, j - 1)]), per))
                pts += list(np.linspace(lo, hi, n - min(len(pts), n_run)))
                got = _take(pts, lo, hi, n, grid)
                if got:
                    return got
        # ★ユーザ設計 interval_seg / all_classes からの移植:
        #   長い動画では **FO 個数が変わる時刻で区間を割り、各区間へ同じ枚数**を配る。
        #   「弱く寄せる」だけだと長い区間で取り逃す。
        if C2_DSYS_SEG:
            tt2, ch = _count_changes(z, mcls, lo, hi)
            cuts = [lo] + [c for c, _ in ch if lo < c < hi] + [hi]
            if len(cuts) > 2:
                pts = _equal_per_segment(list(zip(cuts[:-1], cuts[1:])), n, grid)
                if C2_SHARPEN > 0:
                    pts = _sharpen(dataset, req, pts, C2_SHARPEN)
                got = _take(pts, lo, hi, n, grid)
                if got:
                    return got
        cls = fo_class_of(q)
        if len(mcls) >= 2:
            _, mm = fo_class_mask(z, mcls, mmode, FO_SCORE_THR, 1, lo, hi)
            t2 = z["t"].astype(np.float32)
            t2 = t2[(t2 >= lo) & (t2 <= hi)]
            w = np.interp(secs, t2, _dilate(mm.astype(np.float32), grid))
        else:
            cols = [_FO_CLASSES.index(cls)] if cls else list(range(len(_FO_CLASSES)))
            m = (z["score"][::step][:, cols].astype(np.float32) >= FO_SCORE_THR).any(axis=1)
            w = np.interp(secs, tt[::step], _dilate(m.astype(np.float32), grid * step))
        if w.sum() <= 0:
            return None
        share = C2_NONTIME
    else:
        # ★どのルールも発火しない問は **combo（expI04, 0.4871）へそのまま委譲**する。
        #   ここを combo2 が独自に処理すると、combo の**工程区間アンカー**（expI02-A, 非 time で
        #   +0.0125）が**ガウシアンに置き換わって黙って消える**。2026-08-30 に実害を確認:
        #   全ノブ off の combo2 が combo と 433 問で食い違っていた。
        got = combo_select_times(dataset, req, answer_format, n, grid)
        if got and _after_extra:            # ⑤after の匿名/対数点を優先で混ぜる
            return _take(list(_after_extra) + list(got), lo, hi, n, grid)
        return got

    return _spread(secs, w, lo, hi, n, share, grid)


def build_samples(track: str, fold: int, part: str, n_frames: int = 16,
                  size: int = 448, limit: int | None = None, version: str = "v004",
                  grid: float | None = None, extract: bool = True,
                  workers: int = 16, anchor: bool = False,
                  stride: float | None = None, n_min: int = 8,
                  n_max: int | None = None,
                  adaptive_size: tuple[int, ...] | None = None,
                  aug_configs: list | None = None,
                  aug_seed: int = 42,
                  frame_select: str = "uniform",
                  anon_note: bool = False) -> list[VideoVQASample]:
    """`grid=None` ならトラック既定（PROCEDURE=5s キーフレーム格子 / 他=1s）。

    `anchor=True` で**問題文中の `hh:mm:ss` のフレームを必ず含める**
    （最も近い一様サンプル点を置き換える。総枚数は変えない）。
    該当するのは SEGMENT 17.1% / PROCEDURE 27.8% / FRAME 3.3% の問のみで、
    残りは `anchor=False` と**完全に同一の入力**になる＝ matched 比較で効果が直接見える。
    """
    if grid is None:
        grid = grid_for(track)
    # `stride` 指定時は **問ごとに枚数を変える**（`n_frames` は使わず n_max の既定になる）
    if stride is not None and n_max is None:
        n_max = n_frames
    rows = load_qa_rows(version, track, fold, part, limit)
    idx = _qa_index(track)
    # ★aug は **サンプルごとに決定的**（uid で seed）。エポック間で同じ構成になるが、
    #   「どの問がどの構成か」は固定されるので **学習の再現性が保てる**。
    #   エポックごとに変えたい場合は Trainer 側で再構築する必要がある（未実装）。
    _rng = __import__("random").Random(aug_seed)

    # 1) 必要フレームを全問分まとめてから並列抽出（1問ずつ ffmpeg を呼ぶと遅すぎる）
    plan = []
    jobs: dict[Path, tuple] = {}
    for r in rows:
        got = idx.get((r["dataset"], r["qID"]))
        if got is None:
            continue
        req, ref = got
        ivs = anon_intervals(r["dataset"], req.videoID) if (
            frame_select in ("sharp", "visible", "visible_sharp") or anon_note) else []
        anc = question_anchor_times(req.question) if anchor else None
        _nf, _sz, _st = n_frames, size, stride
        if aug_configs is not None:
            _nf, _sz, _st = sample_aug_config(
                __import__("random").Random(f"{r['dataset']}:{r['qID']}:{aug_seed}".__hash__()),
                aug_configs)
        nf = (frames_for_stride(req.end_time - req.start_time, _st, n_min, _nf)
              if _st is not None else _nf)
        if frame_select in ("visible", "visible_sharp"):
            # ★**匿名秒を候補から落としてから**、残った可視秒の上で一様に取る。
            #   `sample_times` → 個別置換（= "sharp"）とは別物で、
            #   インデックス上で等間隔に取るので**重複も寄り合いも原理的に起きない**。
            cands = visible_candidates(req.start_time, req.end_time, grid, ivs)
            # ★★匿名が1秒も無いクリップでは **uniform と完全一致させる**。
            #   インデックス丸めの差で 1622問中6問がズレていた（2026-08-15 実測）。
            #   これは施策と無関係な**不要な交絡**なので、経路ごと uniform に倒す。
            #   こうすると「差が出た問 = 匿名があった問」だけになり、効果が clean に測れる。
            n_grid = int(math.floor((req.end_time - req.start_time) / grid + 1e-9)) + 1
            if len(cands) == n_grid:
                ts = sample_times(req.start_time, req.end_time, nf, grid, anc)
            else:
                ts = pick_from_candidates(cands, nf, keep=anc, grid=grid) or \
                    sample_times(req.start_time, req.end_time, nf, grid, anc)
            if frame_select == "visible_sharp":
                ts = sharpen_times(r["dataset"], req.videoID, ts, ivs, keep=anc, grid=grid)
        elif frame_select == "segrules":
            # ★expN00: SEGMENT 質問文ルール（between / after-T の窓密化）。
            #   非発火問は下の sample_times と**ビット同一**（anc も同じ経路）
            ts = segrules_select_times(req, nf, grid) \
                or sample_times(req.start_time, req.end_time, nf, grid, anc)
        elif frame_select == "combo2":
            ts = combo2_select_times(r["dataset"], req, r.get("answer_format"), nf, grid) \
                or sample_times(req.start_time, req.end_time, nf, grid, anc)
        elif frame_select == "combo":
            ts = combo_select_times(r["dataset"], req, r.get("answer_format"), nf, grid) \
                or sample_times(req.start_time, req.end_time, nf, grid, anc)
        elif frame_select == "fo":
            ts = fo_select_times(r["dataset"], req, r.get("answer_format"), nf, grid) \
                or sample_times(req.start_time, req.end_time, nf, grid, anc)
        elif frame_select in ("phase", "phase_anchor"):
            ts = phase_select_times(r["dataset"], req, r.get("answer_format"), nf, grid,
                                    only_phase_policies=(frame_select == "phase_anchor")) \
                or sample_times(req.start_time, req.end_time, nf, grid, anc)
        else:
            ts = sample_times(req.start_time, req.end_time, nf, grid, anc)
            # ★各スロットを **匿名でなく最も鮮明な**時刻に置き換える。
            #   候補が無い（全部匿名）スロットは元の時刻のまま残し、
            #   匿名なら下の `anon_ranges` で本文に明示されるので情報は失われない。
            if frame_select == "sharp":
                ts = sharpen_times(r["dataset"], req.videoID, ts, ivs, keep=anc, grid=grid)
        # ★expP00: 匿名(単色)フレームを全経路で外す（opt-in, C2_SKIP_ANON=1）。
        #   combo2 は count_inserted 以外の経路で匿名を考慮していなかった。
        if C2_SKIP_ANON:
            ts = drop_anon_times(r["dataset"], req.videoID, ts,
                                 req.start_time, req.end_time, grid)
        # ★**問ごとに解像度を変える**（枚数が少ない＝予算が余る問だけ高解像度にする）
        sz = size_for_frames(len(ts), adaptive_size) if adaptive_size else _sz
        resolved = [resolve_frame(r["dataset"], req.videoID, t, sz) for t in ts]
        paths = [p for p, _ in resolved]
        tgt = next((t2 for _, t2 in resolved if t2 is not None), None)
        for t, (p, _) in zip(ts, resolved):
            if not p.exists():
                jobs.setdefault(p, (r["dataset"], req.videoID, t, p, sz))
        plan.append((r, req, ref, ts, paths, tgt))
    if extract:
        extract_all(list(jobs.values()), workers=workers)

    # 2) サンプル化（抽出に失敗したフレームは落として続行）
    out: list[VideoVQASample] = []
    n_missing = 0
    for r, req, ref, ts, paths, tgt in plan:
        keep = [(t, p) for t, p in zip(ts, paths) if p.exists()]
        if not keep:
            n_missing += 1
            continue
        sysp, fmt = build_system_prompt(req.question)
        out.append(VideoVQASample(
            uid=r["uid"], qID=req.qID, dataset=r["dataset"], videoID=req.videoID, track=track,
            start_time=req.start_time, end_time=req.end_time,
            frame_times=[t for t, _ in keep], frame_paths=[p for _, p in keep],
            target_size=tgt,
            anon_ranges=(anon_ranges_in(req.start_time, req.end_time, ivs)
                         if anon_note else []),
            system_prompt=sysp, question=req.question, answer=str(ref.answer), fmt=fmt,
            procedure_type=getattr(req, "procedure_type", "") or "",
            primary=r["primary"], group=_group(r["primary"])))
    # ★格子で頭打ちになると実効フレーム数が n_frames を下回る。**必ずログに出す**
    #   （黙って 16f→9f になっていると、後から結果を比べたときに原因が分からない）
    eff = [len(s.frame_paths) for s in out]
    eff_mean = sum(eff) / len(eff) if eff else 0.0
    n_anch = sum(1 for _, req, _, _, _, _ in plan if question_anchor_times(req.question)) if anchor else 0
    log.info(f"built {len(out)} {track} samples (fold={fold}/{part}, {n_frames}f@{size}px, "
             f"grid={grid}s, anchor={anchor}"
             + (f", stride={stride}s(n {n_min}..{n_max})" if stride is not None else "")
             + (f"（時刻あり {n_anch}問 = {n_anch/max(len(plan),1):.1%}）" if anchor else "")
             + f", 実効フレーム数 mean={eff_mean:.1f} min={min(eff, default=0)} "
             f"max={max(eff, default=0)}, distinct frames {len(jobs)}, dropped {n_missing})")
    return out


def _group(primary: str) -> str:
    from focus import Capability
    try:
        return Capability[primary].group.name
    except KeyError:
        return primary


def build_messages(s: VideoVQASample, with_procedure: bool = False) -> list[dict]:
    """フレームごとに `[HH:MM:SS]` テキスト → 画像 の順で並べた chat messages。

    ★1フレーム（FRAME トラック）のときは「動画から抽出した複数フレーム」という前置きが
      嘘になるので文言を変える。joint 学習では FRAME と SEGMENT が同じバッチ列に混ざるため、
      **入力の説明文がサンプルの実態と一致していること**が重要。
    """
    if len(s.frame_paths) == 1:
        head = f"Frame at [{hhmmss(s.frame_times[0])}]:"
    else:
        head = (f"Video frames sampled from {hhmmss(s.start_time)} to "
                f"{hhmmss(s.end_time)} ({len(s.frame_paths)} frames):")
    # ★★`procedure_type` は公式が推論時にもくれる情報（コンテナの request.json に実在）。
    #   ★**system ではなく user ターンの先頭に置く**。LoRA は system の指示をほぼ無視する
    #   （2026-08-14 expG00: system に置いた検出指示が届かず、user に移して初めて効いた）。
    if with_procedure and s.procedure_type:
        head = f"Procedure: {s.procedure_type}\n" + head
    # ★匿名化区間を**本文で明示**する。黙って除外すると「そこに何も無かった」のか
    #   「見えなかった」のかをモデルが区別できず、`first visible` 系で誤答の理由になる。
    #   実測: SEGMENT val の 19.6% の問が匿名フレームを1枚以上含む。
    if getattr(s, "anon_ranges", None):
        rngs = ", ".join(f"{hhmmss(a)}-{hhmmss(b)}" for a, b in s.anon_ranges)
        head += ("\nNOTE: the source video is masked (blanked out for anonymisation) during "
                 f"{rngs}. No image is available there — you cannot tell what happens "
                 "in those periods.")
    content: list[dict] = [{"type": "text", "text": head}]

    def _img(p):
        if s.target_size:
            # ★キャッシュより小さい解像度が要求されたら**読み込み時に縮小**する。
            #   ffmpeg の `scale=W:-2` と同じく幅を合わせ、高さは偶数に丸める。
            from PIL import Image
            im = Image.open(p).convert("RGB")
            w = s.target_size
            h = max(2, int(round(im.height * w / im.width / 2)) * 2)
            return {"type": "image", "image": im.resize((w, h), Image.BICUBIC)}
        return {"type": "image", "image": f"file://{p}"}

    for t, p in zip(s.frame_times, s.frame_paths):
        if len(s.frame_paths) > 1:
            content.append({"type": "text", "text": f"[{hhmmss(t)}]"})
        content.append(_img(p))
    # ★重畳は原画像の**後ろ**に足し、直後に説明文を置く（submit/v008 の
    #   `build_user_content` と同一レイアウト: [head, 原画像, 重畳, 説明, 質問]）。
    if getattr(s, "overlay_path", None):
        content.append(_img(s.overlay_path))
        if s.overlay_note:
            content.append({"type": "text", "text": s.overlay_note})
    elif getattr(s, "overlay_note", ""):
        # ★expN00: in-place 重畳（フレーム自体に描き込み済み）。画像は増やさず
        #   説明文だけを質問の直前に置く。overlay_path 経路（expM00 の2枚渡し）とは排他
        content.append({"type": "text", "text": s.overlay_note})
    content.append({"type": "text", "text": s.question})
    return [{"role": "system", "content": s.system_prompt},
            {"role": "user", "content": content}]


def build_multitrack_samples(specs: list[dict], fold: int, part: str,
                             version: str = "v004", grid: float | None = None,
                             extract: bool = True, workers: int = 16,
                             shuffle_seed: int = 42) -> list[VideoVQASample]:
    """複数トラックを混ぜた学習セットを作る（joint 学習用）.

    specs 例:
        [{"track": "FRAME",   "n_frames": 1,  "size": 448, "limit": 6000},
         {"track": "SEGMENT", "n_frames": 16, "size": 448, "limit": 3000}]

    ★spec ごとに `"grid"` を書けばトラック別に上書きできる。省略時はトラック既定
      （PROCEDURE のみ 5s キーフレーム格子 = `KEYFRAME_GRID_S`）。
      引数 `grid` を明示すると**全トラックを一律に上書き**する。
    ★トラックごとに n_frames が違ってよい（FRAME は1枚、SEGMENT は16枚）。
      collator は bs=1 前提なので、系列長がサンプルごとに変わっても問題ない。
    ★**混ぜたあと必ずシャッフルする**。トラック順に並んだままだと、
      前半 FRAME・後半 SEGMENT という学習曲線になり、
      cosine スケジュールの後半（低 lr）が SEGMENT だけに当たってしまう。
    """
    import random as _rnd
    out: list[VideoVQASample] = []
    for sp in specs:
        g = grid if grid is not None else sp.get("grid")
        # ★expP00: **spec ごとに** frame_select / anchor を渡せるようにする。
        #   これが無いと joint 学習は常に uniform / anchor なしになり、
        #   推論側（run_infer.py --frame-select combo2 --anchor）と入力分布がズレたままになる。
        got = build_samples(sp["track"], fold, part,
                            n_frames=int(sp.get("n_frames", 16)),
                            size=int(sp.get("size", 448)),
                            limit=sp.get("limit"), version=version, grid=g,
                            extract=extract, workers=workers,
                            aug_configs=sp.get("aug_configs"),
                            frame_select=str(sp.get("frame_select", "uniform")),
                            anchor=bool(sp.get("anchor", False)))
        log.info(f"  [{sp['track']}] {len(got)} 件 "
                 f"({sp.get('n_frames',16)}f@{sp.get('size',448)}px, "
                 f"grid={g if g is not None else grid_for(sp['track'])}s, "
                 f"frame_select={sp.get('frame_select','uniform')}, "
                 f"anchor={bool(sp.get('anchor', False))})")
        out.extend(got)
    _rnd.Random(shuffle_seed).shuffle(out)
    from collections import Counter
    log.info(f"joint 学習セット {len(out)} 件  内訳 {dict(Counter(s.track for s in out))}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for n in ("httpx", "httpcore", "datasets", "huggingface_hub", "filelock", "fsspec"):
        logging.getLogger(n).setLevel(logging.WARNING)
    import argparse
    ap = argparse.ArgumentParser()
    # ★FRAME も扱える。FRAME は start_time == end_time なので sample_times() が
    #   1点だけ返し、自然に「1フレーム入力」になる（joint 学習で混ぜるために必要）。
    ap.add_argument("--track", default="SEGMENT", choices=["FRAME", "SEGMENT", "PROCEDURE"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", default="val")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    # ★qa split のバージョン。**fold バージョンとは別物**なので注意:
    #   qa_v004 は fold v003、qa_v005 は fold v001 の上に作られている。
    #   過去の FRAME 実験（expD*）は `splits.cv_folds()` の既定 = fold v001 基準なので、
    #   その重みを評価するときは **qa_v005** を使わないとリークする。
    ap.add_argument("--version", default="v004")
    # ★サンプリング格子（秒）。既定はトラック依存で PROCEDURE のみ 5s（本番クリップの
    #   キーフレーム間隔）。0 を渡すとスナップ無しの生の一様サンプルになる（対照用）。
    ap.add_argument("--grid", type=float, default=None)
    a = ap.parse_args()
    ss = build_samples(a.track, a.fold, a.part, n_frames=a.n_frames, size=a.size,
                       limit=a.limit, version=a.version, grid=a.grid,
                       extract=not a.no_extract, workers=a.workers)
    for s in ss[:5]:
        log.info(f"{s.fmt:15s} {s.group:20s} [{hhmmss(s.start_time)}..{hhmmss(s.end_time)}] "
                 f"{len(s.frame_paths)}f  Q={s.question[:70]!r} A={s.answer!r}")
