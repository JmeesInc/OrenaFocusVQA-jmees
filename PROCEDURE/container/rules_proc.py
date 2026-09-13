"""PROCEDURE のフレーム選択ルール（提出コンテナ用の自己完結版）。

★`workspace/expE01_segproc_baseline/dataset_seg.py` から**機械的に抽出**したもの。
  提出物は workspace に依存してはいけないので写してある。ロジックは同一。

★**刻み非依存化**（コンテナ固有の変更）
  val の索引は 5 秒刻みだが、コンテナは時間予算の都合で 15 秒刻みで作る。
  平滑化は元コードで「連続 k **サンプル**」だったので、そのままだと
  25 秒 → 75 秒に化けて意味が変わる。**秒で表して刻みから k を計算する**形にした。
  索引が 5 秒刻みなら `_k(25)=5` で元と完全一致する（テストで確認）。

★索引のクラス列は npz が自己申告する（expF40 は Gallstone を落として 7 クラス）。
"""
from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# --- 索引の刻み（コンテナが作った索引に合わせて set_stride() で更新する）------- #
INDEX_STRIDE_S = 5.0


def set_stride(s: float) -> None:
    """索引の実際の刻みを教える。平滑化幅（秒）を k サンプルへ直すのに使う。"""
    global INDEX_STRIDE_S
    INDEX_STRIDE_S = float(s)


def _k(seconds: float) -> int:
    """秒で指定した平滑化幅を、いまの刻みでの連続サンプル数に直す（最低 1）。"""
    return max(1, int(round(float(seconds) / max(INDEX_STRIDE_S, 1e-6))))


# --- 定数（workspace 版と同じ値）------------------------------------------- #
_FO_CLASSES = ["sponge", "clip", "Specimen_Bag", "Silicon_Loop",
               "External_Drain", "Needle", "Gallstone", "Specimen"]
_FO_CLASSES_LOCKED = False
_QA2DET = {"Sponge": "sponge", "Clip": "clip", "Specimen Bag": "Specimen_Bag",
           "Silicone Loop": "Silicon_Loop", "External Drain": "External_Drain",
           "Needle": "Needle", "Gallstone": "Gallstone", "Specimen": "Specimen"}
FO_SCORE_THR = float(os.environ.get("FO_SCORE_THR", "0.30"))
FO_UNIFORM_SHARE = float(os.environ.get("FO_UNIFORM_SHARE", "0.25"))
FO_DILATE_S = int(os.environ.get("FO_DILATE_S", "30"))
# ★workspace 版は 5 秒索引を `step = FO_STRIDE_S // 5` で間引いて使う（既定 15 秒）。
#   コンテナは索引自体を粗く作るので、**実効の走査刻みを合わせる**必要がある:
#     実効刻み = INDEX_STRIDE_S * (FO_STRIDE_S // 5)
#   索引を 15 秒で作るなら FO_STRIDE_S=5（間引かない）で実効 15 秒になり、
#   5 秒で作るなら FO_STRIDE_S=15（3 つおき）で実効 15 秒になる。
FO_STRIDE_S = int(os.environ.get("FO_STRIDE_S", "5"))

C2_INTERVAL = os.environ.get("C2_INTERVAL", "1") == "1"
C2_BRIDGE = float(os.environ.get("C2_BRIDGE", "0.30"))
C2_NONTIME = float(os.environ.get("C2_NONTIME", "0.70"))
C2_ANCHOR_SD = float(os.environ.get("C2_ANCHOR_SD", "30"))
C2_EDGE = os.environ.get("C2_EDGE", "first")
C2_EDGE_WIN = float(os.environ.get("C2_EDGE_WIN", "90"))
C2_EDGE_THR = float(os.environ.get("C2_EDGE_THR", "0.70"))
C2_EDGE_SMOOTH_S = float(os.environ.get("C2_EDGE_SMOOTH_S", "25"))   # ★元の k=5 @5s
C2_EDGE_SHARE = float(os.environ.get("C2_EDGE_SHARE", "0.125"))
C2_EDGE_RESCAN = float(os.environ.get("C2_EDGE_RESCAN", "0"))
C2_EDGE_LADDER = os.environ.get("C2_EDGE_LADDER", "0.30:3")
C2_EDGE_MULTI = int(os.environ.get("C2_EDGE_MULTI", "0"))
C2_AFTER = os.environ.get("C2_AFTER", "1") == "1"
C2_AFTER_ANON = os.environ.get("C2_AFTER_ANON", "0") == "1"
C2_MULTICLASS = os.environ.get("C2_MULTICLASS", "1") == "1"
C2_MC_DILATE = float(os.environ.get("C2_MC_DILATE", "0"))
C2_DSYS_SEG = os.environ.get("C2_DSYS_SEG", "0") == "1"
C2_DSYS_PERRUN = float(os.environ.get("C2_DSYS_PERRUN", "0"))
C2_SHARPEN = float(os.environ.get("C2_SHARPEN", "0"))
C2_RULES = set(x for x in os.environ.get(
    "C2_RULES",
    "lastseen_pair,longest_cooc,after_first_inserted,last_visible,"
    "applied_times,count_inserted,present_at_t,interval_long").split(",") if x)

HERE = Path(__file__).parent
_TS_RE = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")
_R: list = []
_TS = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b")
_BETWEEN_RE = re.compile(r"\bbetween\b.*?\d{1,2}:\d{2}:\d{2}.*?\band\b.*?\d{1,2}:\d{2}:\d{2}", re.I | re.S)
_RETRIEVAL_RE = re.compile(r"when is it retrieved|retrieved from the surgical site", re.I)
_AND_RE = re.compile(r"at the same time|co-?occur|simultaneous|\bboth\b", re.I)
_FIRST_RE = re.compile(r"first visible|first inserted|for the first time", re.I)
_LAST_RE = re.compile(r"last visible|last seen|final and last", re.I)


def set_classes(cls) -> None:
    """索引 npz が申告するクラス列に合わせる（違う検出器を混ぜたら壊れるので明示的に）。"""
    global _FO_CLASSES
    if cls:
        _FO_CLASSES = [str(c) for c in cls]


def phase_timeline(dataset, video_id):
    """★コンテナには工程分類モデルを積まないので常に None（＝combo は FO 索引だけを使う）。"""
    return None


def anon_intervals(dataset, video_id):
    """★コンテナでは匿名区間表を持たない（本番動画には無い）。空を返す。"""
    return []


def question_anchor_times(question: str) -> list[float]:
    """問題文中の `hh:mm:ss` を秒に変換して返す（**絶対時刻**）.

    出現例: `the Sponge visible in the frame at 00:19:49` / `There is one Sponge in
    the frame at 02:43:20. When is it retrieved ...`。
    ★回答フォーマット指示の中の `hh:mm:ss` は**リテラルの書式指定**（数字ではない）なので
      正規表現に一致せず、誤検出しない。
    """
    return [float(int(h) * 3600 + int(m) * 60 + int(s))
            for h, m, s in _TS_RE.findall(str(question))]


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


def _opening(mask, k):
    """連続 k コマ成立している所だけ True にする（誤検出1発で端が前倒しになるのを防ぐ）。"""
    if k <= 1:
        return mask
    c = np.convolve(mask.astype(np.int32), np.ones(k, dtype=np.int32), "same")
    return c[:len(mask)] >= k        # ★"same" は短い入力で max(M,N) 長を返す（上の _dilate 参照）


def _runs_of(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _dilate(w: np.ndarray, grid: float) -> np.ndarray:
    """検出マスクを前後 `FO_DILATE_S` 秒へ広げる（0 なら何もしない）。"""
    if FO_DILATE_S <= 0:
        return w
    k = int(round(FO_DILATE_S / grid)) * 2 + 1
    # ⚠️np.convolve(..., "same") は入力がカーネルより短いと **max(M,N) 長**を返す。
    #   そのまま使うと短いクリップで w と w_anc の shape が食い違って例外→コンテナが落ちる。
    return (np.convolve(w, np.ones(k), "same")[:len(w)] > 0).astype(np.float32)


def _gauss(secs, centers, sd):
    w = np.zeros_like(secs, dtype=np.float32)
    for c in centers:
        w += np.exp(-0.5 * ((secs - c) / max(sd, 1e-6)) ** 2)
    return w


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


def _edge_kind(q: str) -> str:
    s = str(q).lower()
    if _LAST_RE.search(s):
        return "LAST"
    if _FIRST_RE.search(s) and "last" not in s:
        return "FIRST"
    return "OTHER"


def _edge_alloc(z, ci, kind, lo, hi, n, grid, classes=None, mode="any"):
    """平滑化した検出マスクの端(FIRST=立ち上がり / LAST=立ち下がり)± win に予算を寄せる。

    `classes` を渡すと**複数クラスを合成したマスク**の端を取る（`mode="all"` で積）。
    渡さなければ従来どおり単一クラス `ci` のマスク。
    """
    sc = None
    if classes:
        tt, m = fo_class_mask(z, classes, mode, C2_EDGE_THR, _k(C2_EDGE_SMOOTH_S), lo, hi)
    else:
        tt = z["t"].astype(np.float32)
        sc = z["score"][:, ci].astype(np.float32)
        sel = (tt >= lo) & (tt <= hi)
        tt, sc = tt[sel], sc[sel]
        m = _opening(sc >= C2_EDGE_THR, _k(C2_EDGE_SMOOTH_S)) if len(tt) else np.zeros(0, dtype=bool)
    if len(tt) == 0:
        return None
    rs = _runs_of(m)
    for step_ in (C2_EDGE_LADDER.split(",") if (not rs and C2_EDGE_LADDER) else []):
        thr_, k_ = step_.split(":")
        if classes:
            _, m2 = fo_class_mask(z, classes, mode, float(thr_), _k(float(k_) * 5.0), lo, hi)
        else:
            m2 = _opening(sc >= float(thr_), _k(float(k_) * 5.0))
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
            m2 = _opening(sc[sub] >= FO_SCORE_THR, _k(C2_EDGE_SMOOTH_S / 2))
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


class _ClipReq:
    """start_time だけ差し替えた Request の薄いラッパ（元の Request は不変のまま使う）。"""

    def __init__(self, req, lo):
        self._req, self.start_time = req, lo

    def __getattr__(self, k):
        return getattr(self._req, k)


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


def _sharpen(dataset, req, times, half=2.0):
    """各時刻を ±half 秒の範囲で**匿名でなく最も鮮明な**時刻へ寄せる（無ければ元のまま）。
    ★`fo_class` など時刻を答えない形式では許容誤差の制約が無いので自由にずらせる。"""
    ivs = anon_intervals(dataset, req.videoID)
    out = []
    for t in times:
        c = pick_sharpest(dataset, req.videoID, float(t), half, ivs)
        out.append(c if c is not None else float(t))
    return out


def _reg(name, pat):
    def deco(fn):
        _R.append((name, re.compile(pat, re.I), fn))
        return fn
    return deco


def _cls_runs(z, classes, lo, hi, thr=0.30, k=None, mode="any"):
    t, m = fo_class_mask(z, classes, mode, thr, _k(15.0) if k is None else k, lo, hi)
    return t, m, _runs_of(m)


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


def is_anon(t: float, ivs: list[tuple[float, float]], pad: float = 0.5) -> bool:
    return any(a - pad <= t <= b + pad for a, b in ivs)


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

