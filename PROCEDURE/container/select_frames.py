"""クリップ 1 本ぶんの「デコード → 索引 → 工程 → 64 枚の時刻決定」をまとめる。

★デコードは **1 回だけ**。本番クリップはキーフレームが 5 秒ごとにある（公式仕様）ので、
  全キーフレームを 1 度取り出して 3 用途へ配る:
      15 秒間引き → FO 索引（Mask2Former / TRT）
      5 秒のまま  → 工程 timeline（ConvNeXtV2 + MS-TCN）
      5 秒格子    → ルールが選んだ 64 枚を VLM へ
  索引・工程のためだけに追加でデコードしない、というのが時間予算の要。

★索引は動画ごとに決まるので `videoID` をキーに日和見キャッシュへ書く。
  ⚠️1 バッチに同一動画は 2 問入らない仕様なので、効くのは**バッチ間**だけ。
    grand-challenge が実行間でファイルシステムを残す保証は無いので、
    「残ったら儲けもの・残らなくても損しない」作りにしてある。
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

import fo_index
import phase_index
import rules_proc as RP

log = logging.getLogger(__name__)

N_FRAMES = int(os.environ.get("FOCUS_N_FRAMES", "64"))
GRID_S = 5.0
INDEX_STRIDE_S = float(os.environ.get("FOCUS_INDEX_STRIDE_S", "15"))
PHASE_STRIDE_S = float(os.environ.get("FOCUS_PHASE_STRIDE_S", "5"))
USE_PHASE = os.environ.get("FOCUS_USE_PHASE", "1") == "1"
MIN_INDEX_PTS = int(os.environ.get("FOCUS_MIN_INDEX_PTS", "96"))  # これ以上は粗くしない
# ★粗走査デコードが予算を食い切っても、検出器にはこれだけは残す（索引を丸ごと失わない）
MIN_INDEX_BUDGET_S = float(os.environ.get("FOCUS_MIN_INDEX_BUDGET_S", "3.0"))
LONG_CLIP_S = float(os.environ.get("FOCUS_LONG_CLIP_S", "3600"))   # これ以上なら粗→密
COARSE_S = float(os.environ.get("FOCUS_COARSE_S", "60"))           # 1 段目の刻み
FINE_S = float(os.environ.get("FOCUS_FINE_S", "15"))               # 2 段目＝最終格子
COARSE_THR = float(os.environ.get("FOCUS_COARSE_THR", "0.20"))     # 粗い段は緩めに拾う
RECO_STRIDE_S = float(os.environ.get("FOCUS_RECO_STRIDE_S", "15"))       # 推薦段階の既定刻み
RECO_TARGET_PTS = int(os.environ.get("FOCUS_RECO_TARGET_PTS", "800"))    # 目標枚数
RECO_MAX_STRIDE_S = float(os.environ.get("FOCUS_RECO_MAX_STRIDE_S", "30"))


def plan_secs(keys: list[float], stride_s: float) -> list[float]:
    """キーフレーム時刻の列から、目標刻みに最も近いものを選ぶ。

    ★「キーフレームは 5 秒ごと」を仮定して `[::3]` のように間引いてはいけない。
      実測ではクリップによって 2.0〜2.8 秒とばらつく（video.keyframe_times 参照）。
    """
    if not keys:
        return []
    ka = np.asarray(keys, np.float64)
    want = np.arange(ka[0], ka[-1] + stride_s, stride_s)
    return [keys[int(i)] for i in np.unique(np.abs(ka[:, None] - want[None, :]).argmin(0))]


def plan_models(req, answer_format: str | None) -> tuple[bool, bool]:
    """★質問文だけで「どのモデルを回す必要があるか」を先に決める（ユーザ設計 2026-09-01）。

    返り値 `(索引が要る, 工程が要る)`。工程は val 2,000 問中 **414 問（21%）でしか要らない**
    ＝ 79% の問で 8〜18 秒を丸ごと省ける。

    ⚠️**索引は常に要るとみなす**。`combo2_select_times` は冒頭で無条件に `fo_timeline()` を
      呼び、①interval も ②bridge も条件を満たさなければ後段へ落ちる
      （interval は 10 分超なら interval_long へ / bridge は配分が n/4 未満なら次へ）。
      「アンカーが 2 つなら索引不要」と単純化したら **204 問で取りこぼした**（2026-09-01 実測）。
      分岐を要約せず、**実挙動で検証してから**でなければ飛ばしてはいけない。
    """
    q = str(req.question)
    anc = RP.question_anchor_times(q)
    need_phase = bool(anc)          # 工程は combo のアンカー経路でしか使わない
    if need_phase:
        for name, pat, _fn in RP._R:            # ⑫テンプレルールは工程を使わない
            if name in RP.C2_RULES and pat.search(q):
                need_phase = False
                break
    return True, need_phase


def _img(v):
    """パスなら開いて配列にする。既に配列ならそのまま。"""
    if isinstance(v, np.ndarray):
        return v
    from PIL import Image
    with Image.open(v) as im:
        return np.asarray(im.convert("RGB"))


class ClipContext:
    """1 問ぶんの索引と工程を保持し、`times()` で 64 枚の時刻を返す。

    ★`decoded` は「秒 → JPEG パス」。**必要な枚だけ**その場で開く。
      全キーフレーム（3 時間で 4,168 枚）を先に numpy 化すると、それだけで 36.6 秒かかるのに、
      索引に使うのは 713 枚・工程は 2,136 枚だけ（2026-09-01 実測）。
    """

    def __init__(self, req, det, decoded: dict[float, object], budget_s: float | None = None,
                 need_phase: bool = True):
        self.req = req
        lo, hi = float(req.start_time), float(req.end_time)
        self.z = self._index(req, det, decoded, lo, hi, budget_s)
        self.ph = self._phase(req, decoded, lo, hi) if (USE_PHASE and need_phase) else None

    @classmethod
    def from_reader(cls, req, det, reader, budget_s: float | None = None,
                    answer_format: str | None = None):
        """★推薦段階: **粗いフレームを 1 回だけ読み、索引と工程で共有する**。

        索引(M2F) も工程(CNN) も同じ 15 秒刻みのフレームで足りるので、
        decord の一括読みを 1 回で済ませる。VLM に渡す 64 枚は
        ルールが時刻を決めてから**別途**読む（`ClipReader.read`）。
        """
        need_idx, need_ph = plan_models(req, answer_format)
        stride = cls.reco_stride(reader.duration)
        want = reader.grid_times(stride)
        # ★★`budget_s` は**推薦段階ぜんたい**の予算として扱う。以前は検出器にしか効いておらず、
        #   その前段の粗走査デコードが予算の外にあった。長尺（540〜712枚）ではデコードだけで
        #   9〜13 秒かかるので、予算 15s のつもりが実測 23〜28 秒になり、**問が 30s を超えて
        #   不正解になっていた**（2026-09-11 dl2 実測）。デコードに使った分を差し引いて渡す。
        # ★★デコードは「読む前に枚数を決める」しかないので、予算から逆算して間引く。
        #   実測 43 枚/s（dl2 / RTX4090 / --cpus 4 / 1024x576 5fps）。予算の 60% をデコードに割く。
        #   ⚠️効くのは **枚数が上限を超える長尺だけ**（9,000s 超で 540〜712 枚）。
        #     短中尺は 42〜277 枚なので**一切変わらない**＝そこの索引精度は落ちない。
        #   長尺は刻みが 15s→30s へ粗くなるが、**30s 超過で 0 点になるより粗い索引の方が良い**
        #     （走査刻みは 60s でも一様比 1.61 倍の到達率を保つ: expI03）。
        if budget_s is not None:
            dec_fps = float(os.environ.get("FOCUS_RECO_DECODE_FPS", "43"))
            max_pts = max(int(budget_s * 0.6 * dec_fps), MIN_INDEX_PTS)
            while len(want) > max_pts and len(want) > MIN_INDEX_PTS:
                want = want[::2]
                stride *= 2
        _t0 = time.time()
        secs, arr = reader.read(want)
        decoded = {t: arr[i] for i, t in enumerate(secs)}
        dec_s = time.time() - _t0
        left = None if budget_s is None else max(budget_s - dec_s, MIN_INDEX_BUDGET_S)
        log.info("推薦用フレーム: %d枚 刻み%.0fs %.1fs（索引=%s 工程=%s 残予算=%s）",
                 len(secs), stride, dec_s, need_idx, need_ph,
                 "なし" if left is None else f"{left:.1f}s")
        return cls(req, det if need_idx else None, decoded, left, need_phase=need_ph)

    @staticmethod
    def reco_stride(dur_s: float) -> float:
        """推薦段階の刻み。既定 15 秒、長尺だけ粗くして枚数を一定に保つ。"""
        st = RECO_STRIDE_S * max(1.0, np.ceil(dur_s / max(RECO_TARGET_PTS, 1) / RECO_STRIDE_S))
        return float(min(st, RECO_MAX_STRIDE_S))

    # --- FO 索引（粗→密の 2 段）--------------------------------------------- #
    def _coarse_to_fine(self, det, decoded, keys, budget_s):
        """★長尺向け: **粗く全体を見て → FO が居た所だけ密に見る**。

        一律に細かく走らせると 178 分の動画で 713 枚 = 78 秒かかる（予算 30 秒）。
        しかし FO が写っているのは動画の一部なので、粗い走査で当たりを付けてから
        その周辺だけ細かく見れば、**同じ密度の索引をずっと安く**作れる。

        ★出力は **一様グリッド（FINE_S 刻み）に揃える**。ルール側は等間隔を前提に
          平滑化（連続 k コマ）しているので、不等間隔の索引を渡すと意味が変わる。
          粗い段階で検出が無かった区間は「検出なし（score=0）」で埋める。
        ★匿名（単色）フレームは検出器に**渡さない**。発火は事実上すべて偽陽性で、
          計算だけ食う（expI04 実測）。
        """
        t0 = time.time()
        coarse = plan_secs(keys, COARSE_S)
        imgs, ok = [], []
        for sec in coarse:
            im = _img(decoded[sec])
            if fo_index.is_blank(im):
                continue
            imgs.append(fo_index.preprocess(im))
            ok.append(sec)
        if not ok:
            return None
        zc = fo_index.build_index(det, imgs, np.asarray(ok, np.float32))
        hit = (zc["score"].astype(np.float32) >= COARSE_THR).any(axis=1)
        n_blank = len(coarse) - len(ok)
        # 検出のあった粗点の周り ±COARSE_S を「密に見る区間」にする
        want = np.arange(keys[0], keys[-1] + FINE_S, FINE_S)
        fine_mask = np.zeros(len(want), bool)
        for sec, h in zip(ok, hit):
            if h:
                fine_mask |= (want >= sec - COARSE_S) & (want <= sec + COARSE_S)
        # 予算に収まるまで密領域を間引く
        fps = float(os.environ.get("FOCUS_INDEX_FPS", "40"))
        left = (budget_s or 1e9) - (time.time() - t0)
        idx = np.flatnonzero(fine_mask)
        while len(idx) / fps > max(left, 1.0) and len(idx) > MIN_INDEX_PTS:
            idx = idx[::2]
        ka = np.asarray(keys, np.float64)
        sel = [keys[int(i)] for i in np.abs(ka[:, None] - want[idx][None, :]).argmin(0)]
        imgs2, ok2, pos2 = [], [], []
        for j, sec in zip(idx, sel):
            im = _img(decoded[sec])
            if fo_index.is_blank(im):
                continue
            imgs2.append(fo_index.preprocess(im))
            ok2.append(sec)
            pos2.append(j)
        n_cls = len(det.classes) if det.classes else 7
        S = np.zeros((len(want), n_cls), np.float32)
        C = np.zeros((len(want), n_cls), np.int16)
        A = np.zeros((len(want), n_cls), np.float32)
        if imgs2:
            zf = fo_index.build_index(det, imgs2, np.asarray(ok2, np.float32))
            S[pos2] = zf["score"].astype(np.float32)
            C[pos2] = zf["count"]
            A[pos2] = zf["area"].astype(np.float32)
        log.info("索引(粗→密): 粗 %d枚(空白%d) → 密 %d枚 / 全 %d 格子 %.1fs",
                 len(ok), n_blank, len(imgs2), len(want), time.time() - t0)
        return {"t": want.astype(np.int32), "score": S.astype(np.float16),
                "count": C.astype(np.int8), "area": A.astype(np.float16),
                "classes": np.array(det.classes if det.classes else [])}

    def _index(self, req, det, decoded, lo, hi, budget_s):
        z = fo_index.load_cached(req.videoID, lo, hi)
        if z is not None:
            return z
        if det is None:
            return None
        # ★索引は **INDEX_STRIDE_S 秒ごと**に間引く。
        #   ⚠️「キーフレームは 5 秒ごと」という前提でインデックス間引き（`[::3]`）をすると
        #     クリップによって実効刻みが変わる。テストクリップは実測 2.6 秒間隔で、
        #     3 時間動画に 1,390 枚（実効 7.7 秒）流れて 186 秒かかった（2026-09-01）。
        #   → **実際の秒を見て**目標刻みに最も近い枚を選ぶ。
        # ⚠️キーは元の Python float のまま扱う（float32 に丸めると dict 参照が外れる）
        keys = sorted(decoded)
        if not keys:
            return None
        # ★長い動画は粗→密で作る（一律に細かくすると予算を食い潰す）
        if COARSE_S > 0 and (keys[-1] - keys[0]) > LONG_CLIP_S:
            return self._coarse_to_fine(det, decoded, keys, budget_s)
        sel = plan_secs(keys, INDEX_STRIDE_S)
        secs = np.asarray(sel, np.float32)
        if len(secs) == 0:
            return None
        # ★長尺は索引だけで予算を食い潰す（178 分 = 713 枚で 78 秒／RTX 8000・TRT 無し）。
        #   **想定 fps から所要を見積もり、予算に収まるまで刻みを倍にする**。
        #   短いクリップは触らないので、そこの精度は落ちない。
        if budget_s is not None:
            fps = float(os.environ.get("FOCUS_INDEX_FPS", "40"))
            while len(secs) / fps > budget_s and len(secs) > MIN_INDEX_PTS:
                sel = sel[::2]
                secs = secs[::2]
            if len(secs) < len(sel):
                sel = sel[:len(secs)]
        t0 = time.time()
        z = fo_index.build_index(det, [fo_index.preprocess(_img(decoded[s])) for s in sel], secs)
        log.info("索引 %s: %d枚 %.1fs", req.videoID, len(secs), time.time() - t0)
        fo_index.save_cached(req.videoID, lo, hi, z)
        return z

    # --- 工程 timeline ------------------------------------------------------ #
    @staticmethod
    def _phase_stride(dur_s: float) -> float:
        """★動画の長さで工程の刻みを決める（枚数を一定に保つ）。

        短いクリップは 5 秒のまま（もともと安い）、長くなるほど粗くする。
        178 分の動画では 5s→2,137 枚 17.2 秒 が 15s→713 枚 7.7 秒になる。
        ⚠️TCN は **stride5 のまま**使う（stride10/20 は平滑化が弱く工程が細切れになる）。
        """
        st = 5.0 * max(1.0, np.ceil(dur_s / max(PHASE_TARGET_PTS, 1) / 5.0))
        return float(min(st, PHASE_MAX_STRIDE_S))

    def _phase(self, req, decoded, lo, hi):
        name = phase_index.route(getattr(req, "procedure_type", None), req.videoID)
        m = phase_index.get(name)
        if m is None:
            return None
        # 工程も同様に「実際の秒」で間引く（キーフレーム間隔はクリップ依存）
        keys = sorted(decoded)
        if not keys:
            return None
        # ★推薦段階では**索引と同じフレーム**を使う（読み直さない）
        secs = keys
        os.environ["FOCUS_PHASE_STRIDE_S"] = str(
            (keys[-1] - keys[0]) / max(len(keys) - 1, 1) if len(keys) > 1 else 5.0)
        if not secs:
            return None
        try:
            t0 = time.time()
            out = m([phase_index.preprocess(_img(decoded[s])) for s in secs],
                    np.array(secs, np.float32), int(hi) + 1)
            log.info("工程 %s(%s): %d枚 %.1fs", req.videoID, name, len(secs), time.time() - t0)
            return out
        except Exception as e:                          # noqa: BLE001
            log.warning("工程 timeline 失敗（工程なしで続行）: %s", e)
            return None

    # --- 時刻の決定 --------------------------------------------------------- #
    def times(self, answer_format: str | None, n: int | None = None) -> list[float]:
        """ルールに従って **n 枚**（既定 N_FRAMES=64）の時刻を返す。索引が無ければ一様に倒す。

        ★v016 で `n` を受け取るようにした。v010 までは `N_FRAMES` 固定で、
          **anytime の梯子が索引経路で完全に無効化されていた**（呼び出し側が枚数を
          落としても常に 64 枚が返る）。2026-09-05 に dl1 のテストで実測:
          自前締切 612s を超えても 3 問目に 64 枚を渡し続けた。
          遅い GPU や異常に長いクリップで**予算を使い切って全問 0 点**になる経路なので塞ぐ。
        ★既定は 64 のまま（学習が 64 枚なので通常運転では**挙動は 1 バイトも変わらない**）。
        """
        nf = int(n) if n else N_FRAMES
        lo, hi = float(self.req.start_time), float(self.req.end_time)
        if self.z is None:
            return _uniform(lo, hi, nf)
        RP.set_stride(float(self.z["t"][1] - self.z["t"][0]) if len(self.z["t"]) > 1 else GRID_S)
        if "classes" in self.z:
            RP.set_classes([str(c) for c in self.z["classes"]])
        RP.fo_timeline = lambda ds, vid: self.z          # ルールは索引を関数経由で引く
        RP.phase_timeline = lambda ds, vid: self.ph
        try:
            got = RP.combo2_select_times("", self.req, answer_format, nf, GRID_S)
        except Exception as e:                           # noqa: BLE001
            log.warning("ルール適用に失敗（一様へ）: %s", e)
            got = None
        return got or _uniform(lo, hi, nf)


def _uniform(lo: float, hi: float, n: int) -> list[float]:
    if hi <= lo or n <= 1:
        return [lo]
    k = int((hi - lo) // GRID_S)
    idx = np.unique(np.round(np.linspace(0, k, n)).astype(int))
    return [round(lo + int(i) * GRID_S, 3) for i in idx]
