"""クリップから**必要なフレームだけ**を取り出す（decord の一括読み）。

★公式テンプレートが推奨する方式:
    "The clip is decoded with decord using a **single batched read** of the selected
     frame indices, which is far cheaper than a per-frame cv2.VideoCapture loop."

★2 段構成にする（ユーザ設計 2026-09-01）:
    ① 推薦段階: 15 秒刻みの粗いフレームを **1 回だけ**読み、索引(M2F)と工程(CNN)で共有する
    ② VLM 段階: ルールが 64 枚を決めてから、その時刻だけを **もう 1 回**読む
  一括読みは「要る index だけ」なので、全キーフレームを展開するより安い。
  本番仕様（5fps / キーフレーム 5 秒ごと）のクリップで実測:
    索引用 116 枚 = 2.89s（threads=4）／ ffmpeg で全キーフレーム 348 枚 = 7.20s

★時刻は `index / fps` で**厳密に決まる**。キーフレーム時刻を ffprobe で推測する必要が無く、
  「i 枚目 = i*5 秒」という危険な仮定も要らなくなる。
★import 順序に注意: **decord は torch の後**に import する
  （公式の警告: 先に import すると CUDA 初期化が壊れる）。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch  # noqa: F401  ★decord より先に import すること

import decord  # noqa: E402

log = logging.getLogger(__name__)

THREADS = int(os.environ.get("FOCUS_DECORD_THREADS", "4"))


class ClipReader:
    """1 本のクリップを開き、時刻を指定してフレームを取り出す。

    `start_time` はクリップ**外**（元動画のタイムライン）の基準。公式仕様どおり
    クリップ自体は先頭から始まるので、`絶対秒 = start_time + index / fps` で写像する。
    """

    def __init__(self, path: Path, start_time: float = 0.0):
        self.vr = decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=THREADS)
        self.fps = float(self.vr.get_avg_fps()) or 5.0
        self.n = len(self.vr)
        self.start = float(start_time)
        # ★キーフレーム位置。**要求点をキーフレームへ吸着**させると前進復号が消えて速い。
        #   公式仕様（キーフレーム 5 秒ごと）なら 15 秒刻みは全点がキーフレーム上に乗るので
        #   吸着しても位置は変わらない（無害）。仕様外のクリップでは 1.5 倍速くなる。
        #   実測: 116 枚の読み込みが 4.95s → 3.27s（キーフレーム上 12/116 のクリップ）。
        try:
            self.keys = np.asarray(sorted(self.vr.get_key_indices()), np.int64)
        except Exception:                                    # noqa: BLE001
            self.keys = np.zeros(0, np.int64)

    @property
    def duration(self) -> float:
        return self.n / self.fps

    def grid_times(self, stride_s: float, snap: bool = True) -> list[float]:
        """`stride_s` 刻みの**絶対秒**。既定でキーフレームへ吸着させる（復号が速くなる）。"""
        step = max(1, int(round(self.fps * stride_s)))
        idx = np.arange(0, self.n, step, dtype=np.int64)
        if snap and len(self.keys):
            idx = np.unique(self.keys[np.abs(self.keys[:, None] - idx[None, :]).argmin(0)])
        return [round(self.start + int(i) / self.fps, 3) for i in idx]

    def read(self, abs_times: list[float]) -> tuple[list[float], np.ndarray]:
        """指定した絶対秒に**最も近い実フレーム**をまとめて読む。

        返すのは (実際の絶対秒, (T,H,W,3) uint8)。要求時刻ではなく**実時刻**を返すので、
        呼び出し側はそれをそのままラベルに使えばよい。
        """
        if not abs_times:
            return [], np.zeros((0, 0, 0, 3), np.uint8)
        idx = sorted({int(np.clip(round((t - self.start) * self.fps), 0, self.n - 1))
                      for t in abs_times})
        arr = self.vr.get_batch(idx).asnumpy()
        return [round(self.start + i / self.fps, 3) for i in idx], arr

    def close(self) -> None:
        self.vr = None
