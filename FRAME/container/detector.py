"""FO instance segmentation を **VQA フレームへの視覚ヒント**として使う（v009）.

v008 からの変更点（すべて **expM00-A の学習条件に合わせるため**）:

| # | v008 | v009 |
|---|---|---|
| 検出器 | expF23 単体（8クラス） | **expF40（7クラス, Gallstone 除外）+ expF39（clip specialist, 768x1344）** |
| conf | 0.5 | **0.25** |
| 描画 | 原寸に描いてから縮小 | **モデルが見る幅に縮めてから描く** |
| ラベル | 内部クラス名（`Silicon_Loop`）| **公式表記（`Silicone Loop`）** |
| インスタンス | 全部にクラス名 | **クラス名は「1番」だけ + 番号**（variant r1） |
| 検出0件 | 原画像と同一の2枚目を付ける | **2枚目を付けない**（expG00 §39: 空の重畳は −0.0429） |

★描画の実体は `overlay_render.py`（**学習コードの逐語コピー**）。ここは薄いラッパで、
  **描画規則をこのファイルに書かない**こと（2箇所に散ると必ずズレる）。

Source:
  detector      = workspace/expF00_fo_instseg/results/expF40_m2f_ps1sN_noGallstone_0828/fold0/best_model
                  （instseg fold0 val: segm AP50 0.5465 / neg_clean 0.933）
  detector_clip = workspace/expF00_fo_instseg/results/expF39_m2f_clip_hires_strongaug_0828/fold0/best_model
                  （clip AP50 0.4207 — 全構成中トップ。**768x1344 で学習**しているので推論も同解像度）
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class Detector:
    """全クラス検出器 + clip specialist。`overlay(PIL) -> (PIL|None, note)`。

    ⚠️ **fail-soft の扱い**: ロードに失敗したら `ok=False` にして重畳なしで走る。
       ただし v008 で「scipy 欠落で静かに重畳なしになり ✓ が出た」事故があったので、
       **失敗は必ず ERROR ログに残す**（呼び出し側が件数で検算する）。
    """

    def __init__(self, ckpt: Path, conf: float = 0.25,
                 clip_ckpt: Path | None = None, variant: str = "r1"):
        self.ok = False
        self.conf, self.variant = conf, variant
        self._cache: dict = {}   # (qID, 幅) → 検出結果（メンバー間で共有）
        try:
            from overlay_render import M2F, MergedDetector
            spec = {}
            if clip_ckpt is not None and Path(clip_ckpt).exists():
                # ★expF39 は 768x1344 で学習。既定の 512x896 で回すと clip（数ピクセル）が落ちる
                spec["Clip"] = M2F(clip_ckpt, conf=conf, height=768, width=1344)
            else:
                log.error("★clip specialist が無い: %s — 全クラス検出器の clip を使う", clip_ckpt)
            self.det = MergedDetector(M2F(ckpt, conf=conf), spec, conf=conf, conf_lo=conf)
            self.ok = True
            log.info("detector: all=%s clip=%s conf>=%.2f variant=%s",
                     ckpt, clip_ckpt, conf, variant)
        except Exception:
            log.exception("★detector のロードに失敗 — 重畳なしで走る（スコアは落ちる）")

    def overlay(self, pil_img, variant: str = "r4", key=None):
        """PIL(RGB, **既に member の幅**) → (重畳 PIL | None, 説明文)。

        ★**メンバーごとに描画バリアントが違う**（A は r1 / E は r4）。
          検出そのものはバリアントに依らないので、`key=(qID, 幅)` で 1 問 1 回に畳む。
          幅が違うと検出結果も変わる（検出は表示サイズの画像に対して行う）ので、
          **キーには必ず幅を含める**こと。
        ★入力は**モデルが見るサイズ**で渡すこと。ここでは resize しない
          （学習時 `render_cache.py` が `resize_to(out_width)` 済みの画像に描いたのと同条件）。
        """
        if not self.ok:
            return None, ""
        try:
            import cv2
            import numpy as np
            from overlay_render import render
            img = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
            if key is not None and key in self._cache:
                dets = self._cache[key]
            else:
                dets = self.det.predict(img)
                if key is not None:
                    # ★1問ぶんだけ持てばよい（メンバー間で共有）。全問ぶん抱えると
                    #   ホスト RAM を食う（マスクは H×W×N の uint8）。
                    self._cache = {key: dets}
            pil, meta = render(img, dets, variant=variant,
                               conf=self.conf, conf_lo=self.conf)
            return pil, (meta.get("note", "") if pil is not None else "")
        except Exception:
            log.exception("★重畳に失敗 — この問は原画像のみで答える")
            return None, ""
