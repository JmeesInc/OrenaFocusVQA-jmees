"""描画バリアント（r0/r1/r2/r3）を **採点の前に目視**するためのカード画像.

「スコアの前に出力を見ろ」（CLAUDE.md）。768px でラベル文字が読めるのか、
r2 の色分けが組織と区別できるのか、r3 の低信頼輪郭が邪魔でないのかを先に見る。

    python make_variant_cards.py --n 6 --out variant_cards.png

★フレームは **FRAME val (fold0) = 検出器が一度も学習に使っていない 26 動画**から取る。
  「同一クラスが複数写っている」フレームを優先して選ぶ（r1 の番号付けが見たいので）。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "workspace/expE01_segproc_baseline"))
sys.path.insert(0, str(ROOT / "reference/src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("expM00.cards")

from dataset_seg import build_samples, grid_for  # noqa: E402
from overlay_render import (VARIANTS, M2F, MergedDetector, count_text,  # noqa: E402
                            render, resize_to)

F40 = ROOT / "workspace/expF00_fo_instseg/results/expF40_m2f_ps1sN_noGallstone_0828/fold0/best_model"
F39 = ROOT / "workspace/expF00_fo_instseg/results/expF39_m2f_clip_hires_strongaug_0828/fold0/best_model"


def wrap(text: str, width: int, fs: float, th: int) -> list[str]:
    out, line = [], ""
    for w in text.split():
        t = (line + " " + w).strip()
        if cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0] > width - 16:
            out.append(line)
            line = w
        else:
            line = t
    if line:
        out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--width", type=int, default=768, help="モデルが見る幅（描画の基準）")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--conf-lo", type=float, default=0.25)
    ap.add_argument("--out", default="variant_cards.png")
    ap.add_argument("--limit-scan", type=int, default=400, help="候補として走査するフレーム数")
    a = ap.parse_args()

    samples = build_samples("FRAME", 0, "val", n_frames=1, size=args_size(a.width),
                            grid=grid_for("FRAME"))
    log.info("val %d 問", len(samples))
    # ★expF39(clip specialist) は **768x1344 で学習**している。既定の 512x896 で
    #   推論すると学習解像度と違う（clip は数ピクセルなのでここが効く）。
    det = MergedDetector(M2F(F40, conf=a.conf_lo),
                         {"Clip": M2F(F39, conf=a.conf_lo, height=768, width=1344)},
                         conf=a.conf, conf_lo=a.conf_lo)

    # 候補走査: 同一クラス複数インスタンスを優先
    seen: set[str] = set()
    cand = []
    for s in samples:
        fp = str(s.frame_paths[0])
        if fp in seen:
            continue
        seen.add(fp)
        img = resize_to(cv2.imread(fp, cv2.IMREAD_COLOR), a.width)
        dets = det.predict(img)
        hi = [d for d in dets if d["score"] >= a.conf]
        per: dict[str, int] = {}
        for d in hi:
            per[d["cls"]] = per.get(d["cls"], 0) + 1
        if hi:
            cand.append((max(per.values()), len(hi), s, img, dets))
        if len(seen) >= a.limit_scan:
            break
    cand.sort(key=lambda t: (-t[0], -t[1]))
    pick = cand[:a.n]
    log.info("候補 %d / 走査 %d フレーム → %d 枚を描画", len(cand), len(seen), len(pick))

    fs, th = 0.5, 1
    rows = []
    for _, _, s, img, dets in pick:
        cells = []
        for v in ("r1","r4"):
            pil, meta = render(img, dets, variant=v, conf=a.conf, conf_lo=a.conf_lo)
            cell = (img.copy() if pil is None
                    else cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
            head = f"{v}  n_hi={meta['n_hi']} n_lo={meta['n_lo']}"
            note = meta.get("note", "(検出0件 → 重畳なし = 単画像)")
            lines = [head] + wrap(note, cell.shape[1], fs, th)
            if v == "r0":
                lines += ["T1 text: " + count_text([d for d in dets
                                                    if d['score'] >= a.conf])]
            pad = np.full((18 * len(lines) + 12, cell.shape[1], 3), 30, np.uint8)
            for i, ln in enumerate(lines):
                cv2.putText(pad, ln, (8, 18 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                            (255, 255, 255), th, cv2.LINE_AA)
            cells.append(np.vstack([cell, pad]))
        h = max(c.shape[0] for c in cells)
        cells = [np.vstack([c, np.full((h - c.shape[0], c.shape[1], 3), 30, np.uint8)])
                 for c in cells]
        tag = np.full((26, sum(c.shape[1] for c in cells), 3), 15, np.uint8)
        cv2.putText(tag, f"{s.videoID}  t={s.frame_times[0]:.0f}s  uid={s.uid}",
                    (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
        rows.append(np.vstack([tag, np.hstack(cells)]))
    card = np.vstack(rows)
    cv2.imwrite(str(HERE / a.out), card)
    log.info("書き出し %s  %s", HERE / a.out, card.shape)
    return 0


def args_size(width: int) -> int:
    """frames_cache に存在する解像度へ丸める（768 / 1024 / 448）。"""
    return 768 if width <= 768 else 1024


if __name__ == "__main__":
    raise SystemExit(main())
