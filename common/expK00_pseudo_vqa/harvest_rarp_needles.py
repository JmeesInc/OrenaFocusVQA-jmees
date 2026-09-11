"""SAR-RARP50 から Suturing needle の RGBA crop を収穫する（cut-paste 合成の素材）.

- train 46動画の 1Hz セマンティックマスク（class 4 = Suturing needle）から連結成分を抽出
- 針は把持鉗子で分断されて複数 CC になるので **closing(15px) 後に CC を取り、
  1 CC のフレームだけ採用**（複数針/分断の曖昧さを除外）
- 同一動画内 10 秒間隔で間引き（ほぼ同じ見た目の crop を量産しない）
- RGBA（alpha=マスクを 1px 膨張 + 2px feather）で out/rarp_needle_crops/ に保存
"""
from __future__ import annotations

import glob
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

RARP = Path("/data4/shared/miccai/EndoVis2022/SAR-RARP50/train")
HERE = Path(__file__).parent
OUT = HERE / "out/rarp_needle_crops"
NEEDLE_ID = 4
CLOSE_KERNEL = 15
MIN_AREA = 600          # 1080p 基準。小さすぎる針片はノイズ
MAX_AREA = 40000
MIN_GAP_S = 10.0        # マスクは 1Hz（60 フレーム毎）
PAD = 6

log = logging.getLogger("expK00.rarp")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(HERE / f"out/harvest_{ts}.log"),
                                  logging.StreamHandler()])
    meta = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    for vdir in sorted(RARP.glob("video_*")):
        last_t = -1e9
        n_video = 0
        for mp in sorted(glob.glob(str(vdir / "segmentation/*.png"))):
            frame_no = int(Path(mp).stem)
            t = frame_no / 60.0
            if t - last_t < MIN_GAP_S:
                continue
            m = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            if m.ndim == 3:
                m = m[..., 0]
            needle = (m == NEEDLE_ID).astype(np.uint8)
            if needle.sum() < MIN_AREA:
                continue
            closed = cv2.morphologyEx(needle, cv2.MORPH_CLOSE, kernel)
            n_cc, lab = cv2.connectedComponents(closed)
            if n_cc - 1 != 1:
                continue                      # 分断/複数針は曖昧なので捨てる
            area = int(needle.sum())
            if not (MIN_AREA <= area <= MAX_AREA):
                continue
            rgb_path = vdir / "rgb" / Path(mp).name
            img = cv2.imread(str(rgb_path))
            if img is None:
                continue
            ys, xs = np.nonzero(needle)
            y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, needle.shape[0])
            x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, needle.shape[1])
            crop = img[y0:y1, x0:x1]
            alpha = needle[y0:y1, x0:x1] * 255
            alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8))
            alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
            rgba = np.dstack([crop, alpha])
            name = f"{vdir.name}_{frame_no:09d}.png"
            cv2.imwrite(str(OUT / name), rgba)
            meta.append({"file": name, "video": vdir.name, "frame": frame_no,
                         "area": area, "w": int(x1 - x0), "h": int(y1 - y0)})
            last_t = t
            n_video += 1
        log.info(f"{vdir.name}: {n_video} crops")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info(f"total {len(meta)} needle crops → {OUT}")


if __name__ == "__main__":
    main()
