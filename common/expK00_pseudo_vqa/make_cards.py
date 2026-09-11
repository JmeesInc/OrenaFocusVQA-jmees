"""pseudo_frame_v1 の QA カード — 全 family ×2 件をアノテーション重畳付きで格子表示.

描画規約（CLAUDE.md）: 物体の塗りは青・輪郭太線、ラベル文字は緑。
"""
from __future__ import annotations

import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_util

HERE = Path(__file__).parent
DUMPS = {
    "heico": Path("/mnt/data/data4/input/survis-anno/focus-heico/focus-heico_20260825_001"),
    "lapchole": Path("/mnt/data/data4/input/survis-anno/focus-lapchole/focus-lapchole_20260825_001"),
}
SEED = 7
N_PER_FAMILY = 2


def load_anns() -> dict:
    per = {}
    for ds, dump in DUMPS.items():
        coco = json.load(open(dump / "coco/instances.json"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        img = {im["id"]: (im["video"], im["frame_number"]) for im in coco["images"]}
        by = defaultdict(list)
        for a in coco["annotations"]:
            v, fn = img[a["image_id"]]
            by[(ds, v, fn)].append((cats[a["category_id"]], a))
        per.update(by)
    return per


def main() -> None:
    df = pd.read_parquet(HERE / f"out/pseudo_frame_{VERSION}.parquet")
    anns = load_anns()
    sample = (df.groupby("family").sample(N_PER_FAMILY, random_state=SEED)
                .sort_values("family").reset_index(drop=True))

    n = len(sample)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 5.6 * rows))
    fig.suptitle(f"expK00 pseudo_frame_{VERSION} — pseudo FRAME QA generated from instseg annotations "
                 "(blue = annotation overlay / green = class label)", fontsize=16)
    for ax in axes.flat:
        ax.axis("off")
    for i, r in sample.iterrows():
        ax = axes.flat[i]
        im = np.array(Image.open(r.frame_path).convert("RGB"), dtype=np.float32)
        for cls, a in anns.get((r.dataset, r.video, r.frame_number), []):
            try:
                m = mask_util.decode(a["segmentation"]).astype(bool)
                im[m] = im[m] * 0.45 + np.array([0, 60, 255]) * 0.55
            except Exception:
                pass
            x, y, w, h = a["bbox"]
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="#00c8ff", lw=3))
            ax.text(x, max(y - 6, 10), cls, color="#00ff40", fontsize=11, weight="bold")
        h_img, w_img = im.shape[:2]
        ax.axhline(h_img / 2, color="white", lw=0.6, alpha=0.5)
        ax.axvline(w_img / 2, color="white", lw=0.6, alpha=0.5)
        ax.imshow(im.astype(np.uint8))
        q = textwrap.fill(r.question, 88)
        if len(q) > 260:
            q = q[:260] + "…"
        title = (f"[{r.family} | {r.answer_format} | {r.split}"
                 f"{' | instseg-train' if r.instseg_train else ''}]\n{q}")
        ax.set_title(title, fontsize=9, loc="left")
        ax.text(0.0, -0.04, f"A: {r.answer}", transform=ax.transAxes, fontsize=12,
                color="green", weight="bold", va="top")
        ax.text(0.0, -0.11, f"{Path(r.video).stem}  frame {r.frame_number}  t={r.timestamp:.0f}s",
                transform=ax.transAxes, fontsize=8, color="gray", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = HERE / (f"out/qa_cards_{VERSION}.png" if VERSION != "v1" else "out/qa_cards.png")
    fig.savefig(out, dpi=90)
    print(f"saved {out}")


VERSION = "v1"

if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--version", default="v1")
    VERSION = _ap.parse_args().version
    main()
