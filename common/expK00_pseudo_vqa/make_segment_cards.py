"""SEGMENT 擬似 QA の目視カード（クリップの全フレームを帯で並べ、Q/A と一緒に確認する）."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

HERE = Path(__file__).parent
PACKED = HERE / "out/packed"
N_PER_FAMILY = 1
STRIP_MAX = 10


def hhmmss(s: float) -> str:
    s = int(round(s)); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def main() -> None:
    df = pd.read_parquet(HERE / f"out/pseudo_segment_{VERSION}_packed.parquet")
    sample = (df.groupby("family").sample(N_PER_FAMILY, random_state=3)
                .sort_values("family").reset_index(drop=True))
    rows = len(sample)
    fig, axes = plt.subplots(rows, 1, figsize=(20, 3.6 * rows))
    fig.suptitle(f"expK00 pseudo_segment_{VERSION} — SEGMENT pseudo QA built from FRAME annotations "
                 "(clip = the annotated frames themselves)", fontsize=15, y=0.995)
    for i, r in sample.iterrows():
        ax = axes[i] if rows > 1 else axes
        paths = json.loads(r.frame_paths)
        times = json.loads(r.frame_times)
        if len(paths) > STRIP_MAX:
            idx = [round(k) for k in
                   [j * (len(paths) - 1) / (STRIP_MAX - 1) for j in range(STRIP_MAX)]]
            paths = [paths[k] for k in idx]; times = [times[k] for k in idx]
        ims = [Image.open(PACKED / p).convert("RGB") for p in paths]
        h = min(im.height for im in ims)
        ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
        W = sum(im.width for im in ims)
        strip = Image.new("RGB", (W, h), "black")
        x = 0
        for im in ims:
            strip.paste(im, (x, 0)); x += im.width
        ax.imshow(strip); ax.axis("off")
        x = 0
        for im, t in zip(ims, times):
            ax.text(x + 6, 24, hhmmss(t), color="#00ff40", fontsize=9, weight="bold")
            x += im.width
        q = textwrap.fill(r.question, 150)
        if len(q) > 300:
            q = q[:300] + "…"
        ax.set_title(f"[{r.family} | {r.answer_format} | {r.n_frames}f | "
                     f"{hhmmss(r.start_time)}-{hhmmss(r.end_time)} | {r.split}]\n{q}",
                     fontsize=9, loc="left")
        ax.text(0.0, -0.06, f"A: {r.answer}", transform=ax.transAxes, fontsize=12,
                color="green", weight="bold", va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = HERE / (f"out/segment_qa_cards_{VERSION}.png" if VERSION != "v1" else "out/segment_qa_cards.png")
    fig.savefig(out, dpi=72)
    print("saved", out)


VERSION = "v1"

if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--version", default="v1")
    VERSION = _ap.parse_args().version
    main()
