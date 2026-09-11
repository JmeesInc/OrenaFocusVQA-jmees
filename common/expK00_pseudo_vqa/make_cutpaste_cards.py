"""cut-paste 合成の目視カード（貼り付け品質と Q/A 整合の確認用）."""
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).parent
df = pd.read_parquet(HERE / "out/pseudo_frame_cutpaste_v1.parquet")
sample = df.groupby("family").sample(2, random_state=11).sort_values("family").reset_index(drop=True)
cols, rows = 3, (len(sample) + 2) // 3
fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 5.4 * rows))
fig.suptitle("expK00 pseudo_frame_cutpaste_v1 — RARP needle cut-paste onto FOCUS frames", fontsize=15)
for ax in axes.flat:
    ax.axis("off")
for i, r in sample.iterrows():
    ax = axes.flat[i]
    ax.imshow(Image.open(r.frame_path))
    q = textwrap.fill(r.question, 92)
    if len(q) > 240:
        q = q[:240] + "…"
    ax.set_title(f"[{r.family} | n_pasted={r.n_pasted} | {r.split}]\n{q}", fontsize=9, loc="left")
    ax.text(0, -0.05, f"A: {r.answer}", transform=ax.transAxes, fontsize=12, color="green",
            weight="bold", va="top")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(HERE / "out/cutpaste_cards.png", dpi=90)
print("saved")
