"""uniform と sharp が **実際に選んだフレーム列**を並べて目視する（フィルムストリップ）.

CLAUDE.md「スコアの前に出力を見ろ」の実行。arm C が time を壊した機序を、
数値ではなく**選ばれた絵そのもの**で確認するために作った。

1問につき2段:
  上段 = uniform（対照）  /  下段 = sharp（匿名回避+鮮明選択）
各サムネイルの下に `[HH:MM:SS]` と鮮明度。枠の色で状態を示す:
  **青枠** = sharp で時刻が動いたフレーム / **黄枠** = ベタ塗り（匿名）フレーム
  **緑の縦帯** = GT 時刻に最も近いフレーム（time 問のみ）
※手術映像そのものには描画しない（フレーム外の枠のみ）ので、映像上の配色規約は無関係。

Usage:
  PYTHONPATH=$PWD/reference/src .venv/bin/python \
    workspace/expE01_segproc_baseline/make_selection_strips.py \
    --pick flip_bad --n 6 --out selection_strips_flipbad.png
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image

for _fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/migmix/migmix-2m-regular.ttf"]:
    if Path(_fp).exists():
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
        break

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
logging.basicConfig(level=logging.WARNING)
import dataset_seg as D  # noqa: E402

R = HERE / "results"
CTRL, SHARP = "eval_dl2_seg32f768_ctrl_n2000", "eval_dl2_seg32f768_sharp_n2000"


def hhmmss(t: float) -> str:
    t = int(round(t))
    return f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"


def flat_and_sharp(p: Path) -> tuple[bool, float]:
    """(ベタ塗りか, Laplacian 分散) — サムネイル生成のついでに測る。"""
    im = Image.open(p)
    im.draft("RGB", (128, 128))
    a = np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3)
    return float(a.std(axis=0).max()) < 3.0, 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", default="flip_bad",
                    choices=["flip_bad", "flip_good", "anon_heavy"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=16, help="1行に出すサムネ数の上限")
    ap.add_argument("--out", default="selection_strips.png")
    a = ap.parse_args()

    ca = pd.read_csv(R / CTRL / "results_merged.csv")
    cb = pd.read_csv(R / SHARP / "results_merged.csv")
    m = ca[["uid", "fmt", "group", "correct"]].merge(
        cb[["uid", "correct"]], on="uid", suffixes=("_ctrl", "_sharp"))

    kw = dict(track='SEGMENT', fold=0, part='val', n_frames=32, size=768, limit=2000,
              version='v004', extract=False, anchor=True, anon_note=True)
    U = {x.uid: x for x in D.build_samples(frame_select='uniform', **kw)}
    S = {x.uid: x for x in D.build_samples(frame_select='sharp', **kw)}
    import json
    ra = {r["uid"]: r for r in json.loads((R / CTRL / "responses.json").read_text())}
    rb = {r["uid"]: r for r in json.loads((R / SHARP / "responses.json").read_text())}

    if a.pick == "flip_bad":
        sel = m[(m.fmt == "time") & m.correct_ctrl & ~m.correct_sharp]
    elif a.pick == "flip_good":
        sel = m[(m.fmt == "time") & ~m.correct_ctrl & m.correct_sharp]
    else:
        sel = m[m.uid.isin([u for u in U if U[u].anon_ranges])]
    uids = [u for u in sel.uid if u in U and u in S][:a.n]
    if not uids:
        print("該当なし"); return

    tbl = D.sharpness_table(768)
    nrow = len(uids) * 2
    ncol = a.max_frames
    fig = plt.figure(figsize=(ncol * 1.45, len(uids) * 4.3))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.75, wspace=0.06,
                          top=0.965, bottom=0.012, left=0.008, right=0.992)

    for qi, uid in enumerate(uids):
        su, ss = U[uid], S[uid]
        tu, ts = list(su.frame_times), list(ss.frame_times)
        moved = [i for i in range(min(len(tu), len(ts))) if abs(tu[i] - ts[i]) > 1e-6]
        # 動いたフレームが見える窓を切り出す（全32枚は横に長すぎる）
        c = moved[len(moved) // 2] if moved else len(tu) // 2
        lo = max(0, min(c - ncol // 2, len(tu) - ncol))
        idx = list(range(lo, min(lo + ncol, len(tu))))

        gt = ra[uid]["answer"]
        head = (f"[{qi+1}] {ra[uid]['group']} / {ra[uid]['fmt']}  "
                f"clip {hhmmss(su.start_time)}–{hhmmss(su.end_time)} "
                f"({su.end_time-su.start_time:.0f}s, 許容±{min(5.0,1+(su.end_time-su.start_time)*4/360):.2f}s)\n"
                f"Q: {ra[uid]['question'][:150]}\n"
                f"GT: {gt}    PRED(uniform): {ra[uid]['content'][:40]} "
                f"[{'○' if ra[uid]['content'] else '-'}]    "
                f"PRED(sharp): {rb[uid]['content'][:40]}")
        fig.text(0.008, 1 - (qi * 2) / nrow - 0.004, head, ha="left", va="top",
                 fontsize=9.5, family="monospace")

        for ri, (tag, times, samp) in enumerate((("uniform", tu, su), ("sharp", ts, ss))):
            for ci, i in enumerate(idx):
                ax = fig.add_subplot(gs[qi * 2 + ri, ci])
                ax.set_xticks([]); ax.set_yticks([])
                p = samp.frame_paths[i]
                try:
                    im = Image.open(p); im.draft("RGB", (192, 192))
                    ax.imshow(im.convert("RGB"))
                except Exception:
                    ax.text(.5, .5, "n/a", ha="center", va="center")
                t = times[i]
                stem = Path(samp.videoID).stem
                sc = tbl.get((samp.dataset, stem, int(round(t * 1000))), float("nan"))
                is_flat = D.is_anon(t, D.anon_intervals(samp.dataset, samp.videoID))
                col, lw = "0.6", 1.0
                if ri == 1 and i in moved:
                    col, lw = "#1f77ff", 3.0        # 動いた
                if is_flat:
                    col, lw = "#ffcc00", 3.0        # 匿名
                for s in ax.spines.values():
                    s.set_color(col); s.set_linewidth(lw)
                ax.set_xlabel(f"{hhmmss(t)}\n{sc:.0f}", fontsize=7.5, labelpad=1.5)
                if ci == 0:
                    ax.set_ylabel(tag, fontsize=10, fontweight="bold")

    fig.text(0.5, 0.998, "青枠=sharp で時刻が動いた / 黄枠=ベタ塗り(匿名) / "
             "下段の数値=Laplacian 分散(鮮明度)", ha="center", va="top", fontsize=10)
    out = HERE / a.out
    fig.savefig(out, dpi=95, bbox_inches="tight")
    print(f"wrote {out}  ({len(uids)} 問)")


if __name__ == "__main__":
    main()
