"""SEGMENT / PROCEDURE 用 QA カード（フレーム列 = フィルムストリップ表示）.

CLAUDE.md の「全実験で QA カード画像を必ず生成して提示する」に対応。
FRAME 版（`expB00/make_qa_cards.py`）との違いは、**入力が1枚でなくフレーム列**なので
各カードに**時刻ラベル付きのフィルムストリップ**を敷くこと。

出力: `<run_dir>/qa_cards.png`
- 先頭パネル: モデルへ渡した入力の説明（システムプロンプト全文 + フレームの渡し方）
- 各カード: フィルムストリップ（[hh:mm:ss] ラベル付き）/ 質問 / PRED / GT / 正誤 / 形式 / バケット

回答形式を網羅サンプリングする（time / fo_class / number / binary / MC / open_ended / percentage）。

Usage:
  .venv/bin/python workspace/expE01_segproc_baseline/make_qa_cards.py \
     workspace/expE01_segproc_baseline/results/zs_seg_16f448 [--n 14] [--strip 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# 日本語(CJK)フォント登録（豆腐□回避）
# ★カードは `family="monospace"` で描くので、**monospace ファミリの先頭に CJK 等幅を挿す**必要がある。
#   `font.family` だけ差し替えても monospace 指定が DejaVu Sans Mono に落ちて日本語が豆腐になる。
#   MigMix 2M は CJK 対応の等幅フォント（`fc-list :lang=ja` で確認）。
_MONO_CJK = ["/usr/share/fonts/truetype/migmix/migmix-2m-regular.ttf",
             "/usr/share/fonts/truetype/migmix/migmix-1m-regular.ttf"]
_SANS_CJK = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
for _group, _key in ((_MONO_CJK, "font.monospace"), (_SANS_CJK, "font.family")):
    for _fp in _group:
        if not Path(_fp).exists():
            continue
        try:
            font_manager.fontManager.addfont(_fp)
            name = font_manager.FontProperties(fname=_fp).get_name()
            if _key == "font.monospace":
                plt.rcParams["font.monospace"] = [name] + list(plt.rcParams["font.monospace"])
            else:
                plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
plt.rcParams["axes.unicode_minus"] = False

OK_C, NG_C = "#12a150", "#d43a3a"     # 正誤の色は report UI 用途（フレームには描画しない）


def hhmmss(sec: float) -> str:
    s = int(round(max(sec, 0)))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def pick(rows: list[dict], n: int) -> list[dict]:
    """回答形式を網羅するように、形式ごとにラウンドロビンで選ぶ（正解/不正解も混ぜる）。"""
    by = defaultdict(list)
    for r in rows:
        by[r.get("fmt", "?")].append(r)
    for v in by.values():
        # 誤答を先に見たいので「不正解 → 正解」の順に並べる（correct が無ければ順序維持）
        v.sort(key=lambda r: (r.get("correct") is True))
    out, i = [], 0
    keys = sorted(by)
    while len(out) < n and any(by[k] for k in keys):
        k = keys[i % len(keys)]
        if by[k]:
            out.append(by[k].pop(0))
        i += 1
    return out[:n]


def strip_image(paths: list[Path], times: list[float], k: int, h: int = 150):
    """フレーム列から k 枚を等間隔で抜き、横に連結した1枚と、その時刻ラベルを返す。"""
    if not paths:
        return None, []
    idx = np.linspace(0, len(paths) - 1, min(k, len(paths))).round().astype(int)
    idx = sorted(set(idx.tolist()))
    ims, labs = [], []
    for i in idx:
        try:
            im = Image.open(paths[i]).convert("RGB")
        except Exception:
            continue
        w = max(1, int(im.width * h / im.height))
        ims.append(im.resize((w, h)))
        labs.append(hhmmss(times[i]) if i < len(times) else "")
    if not ims:
        return None, []
    total = sum(im.width for im in ims) + 3 * (len(ims) - 1)
    canvas = Image.new("RGB", (total, h), (255, 255, 255))
    x, xs = 0, []
    for im in ims:
        canvas.paste(im, (x, 0))
        xs.append((x + im.width / 2) / total)
        x += im.width + 3
    return canvas, list(zip(xs, labs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--n", type=int, default=14, help="カード枚数")
    ap.add_argument("--strip", type=int, default=8, help="1カードに並べるフレーム数")
    ap.add_argument("--out", default="qa_cards.png")
    a = ap.parse_args()

    rows = json.loads((a.run_dir / "responses.json").read_text())
    # 採点済みなら correct を結合（誤答優先で選ぶため）
    merged = a.run_dir / "results_merged.csv"
    if merged.exists():
        import pandas as pd
        d = pd.read_csv(merged)
        key = {(str(r["dataset"]), str(r["qID"])): bool(r["correct"]) for _, r in d.iterrows()}
        for r in rows:
            r["correct"] = key.get((str(r["dataset"]), str(r["qID"])))

    # フレームパスを再構成（responses.json には保存していないので dataset 側から復元）
    from dataset_seg import frame_path, grid_for, sample_times
    meta = json.loads((a.run_dir / "meta.json").read_text())
    nf, size = int(meta["n_frames"]), int(meta["size"])
    # ★グリッドは meta.json を正とする。無い＝格子導入前の古い run なので当時の既定 1.0 に倒す。
    #   ここを推論側と食い違わせるとカードに**モデルが見ていないフレーム**が並ぶ。
    grid = float(meta["grid"]) if "grid" in meta else 1.0

    sel = pick(rows, a.n)
    n = len(sel)

    sysp = sel[0].get("system_prompt") or ""
    if not sysp:
        from prompts_seg import build_system_prompt
        sysp, _ = build_system_prompt(sel[0]["question"])
    # ★ヘッダ高さは**システムプロンプトの行数から実測して決める**。
    #   固定値にすると FO 定義全文（40行超）がはみ出して先頭カードのフィルムストリップに重なる。
    sys_lines = sum(max(1, len(textwrap.wrap(l, 190))) for l in sysp.split("\n"))
    head_h = 2.2 + 0.16 * (sys_lines + 10)      # 10 は説明文ぶん
    fig_h = 3.0 * n + head_h
    fig = plt.figure(figsize=(19, fig_h), dpi=100)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[head_h] + [3.0] * n, hspace=0.55)

    # ── 先頭: モデルへ渡した入力の説明（システムプロンプト全文を含む）──
    ax0 = fig.add_subplot(gs[0]); ax0.axis("off")
    head = (
        f"【モデルへの入力コンテキスト】 {meta['track']}  base={meta['base_model']}  "
        f"adapter={meta['adapter'] or '(なし=zero-shot)'}  dtype={meta['dtype']}\n"
        f"入力 = 各質問の [start,end] から **{nf} フレームを一様サンプル**（{size}px, "
        f"**{grid:g}秒グリッドにスナップ**"
        + ("＝本番クリップのキーフレーム間隔。実効 "
           f"{meta.get('n_frames_effective_mean', float(nf)):.1f} 枚" if grid >= 5 else "")
        + f"）。各フレームの直前に `[hh:mm:ss]` の絶対時刻テキストを置いて渡す\n"
        f"（公式の timestamp 焼き込みは使わない: start_time は推論時にも与えられるので OCR させる理由が無い）。\n"
        f"latency 平均 {meta['latency_mean']:.2f}s / 最大 {meta['latency_max']:.2f}s"
        f"（上限 {'15' if meta['track']=='SEGMENT' else '30'}s）  入力 {meta['input_tokens_mean']:.0f} tokens\n"
        f"time 回答は [start,end] にクランプ: {meta.get('clamp')}\n"
        f"\n─── システムプロンプト全文（形式 `{sel[0].get('fmt')}` の例）───\n"
        + "\n".join(textwrap.fill(l, 190) for l in sysp.split("\n"))
    )
    ax0.text(0, 1, head, va="top", ha="left", fontsize=7.4, family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc="#f2f5f8", ec="#9bb0c4"))

    for i, r in enumerate(sel):
        ax = fig.add_subplot(gs[i + 1]); ax.axis("off")
        ts = sample_times(r["start_time"], r["end_time"], nf, grid)
        paths = [frame_path(r["dataset"], r["videoID"], t, size) for t in ts]
        keep = [(t, p) for t, p in zip(ts, paths) if p.exists()]
        canvas, labs = strip_image([p for _, p in keep], [t for t, _ in keep], a.strip)

        if canvas is not None:
            iax = ax.inset_axes([0.0, 0.30, 0.62, 0.70])
            iax.imshow(canvas); iax.set_xticks([]); iax.set_yticks([])
            for xf, lb in labs:
                iax.text(xf, -0.06, lb, transform=iax.transAxes, ha="center", va="top",
                         fontsize=6.5, family="monospace", color="#333")

        ok = r.get("correct")
        mark = "—" if ok is None else ("正解 ✓" if ok else "誤り ✗")
        col = "#666" if ok is None else (OK_C if ok else NG_C)
        body = (
            f"[{r['fmt']}] / {r.get('group','')} / {r['dataset']}\n"
            f"clip {hhmmss(r['start_time'])}–{hhmmss(r['end_time'])} "
            f"({r['end_time']-r['start_time']:.0f}s, {r['n_frames']}フレーム)  "
            f"latency {r['latency']:.2f}s\n\n"
            f"Q: {textwrap.fill(r['question'], 88)}\n\n"
            f"PRED: {r['content']!r}\n"
            f"GT  : {r['answer']!r}"
        )
        if r.get("raw") and r["raw"] != r["content"]:
            body += f"\n(raw: {r['raw']!r}  ← クランプ前)"
        ax.text(0.63, 1.0, body, va="top", ha="left", fontsize=8, family="monospace",
                transform=ax.transAxes)
        ax.text(0.63, -0.02, mark, va="bottom", ha="left", fontsize=11, weight="bold",
                color=col, transform=ax.transAxes)

    out = a.run_dir / a.out
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({n} cards, {a.strip} frames/card)")


if __name__ == "__main__":
    main()
