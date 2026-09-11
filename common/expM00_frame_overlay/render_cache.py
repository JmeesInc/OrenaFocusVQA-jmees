"""検出器の重畳画像を **事前生成**してキャッシュする（学習・CV・カード生成で共用）.

学習ループの中で M2F を回すと dataloader が詰まるうえ、CV とコンテナで描画条件が
ズレる余地が残る。**同じ PNG/JPEG を学習にも評価にも使う**のが一番安全。

    # FRAME val（CV 用, 26動画 4,008問 = 検出器が学習していない動画のみ）
    python render_cache.py --part val --variants r0 r1 r2 r3 --shard 0/3

    # FRAME train の DUAL arm（68動画 9,897問）
    python render_cache.py --part train --arm DUAL --variants r0

出力:
    cache/<variant>/<sha1(frame_key)>.jpg
    cache/<variant>/index.json   frame_key -> {file, note, n_hi, n_lo, labels, count_text}

★`file` が null の frame_key は **検出0件**（= 重畳を付けない）。
  「まだ処理していない」と区別できるようにわざとエントリを残す。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "workspace/expE01_segproc_baseline"))
sys.path.insert(0, str(ROOT / "reference/src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("expM00.render")

from dataset_seg import build_samples, grid_for  # noqa: E402
from overlay_render import (VARIANTS, M2F, MergedDetector, count_text,  # noqa: E402
                            render, resize_to)

F40 = ROOT / "workspace/expF00_fo_instseg/results/expF40_m2f_ps1sN_noGallstone_0828/fold0/best_model"
F39 = ROOT / "workspace/expF00_fo_instseg/results/expF39_m2f_clip_hires_strongaug_0828/fold0/best_model"


def frame_key(p) -> str:
    """ROOT 相対のフレームパス（マシンが変わっても同じキーになる）。"""
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="FRAME")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", default="val", choices=["train", "val"])
    ap.add_argument("--version", default="v004")
    ap.add_argument("--size", type=int, default=768, help="frames_cache の解像度")
    ap.add_argument("--out-width", type=int, default=768, help="描画（＝モデルが見る）幅")
    ap.add_argument("--variants", nargs="+", default=["r0"], choices=list(VARIANTS))
    ap.add_argument("--arm", default="", choices=["", "DUAL", "CONTROL"],
                    help="overlay_arm_v001.csv の arm で動画を絞る（train は DUAL のみ描く）")
    ap.add_argument("--videos-file", default="")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--conf-lo", type=float, default=0.25,
                    help="r3 の 2 段目しきい値。検出はここで拾って描画側で振り分ける")
    ap.add_argument("--allclass-ckpt", default=str(F40))
    ap.add_argument("--clip-ckpt", default=str(F39), help="空文字なら specialist を使わない")
    ap.add_argument("--cache-root", default=str(HERE / "cache"))
    ap.add_argument("--shard", default="", metavar="i/n")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    samples = build_samples(a.track, a.fold, a.part, n_frames=1, size=a.size,
                            limit=a.limit, version=a.version, grid=grid_for(a.track))
    if a.arm or a.videos_file:
        import pandas as pd
        keep: set[str] | None = None
        if a.arm:
            arms = pd.read_csv(HERE / "overlay_arm_v001.csv")
            keep = {v.rsplit(".", 1)[0].strip()
                    for v in arms[arms.arm == a.arm].videoID}
        if a.videos_file:
            vf = {x.strip().rsplit(".", 1)[0].strip()
                  for x in Path(a.videos_file).read_text().splitlines() if x.strip()}
            keep = vf if keep is None else (keep & vf)
        n0 = len(samples)
        samples = [s for s in samples if s.videoID.rsplit(".", 1)[0].strip() in keep]
        log.info("arm=%s videos-file=%s で %d → %d 問", a.arm or "-",
                 a.videos_file or "-", n0, len(samples))

    # フレーム単位に畳む（1フレームが複数問で共有される）
    frames = sorted({str(s.frame_paths[0]) for s in samples})
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        frames = [f for k, f in enumerate(frames) if k % n == i]
    log.info("%d 問 → ユニークフレーム %d 枚（shard %s）", len(samples), len(frames),
             a.shard or "-")

    spec = {}
    if a.clip_ckpt:
        spec["Clip"] = M2F(a.clip_ckpt, conf=a.conf_lo, height=768, width=1344)
    det = MergedDetector(M2F(a.allclass_ckpt, conf=a.conf_lo), spec,
                         conf=a.conf, conf_lo=a.conf_lo)

    root = Path(a.cache_root)
    idx: dict[str, dict] = {v: {} for v in a.variants}
    for v in a.variants:
        (root / v).mkdir(parents=True, exist_ok=True)

    t0, n_empty = time.perf_counter(), 0
    for i, fp in enumerate(frames, 1):
        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("読めない: %s", fp)
            continue
        img = resize_to(img, a.out_width)
        dets = det.predict(img)
        key = frame_key(fp)
        h = hashlib.sha1(key.encode()).hexdigest()[:20]
        hi = [d for d in dets if d["score"] >= a.conf]
        if not hi:
            n_empty += 1
        for v in a.variants:
            pil, meta = render(img, dets, variant=v, conf=a.conf, conf_lo=a.conf_lo)
            rec = {"file": None, "n_hi": meta["n_hi"], "n_lo": meta["n_lo"],
                   "labels": meta["labels"], "scores": meta["scores"],
                   "count_text": count_text(hi) if hi else None}
            if pil is not None:
                rel = f"{v}/{h}.jpg"
                pil.save(root / rel, quality=92, subsampling=0)
                rec["file"] = rel
                rec["note"] = meta["note"]
            idx[v][key] = rec
        if i % 200 == 0 or i == len(frames):
            el = time.perf_counter() - t0
            log.info("%d/%d  %.2f f/s  eta %.1f min  空 %d (%.1f%%)", i, len(frames),
                     i / el, (len(frames) - i) / (i / el) / 60, n_empty, 100 * n_empty / i)
            _flush(root, idx, a.shard, a.part)
    _flush(root, idx, a.shard, a.part)
    log.info("完了 frames=%d empty=%d (%.1f%%) → %s", len(frames), n_empty,
             100 * n_empty / max(len(frames), 1), root)
    return 0


def _flush(root: Path, idx: dict, shard: str, part: str) -> None:
    """**書き手ごとに必ず別ファイル**へ書く（`overlay_attach.load_index` が全部マージする）。

    ⚠️ 2026-09-01 の実害: 旧実装は `--shard` を付けたときだけサフィックスを付けており、
    val と train を同時に（どちらも shard なしで）回したら **両方が `index.json` に書いて
    片方が消えた**（画像は両方できているのに index だけ train のものになり、
    `attach` が「重畳キャッシュに無いフレーム」で停止）。
    → **part と shard の両方**をファイル名に入れる。
    """
    suf = (".shard" + shard.replace("/", "of")) if shard else ""
    for v, d in idx.items():
        (root / v / f"index.{part}{suf}.json").write_text(json.dumps(d, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
