"""SAM3 密トラック + GT 注釈 → 動画ごとの FO 存在タイムライン（擬似 SEGMENT QA の土台）.

## なぜタイムラインが要るか

FRAME 擬似 QA は「1枚の注釈」から答えを機械導出できたが、SEGMENT の質問は
**「いつ現れたか／いつ消えたか／同じ個体か／どの区間に何が居たか」**を問う。
これには *時間軸上の存在区間* が要る。

## 情報源と、それぞれの限界（★設計はこの限界に従う）

| 源 | 与えるもの | 限界 |
|---|---|---|
| survis-anno GT 注釈 | **その瞬間の完全なインスタンス集合**（アノテータは全 FO を塗る）| **疎**（median run 0.0s＝ほぼ孤立フレーム。密な区間は全体で20分だけ）|
| SAM3 トラック(expJ00) | GT シードから **前後 ±15s の毎フレーム存在＋個体 ID**（~6fps）| **シードされた個体しか追わない**。窓の途中で新規に入る物体は見えない。score ゲートで**早期に切れる**ことがある |

→ **完全性が保証されるのはシード時刻ちょうど**で、そこから離れるほど劣化する。
   よって擬似 SEGMENT の窓は**シード周辺に限定**し、窓端は必ずトラック被覆内に置く。

## 出力（`out/timeline_v1.json.gz`）

```
{video: {"dataset":..., "fps":..., "seeds":[frame,...],
         "tracks":[{"tid","label","seed_frame","t0","t1",
                    "pts":[[t, cx, cy, area], ...]}, ...]}}
```
- `pts` は ~6fps の観測点（rle 重心・面積）。quadrant は利用側で w,h と併せて判定
- **qa fold v003 fold0（VQA val）の動画は最初から除外**（リーク防止）
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE / "out"
SAM3 = ROOT / "workspace/expJ00_sam3_pseudo/results/pseudo_v001"
QA_FOLD_CSV = ROOT / "workspace/fold/v003/folds.csv"
FPS = {"heico": 25.0, "lapchole": 30.0}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("expK00.timeline")


def val_videos() -> set[str]:
    with open(QA_FOLD_CSV) as f:
        return {os.path.splitext(r["videoID"])[0] for r in csv.DictReader(f)
                if int(r["fold"]) == 0}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    excl = val_videos()
    out: dict = {}
    n_excluded = 0
    for ds in ("heico", "lapchole"):
        fps = FPS[ds]
        for p in sorted(glob.glob(str(SAM3 / ds / "*.jsonl.gz"))):
            stem = Path(p).name.replace(".jsonl.gz", "")
            if stem in excl:
                n_excluded += 1
                continue
            pts: dict = defaultdict(list)
            meta: dict = {}
            wh = None
            with gzip.open(p, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    t = r["frame"] / fps
                    for i in r["insts"]:
                        rle = i["rle"]
                        if wh is None:
                            wh = (int(rle["size"][1]), int(rle["size"][0]))  # (w,h)
                        try:
                            m = mask_util.decode(rle)
                        except Exception:
                            continue
                        ys, xs = np.nonzero(m)
                        if len(xs) == 0:
                            continue
                        key = (i["seed_frame"], i["obj_id"])
                        meta[key] = i["label"]
                        pts[key].append([round(t, 2), round(float(xs.mean()), 1),
                                         round(float(ys.mean()), 1), int(m.sum())])
            tracks = []
            for (seed, oid), arr in pts.items():
                arr.sort()
                tracks.append({"tid": f"{seed}_{oid}", "label": meta[(seed, oid)],
                               "seed_frame": seed, "seed_t": round(seed / fps, 2),
                               "t0": arr[0][0], "t1": arr[-1][0], "pts": arr})
            if not tracks:
                continue
            out[stem] = {"dataset": ds, "fps": fps,
                         "w": wh[0] if wh else None, "h": wh[1] if wh else None,
                         "seeds": sorted({t["seed_frame"] for t in tracks}),
                         "tracks": tracks}
            log.info(f"{stem[:38]:38s} tracks={len(tracks):4d} seeds={len(out[stem]['seeds']):3d}")
    dst = OUT / "timeline_v1.json.gz"
    with gzip.open(dst, "wt") as f:
        json.dump(out, f)
    n_tr = sum(len(v["tracks"]) for v in out.values())
    n_pt = sum(len(t["pts"]) for v in out.values() for t in v["tracks"])
    log.info(f"videos={len(out)}（qa fold0 除外 {n_excluded}）tracks={n_tr:,} points={n_pt:,} → {dst}")


if __name__ == "__main__":
    main()
