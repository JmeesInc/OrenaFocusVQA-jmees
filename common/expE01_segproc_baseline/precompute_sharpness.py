"""キャッシュ済みフレームの **鮮明度（Laplacian 分散）** を先に全部計算して保存する.

## なぜ先に計算するか
フレーム選択のたびに候補を毎回デコードすると、推論のたびに数十万枚を読むことになる。
**1回だけ計算して (dataset, video, 時刻) → 鮮明度 の表にしておけば、以後は参照だけで済む**。

## 指標: variance of Laplacian
グレースケールに Laplacian を当てた分散。ピントが合っている／動きブレが無いほど大きい。
内視鏡映像では **カメラ移動・煙・体液の付着**でブレるので、この指標が効くはず。

⚠️**仮説であって検証済みではない**: 鮮明なフレームが「静止していて何も起きていない瞬間」に
偏る可能性がある（情報のある瞬間は動いていることが多い）。効果は実測で確かめること。

Usage:
  python precompute_sharpness.py --size 768 --workers 32 --out sharpness_768.parquet
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "frames_cache"


def score_one(p: str) -> float:
    """Laplacian 分散。読めなければ NaN（呼び出し側で除外）。"""
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return float("nan")
    # ★縮小してから計算する: 768px のまま全画素で回すと遅く、
    #   しかも高周波ノイズを拾いすぎる。256px 程度で十分に順位が付く。
    if im.shape[1] > 256:
        h = int(im.shape[0] * 256 / im.shape[1])
        im = cv2.resize(im, (256, h), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(im, cv2.CV_64F).var())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = CACHE / str(a.size)
    out = Path(a.out) if a.out else HERE / f"sharpness_{a.size}.parquet"

    paths = [p for d in root.rglob("*") if d.is_dir() for p in d.glob("*.jpg")]
    print(f"対象 {len(paths):,} 枚 @ {a.size}px")
    done = {}
    if out.exists():   # 差分だけ計算（再実行が安い）
        old = pd.read_parquet(out)
        done = dict(zip(old.path, old.sharp))
        print(f"  既存 {len(done):,} 件を再利用")
    todo = [p for p in paths if str(p) not in done]
    print(f"  新規計算 {len(todo):,} 件")

    if todo:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, (p, s) in enumerate(zip(todo, ex.map(score_one, [str(x) for x in todo],
                                                        chunksize=256)), 1):
                done[str(p)] = s
                if i % 20000 == 0:
                    print(f"    {i:,}/{len(todo):,}")

    rows = []
    for p, s in done.items():
        pp = Path(p)
        rows.append({"path": p, "dataset": pp.parents[1].name, "video": pp.parent.name,
                     "ms": int(pp.stem), "sharp": s})
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    ok = df.sharp.notna()
    print(f"保存 {out}  ({len(df):,} 件 / 読めた {ok.sum():,})")
    print(f"  鮮明度 分位: p10={df.sharp[ok].quantile(.1):.1f} "
          f"p50={df.sharp[ok].median():.1f} p90={df.sharp[ok].quantile(.9):.1f}")


if __name__ == "__main__":
    main()
