"""SEGMENT/FRAME 学習用に **1秒グリッドのフレームを全部** 抽出する（貸しGPUへの転送用）.

## なぜ「必要な分だけ」でなく「1秒グリッド全部」か
入力構成をランダム化（96f@448 / 64f@560 / 32f@768 / 刻み2s …）すると必要な時刻集合が
構成ごとに変わる。実測（SEGMENT train 15,994問）:

    96f 195,747 / 64f 178,610 / 32f 146,618 / stride2 184,848
    4構成の和集合 209,884  ←→  **1秒グリッド全部 224,963**（+7% だけ）

**和集合とほぼ同じコストで「どんな構成でも自由」になる**ので、全部取るのが正しい。

## 解像度は 768px だけ取る
768px から 560/448 へは読み込み時に縮小できる（`probe_capacity.py` で検証済み）。
3解像度を個別抽出すると 29GB だが、768 だけなら **14.8GB** で済み、転送量が半分以下になる。

Usage:
  python extract_1s_grid.py --tracks SEGMENT FRAME --size 768 --workers 32
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from dataset_seg import _qa_index, extract_all, frame_path, load_qa_rows  # noqa: E402

log = logging.getLogger("extract1s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["SEGMENT", "FRAME"])
    ap.add_argument("--parts", nargs="+", default=["train", "val"])
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--version", default="v004")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    for n in ("httpx", "httpcore", "datasets", "huggingface_hub", "filelock", "fsspec"):
        logging.getLogger(n).setLevel(logging.WARNING)

    jobs: dict[Path, tuple] = {}
    for track in a.tracks:
        idx = _qa_index(track)
        for part in a.parts:
            for r in load_qa_rows(a.version, track, a.fold, part, None):
                got = idx.get((r["dataset"], r["qID"]))
                if got is None:
                    continue
                req, _ = got
                # ★1秒グリッド（両端含む）。FRAME は start==end なので1枚。
                for t in range(int(req.start_time), int(req.end_time) + 1):
                    p = frame_path(r["dataset"], req.videoID, float(t), a.size)
                    jobs.setdefault(p, (r["dataset"], req.videoID, float(t), p, a.size))
            log.info(f"{track}/{part}: 累計 {len(jobs):,d} 枚")

    todo = [j for j in jobs.values() if not j[3].exists()]
    log.info(f"必要 {len(jobs):,d} 枚 / 未抽出 {len(todo):,d} 枚 "
             f"（推定 {len(todo)*66/1e6:.1f} GB @ {a.size}px）")
    if a.dry_run:
        return
    extract_all(list(jobs.values()), workers=a.workers)


if __name__ == "__main__":
    main()
