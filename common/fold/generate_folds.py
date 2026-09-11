"""全実験共通の fold 割り当てを生成する（動画単位 GroupKFold）.

データの性質: 質問は動画ごとに割り当てられ、同一動画の質問が train/val に分かれると
重度のリーク（同じシーン・同じ異物を見て答える質問が多数）。→ **動画単位**で分割する。
術式（heico=大腸 / lapchole=胆摘）で層化し、両者が各 fold に均等に入るようにする。

Usage:
    PYTHONPATH=reference/src .venv/bin/python workspace/fold/generate_folds.py --version v001 --n-folds 5
出力: workspace/fold/<version>/folds.csv  列: videoID, dataset, fold
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from focus import FocusConfig, FocusDataset, set_config
from focus.enums import DatasetSplit, Track

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v001")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_config(FocusConfig(root_dir=str(ROOT / "data/focus")))

    # 全動画IDを取得（全トラック横断・両データセット）
    vids: dict[str, str] = {}  # videoID -> dataset
    for ds in ["heico", "lapchole"]:
        for track in Track:
            for vid in FocusDataset(ds, DatasetSplit.ALL, track).video_ids():
                vids[vid] = ds

    # 術式で層化して round-robin 割当（乱数はseedで固定した順序）
    import random
    rng = random.Random(args.seed)
    rows = []
    for ds in ["heico", "lapchole"]:
        ds_vids = sorted(v for v, d in vids.items() if d == ds)
        rng.shuffle(ds_vids)
        for i, vid in enumerate(ds_vids):
            rows.append({"videoID": vid, "dataset": ds, "fold": i % args.n_folds})

    out_dir = Path(__file__).parent / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "folds.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["videoID", "dataset", "fold"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["dataset"], r["fold"], r["videoID"])))

    # 分布サマリ
    print(f"wrote {out}  ({len(rows)} videos)")
    for ds in ["heico", "lapchole"]:
        dist = [sum(1 for r in rows if r["dataset"] == ds and r["fold"] == k) for k in range(args.n_folds)]
        print(f"  {ds}: {sum(dist)} videos, per-fold {dist}")


if __name__ == "__main__":
    main()
