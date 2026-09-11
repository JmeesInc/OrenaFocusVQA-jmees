"""稀クラス（陽性動画が極端に少ないクラス）を考慮した fold 再割当を生成する.

背景（expC00, 2026-07-23）:
    FO 分類器の 5-fold CV で **Gallstone は陽性 6動画/63フレームしか無く**、v001 の割当では
    fold0 に 4動画が集中し fold1/fold4 は陽性ゼロ（= AP 評価不能）だった。
    結果、Gallstone の fold 別 AP は 0.337 / -- / 1.000 / 0.026 / -- と無意味な値になり、
    macro-AP の fold 間 std を 0.022 → 0.049 に倍化させていた（＝設定間比較の検出力を潰す）。

方針:
    「稀クラスだけ fold 数を減らす」= 陽性動画を **少数の fold に 2本ずつまとめる**。
    5-fold のうち例えば fold0/1/2 の 3つにだけ Gallstone 陽性を置く（各2動画）。
      - 評価: 陽性ゼロの fold が消え、1 fold あたりの陽性動画が 1→2 に増えて AP が安定する
      - 学習: どの fold を val にしても train に 4〜6 動画が残る（v001 は val=fold0 のとき train が 2動画）
    他クラスへの影響は動かす動画数（6/130）だけに限定し、**fold サイズは donor との交換で厳密に保つ**。

Usage:
    .venv-dl1/bin/python workspace/fold/rebalance_rare.py \
        --base v001 --out v002 \
        --parquet workspace/expC00_fo_classifier/dataset/fo_v001/labels.parquet \
        --rare Gallstone --rare-folds 3
出力: workspace/fold/<out>/folds.csv（列は base と同じ: videoID, dataset, fold）
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pandas as pd

FOLD_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v001", help="元にする fold バージョン")
    ap.add_argument("--out", default="v002", help="出力バージョン")
    ap.add_argument("--parquet", required=True, help="陽性動画を判定する labels.parquet")
    ap.add_argument("--rare", default="Gallstone", help="稀クラス名（present_<name> 列）")
    ap.add_argument("--rare-folds", type=int, default=3, help="稀クラス陽性を集約する fold 数")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base_csv = FOLD_DIR / args.base / "folds.csv"
    rows = list(csv.DictReader(base_csv.open()))
    fold_of = {r["videoID"]: int(r["fold"]) for r in rows}
    ds_of = {r["videoID"]: r["dataset"] for r in rows}

    df = pd.read_parquet(args.parquet)
    col = f"present_{args.rare}"
    pos = df[df[col] == 1].groupby("video").size().sort_values(ascending=False)
    rare_vids = list(pos.index)
    print(f"[{args.rare}] 陽性 {len(rare_vids)} 動画 / {int(pos.sum())} フレーム")
    print("  現行(v001)分布:", {f: sum(1 for v in rare_vids if fold_of[v] == f) for f in range(args.n_folds)})

    # 1) 陽性動画を rare-folds 個の fold へ「フレーム数が均等になるよう」貪欲割当
    target: dict[str, int] = {}
    load = {f: 0 for f in range(args.rare_folds)}
    for v in rare_vids:  # フレーム数降順
        f = min(load, key=lambda k: (load[k], k))
        target[v] = f
        load[f] += int(pos[v])
    print(f"  新割当(fold別フレーム数): {load}")

    # 2) 移動が必要な動画は、移動先 fold の「稀クラス非陽性・同一dataset」動画と交換（fold サイズ保存）
    rare_set = set(rare_vids)
    moves: list[tuple[str, int, int, str]] = []
    for v in rare_vids:
        src, dst = fold_of[v], target[v]
        if src == dst:
            continue
        donors = [u for u in fold_of
                  if fold_of[u] == dst and u not in rare_set and ds_of[u] == ds_of[v]]
        if not donors:
            raise RuntimeError(f"donor が見つからない: {v} {src}->{dst}")
        d = rng.choice(donors)
        fold_of[v], fold_of[d] = dst, src
        moves.append((v, src, dst, d))
    print(f"  交換 {len(moves)} 件:")
    for v, s, d2, d in moves:
        print(f"    {v}  f{s}->f{d2}   (交換相手 {d}  f{d2}->f{s})")

    # 3) 検証
    print("\n検証:")
    sizes = {f: sum(1 for u in fold_of if fold_of[u] == f) for f in range(args.n_folds)}
    print("  fold別動画数:", sizes)
    for ds in sorted(set(ds_of.values())):
        print(f"  {ds}:", {f: sum(1 for u in fold_of if fold_of[u] == f and ds_of[u] == ds)
                           for f in range(args.n_folds)})
    newdist = {f: [v for v in rare_vids if fold_of[v] == f] for f in range(args.n_folds)}
    for f in range(args.n_folds):
        n = len(newdist[f])
        frames = int(sum(pos[v] for v in newdist[f]))
        print(f"  fold{f}: {args.rare} 陽性 {n}動画 / {frames}フレーム"
              + ("   ← val時に評価可能" if n else "   (陽性なし=train専用)"))

    out_dir = FOLD_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "folds.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["videoID", "dataset", "fold"])
        w.writeheader()
        w.writerows(sorted(({"videoID": v, "dataset": ds_of[v], "fold": fold_of[v]} for v in fold_of),
                           key=lambda r: (r["dataset"], r["fold"], r["videoID"])))
    print(f"\nwrote {out} ({len(fold_of)} videos)")


if __name__ == "__main__":
    main()
