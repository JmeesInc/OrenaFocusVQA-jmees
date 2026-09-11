"""LOPO（leave-one-procedure-out）の folds.csv を生成する.

## なぜ必要か（2026-08-17、SEGMENT の LB 内訳を見て作成）

公式 SCORE は **5 capability group × in-/out-of-distribution の10バケット非加重平均**。
つまり **半分が OOD（学習に無い術式）** で決まる。ところが既存の fold（v001〜v004 と
その上の qa_v001〜v005）は **すべて in-distribution 5-fold CV** で、学習データには
`ood=1` が1件も無い。→ **ローカル CV はスコアの半分を構造的に測れていない。**

実際 2026-08-16 の SEGMENT 提出（LB 0.5692 / 3位）を分解すると:

    ID 平均  0.6275 ← 上位5チーム中 **1位**
    OOD平均  0.5109 ← 上位5チーム中 最下位（ただし n=10 の agg_ood が支配。
                       それを除くと 0.5386 で 3位。1.05σ で有意ではない）

有意ではないが **測る手段が無い**こと自体が問題なので、術式跨ぎの汎化を測れる
split を用意する。`splits.py` には既に `lopo_splits()` があるが、学習/推論側が使う
`qa_{version}/qa_split.csv` が LOPO 構成で存在しなかった。ここを埋める。

## 割り当て

`load_qa_rows()` の規約は「part='val' は fold==指定fold、'train' は fold!=指定fold」。
2術式なので fold は2つで足りる:

    fold 0 = lapchole(胆嚢摘出)  → `--fold 0` で train=heico(大腸) / val=lapchole
    fold 1 = heico(大腸)         → `--fold 1` で train=lapchole  / val=heico

`splits.lopo_splits()` の `lopo_train_heico` が fold 0、`lopo_train_lapchole` が fold 1 に対応。

## 重複動画について

lapchole には 4 ペアの重複動画がある（v004 で同一 fold に固定済み）。LOPO では
lapchole 全体が必ず片側に入るので **train/val を跨ぐリークは構造的に起きない**。

Usage:
    PYTHONPATH=reference/src .venv/bin/python workspace/fold/generate_folds_lopo.py
出力: workspace/fold/lopo_v001/folds.csv  列: videoID, dataset, fold
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

FOLD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLD_DIR))

import splits as splits_mod  # noqa: E402

# 術式 → fold。`--fold k` の val がその術式になる。
DATASET_FOLD = {"lapchole": 0, "heico": 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="lopo_v001", help="出力先 workspace/fold/<version>/")
    ap.add_argument("--source", default="v004",
                    help="動画リストの出所となる既存 folds.csv（fold 列は使わない）")
    args = ap.parse_args()

    src = splits_mod._load(args.source)
    rows = [{"videoID": r["videoID"], "dataset": r["dataset"],
             "fold": DATASET_FOLD[r["dataset"]]} for r in src]

    unknown = {r["dataset"] for r in src} - set(DATASET_FOLD)
    if unknown:
        raise ValueError(f"未知の dataset: {unknown}（DATASET_FOLD に追加が必要）")

    out_dir = FOLD_DIR / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "folds.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["videoID", "dataset", "fold"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["dataset"], r["videoID"])))

    print(f"wrote {out}  ({len(rows)} videos, source={args.source})")
    for ds, k in sorted(DATASET_FOLD.items(), key=lambda kv: kv[1]):
        n = sum(1 for r in rows if r["dataset"] == ds)
        other = sum(1 for r in rows if r["dataset"] != ds)
        print(f"  fold {k}: val={ds} {n}本 / train=残り {other}本")

    # 検算: splits.lopo_splits と一致するか
    lopo = splits_mod.lopo_splits(args.version)
    for s, k in ((lopo[0], 0), (lopo[1], 1)):
        val = frozenset(r["videoID"] for r in rows if r["fold"] == k)
        assert s.val_videos == val, f"{s.name} と fold {k} の val が不一致"
    print("  ✅ splits.lopo_splits() と一致")


if __name__ == "__main__":
    main()
