"""問題単位（qID 単位）の split を実体化する — FRAME を起点に、後で SEGMENT/PROCEDURE を足せる形で.

## なぜ必要か
既存の `folds.csv` は **動画単位**（videoID → fold）しかない。学習サブセットは
`build_samples(videos, limit=N, shuffle_seed=42)` という**暗黙の定義**で、
- 中身（capability/format の分布）を検査できない
- n1000 / n4000 / n8000 が本当に入れ子になっているか確認できない
- **他トラック（SEGMENT/PROCEDURE）由来のフレームを混ぜたときに管理できない**
という問題がある。qID 単位で materialize しておけば全部解決する。

## 出力: `workspace/fold/qa_{version}/qa_split.csv`
| 列 | 意味 |
|---|---|
| qID | 質問ID（一意キー）|
| videoID / dataset | 動画とデータセット（heico / lapchole）|
| fold | 動画単位 GroupKFold の fold（**動画から継承 = リーク無し**）|
| track | FRAME / SEGMENT / PROCEDURE（由来トラック）|
| primary / answer_format | capability leaf と回答形式（層別・分析用）|
| start_time / end_time / duration | 秒。SEGMENT/PROCEDURE のフレーム化可否の判断に使う |
| order | **shuffle 後の順位**。`order < N` で取れば n=N のサブセットになり、N を増やすと自動的に入れ子になる |

## 使い方
    import qa_split
    rows = qa_split.load("v001", track="FRAME", fold=0, part="train", limit=8000)

## 注意
- **fold は動画から継承する**こと。qID 単位で切り直すと同一動画が train/val に跨って**リーク**する
- order は dataset を混ぜた上で seed 固定シャッフル。両術式・全形式が満遍なく入る
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

FOLD_DIR = Path(__file__).resolve().parent
ROOT = FOLD_DIR.parents[1]
sys.path.insert(0, str(FOLD_DIR))
sys.path.insert(0, str(ROOT / "reference/src"))

from focus import FocusConfig, FocusDataset, set_config  # noqa: E402
from focus.enums import DatasetSplit, Track  # noqa: E402
import splits as splits_mod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("qa_split")
for noisy in ("httpx", "httpcore", "datasets", "huggingface_hub", "filelock", "fsspec"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ★qID は **データセット間で衝突する**（FRAME で20件確認: heico と lapchole に同じ qID）。
#   一意キーは `uid = "{dataset}:{qID}"`。qID 単体で索引を作ると片方が黙って上書きされる。
COLUMNS = ["uid", "qID", "videoID", "dataset", "fold", "track", "primary", "answer_format",
           "start_time", "end_time", "duration", "ood", "clinical", "order"]


def build(version: str, fold_version: str, tracks: list[str], seed: int) -> list[dict]:
    set_config(FocusConfig(root_dir=str(ROOT / "data/focus")))
    vid2fold = {r["videoID"]: int(r["fold"]) for r in splits_mod._load(fold_version)}
    vid2ds = splits_mod.video_dataset_map(fold_version)

    rows: list[dict] = []
    for tname in tracks:
        trk = Track[tname]
        for ds in ("heico", "lapchole"):
            d = FocusDataset(ds, DatasetSplit.ALL, trk)
            for req, ref in zip(d.requests, d.references):
                if req.videoID not in vid2fold:
                    continue  # folds.csv に無い動画（未使用）はスキップ
                dsn = vid2ds.get(req.videoID, ds)
                rows.append({
                    "uid": f"{dsn}:{req.qID}",
                    "qID": req.qID,
                    "videoID": req.videoID,
                    "dataset": dsn,
                    "fold": vid2fold[req.videoID],   # ★動画から継承（リーク防止）
                    "track": tname,
                    "primary": getattr(ref.primary, "name", str(ref.primary)),
                    # ★`ref.format` は AnswerFormat の**インスタンス**なので str() すると
                    #   `<...FOClass object at 0x...>` になる。公式 evaluator は生の文字列
                    #   `ref._format`（"fo_class" 等）を results_judge.csv に入れているので
                    #   突き合わせ可能なようにそれを使う。
                    "answer_format": ref._format,
                    "start_time": f"{req.start_time:.3f}",
                    "end_time": f"{req.end_time:.3f}",
                    "duration": f"{req.end_time - req.start_time:.3f}",
                    "ood": int(bool(getattr(ref, "ood", False))),
                    "clinical": int(bool(getattr(ref, "clinical", False))),
                })

    # track ごとに seed 固定シャッフルして order を振る。
    # → `order < N` で n=N のサブセットが取れ、N を増やすと必ず入れ子になる。
    for tname in tracks:
        sub = [r for r in rows if r["track"] == tname]
        random.Random(seed).shuffle(sub)
        for i, r in enumerate(sub):
            r["order"] = i
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v001", help="出力先 qa_{version}/")
    ap.add_argument("--fold-version", default="v001", help="動画単位 folds.csv のバージョン")
    ap.add_argument("--tracks", nargs="+", default=["FRAME"],
                    help="materialize するトラック（既定は FRAME のみ）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = build(args.version, args.fold_version, args.tracks, args.seed)
    out_dir = FOLD_DIR / f"qa_{args.version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "qa_split.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    log.info(f"wrote {len(rows)} rows -> {out}")

    # 検算: fold ごとの件数と、動画が fold を跨いでいないこと
    import collections
    per = collections.Counter((r["track"], r["fold"]) for r in rows)
    for t in args.tracks:
        tot = sum(v for (tt, _), v in per.items() if tt == t)
        line = "  ".join(f"fold{k}={per[(t, k)]}" for k in sorted({f for (tt, f) in per if tt == t}))
        log.info(f"  {t}: 計 {tot}  {line}")
    uids = [r["uid"] for r in rows]
    log.info(f"  uid ユニーク性: {len(set(uids))}/{len(uids)}"
             f"{'' if len(set(uids))==len(uids) else '  ★重複あり'}")
    qs = [r["qID"] for r in rows]
    log.info(f"  （参考）qID 単体のユニーク性: {len(set(qs))}/{len(qs)} ← 一意でないので索引キーに使わない")
    vf = collections.defaultdict(set)
    for r in rows:
        vf[r["videoID"]].add(r["fold"])
    bad = [v for v, fs in vf.items() if len(fs) > 1]
    log.info(f"  fold を跨ぐ動画: {len(bad)} 件（0 でなければリーク）")


if __name__ == "__main__":
    main()
