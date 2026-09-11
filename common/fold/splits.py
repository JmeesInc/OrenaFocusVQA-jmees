"""fold 割当（folds.csv）から学習/検証の split を返す単一の情報源.

全 LoRA / 学習コードはここを経由して split を取得する（各自で切り直さない）。
提供する split は2系統:

  1. **in-distribution 5-fold CV**（`cv_folds`）: 動画単位 GroupKFold。
     heico(colorectal) と lapchole(cholecystectomy) を各 fold に混在させた通常 CV。
  2. **LOPO（leave-one-procedure-out, OOD）**（`lopo_splits`）: 術式跨ぎ汎化の推定。
     - OOD-A: train=heico → val=lapchole
     - OOD-B: train=lapchole → val=heico
     テストは未知術式を含み、プロンプト改善が heico→lapchole に転移しないと判明済み
     （expB02）。in-dist CV だけだと未知術式性能を過大評価するため必ず併用する。

video 単位で train/val の videoID 集合を返す。Request/Reference のフィルタは
`filter_by_videos()` を使う。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

FOLD_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Split:
    name: str              # 例 "cv_fold0", "lopo_train_heico"
    train_videos: frozenset[str]
    val_videos: frozenset[str]
    kind: str              # "cv" | "lopo"

    def __repr__(self) -> str:
        return (f"Split({self.name}, kind={self.kind}, "
                f"train={len(self.train_videos)}v, val={len(self.val_videos)}v)")


def _load(version: str) -> list[dict]:
    """folds.csv を読む。列: videoID, dataset, fold。CRLF を除去。"""
    path = FOLD_DIR / version / "folds.csv"
    rows = []
    for r in csv.DictReader(path.open()):
        rows.append({k: v.replace("\r", "").strip() for k, v in r.items()})
    return rows


def n_folds(version: str = "v001") -> int:
    return len({int(r["fold"]) for r in _load(version)})


def cv_folds(version: str = "v001") -> list[Split]:
    """in-distribution GroupKFold の全 fold を返す（val=fold k, train=残り）。"""
    rows = _load(version)
    folds = sorted({int(r["fold"]) for r in rows})
    out = []
    for k in folds:
        val = frozenset(r["videoID"] for r in rows if int(r["fold"]) == k)
        train = frozenset(r["videoID"] for r in rows if int(r["fold"]) != k)
        out.append(Split(name=f"cv_fold{k}", train_videos=train, val_videos=val, kind="cv"))
    return out


def cv_fold(k: int, version: str = "v001") -> Split:
    return cv_folds(version)[k]


def lopo_splits(version: str = "v001") -> list[Split]:
    """leave-one-procedure-out（OOD）。train=一方の術式全体 → val=他方の術式全体。"""
    rows = _load(version)
    heico = frozenset(r["videoID"] for r in rows if r["dataset"] == "heico")
    lap = frozenset(r["videoID"] for r in rows if r["dataset"] == "lapchole")
    return [
        Split(name="lopo_train_heico", train_videos=heico, val_videos=lap, kind="lopo"),
        Split(name="lopo_train_lapchole", train_videos=lap, val_videos=heico, kind="lopo"),
    ]


def video_dataset_map(version: str = "v001") -> dict[str, str]:
    return {r["videoID"]: r["dataset"] for r in _load(version)}


def filter_by_videos(requests, references, videos: frozenset[str]):
    """(requests, references) を videoID 集合で絞り込む。要素の順序は保持。"""
    idx = [i for i, req in enumerate(requests) if req.videoID in videos]
    return [requests[i] for i in idx], [references[i] for i in idx]


if __name__ == "__main__":
    # 動作確認 & 分布サマリ
    for s in cv_folds():
        print(s)
    for s in lopo_splits():
        print(s)
