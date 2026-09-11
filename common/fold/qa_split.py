"""問題単位 split（qa_split.csv）の読み出し — 学習/評価コードはここを経由する.

`splits.py` が **動画単位**の情報源であるのに対し、こちらは **qID 単位**。
fold は動画から継承済みなので、ここで train/val を切ってもリークしない。

    import qa_split
    rows = qa_split.load(fold=0, part="train", limit=8000)          # FRAME 8000問
    rows = qa_split.load(fold=0, part="train")                       # 全部（15,989問）
    rows = qa_split.load(fold=0, part="val", limit=1500)             # 評価用
    rows = qa_split.load(fold=0, part="train", tracks=["FRAME","SEGMENT"])  # 将来の混合用

★**一意キーは `uid`（"{dataset}:{qID}"）**。qID はデータセット間で衝突するので
索引キーに使わないこと（FRAME で20件の衝突を確認済み）。

`limit` は `order` の昇順で取るので、**N を増やすと必ず入れ子になる**
（n1000 ⊂ n4000 ⊂ n8000）。データ量スケーリングの比較が厳密になる。
"""
from __future__ import annotations

import csv
from pathlib import Path

FOLD_DIR = Path(__file__).resolve().parent

_INT = {"fold", "order", "ood", "clinical"}
_FLOAT = {"start_time", "end_time", "duration"}


def _path(version: str) -> Path:
    return FOLD_DIR / f"qa_{version}" / "qa_split.csv"


def load(fold: int = 0, part: str = "train", limit: int | None = None,
         tracks: list[str] | None = None, version: str = "v001") -> list[dict]:
    """fold の train/val を order 昇順で返す。

    part: "train"（fold != k）/ "val"（fold == k）/ "all"
    """
    if part not in ("train", "val", "all"):
        raise ValueError(part)
    tracks = tracks or ["FRAME"]
    out = []
    with _path(version).open() as f:
        for r in csv.DictReader(f):
            if r["track"] not in tracks:
                continue
            k = int(r["fold"])
            if part == "train" and k == fold:
                continue
            if part == "val" and k != fold:
                continue
            for c in _INT:
                r[c] = int(r[c])
            for c in _FLOAT:
                r[c] = float(r[c])
            out.append(r)
    out.sort(key=lambda r: (r["track"], r["order"]))
    return out[:limit] if limit else out


def summary(version: str = "v001") -> None:
    """fold × track の件数と、fold ごとの capability / format 分布を出す（設計検証用）。"""
    import collections
    rows = load(part="all", tracks=None, version=version) if False else []
    with _path(version).open() as f:
        rows = [r for r in csv.DictReader(f)]
    tracks = sorted({r["track"] for r in rows})
    folds = sorted({int(r["fold"]) for r in rows})
    for t in tracks:
        sub = [r for r in rows if r["track"] == t]
        print(f"\n===== {t} (計 {len(sub)}) =====")
        print("  fold件数: " + "  ".join(
            f"f{k}={sum(1 for r in sub if int(r['fold']) == k)}" for k in folds))
        for col in ("answer_format", "primary", "dataset"):
            keys = [k for k, _ in collections.Counter(r[col] for r in sub).most_common()]
            print(f"  --- {col} の fold 別割合(%) ---")
            hdr = "    " + f"{'key':32s}" + "".join(f"{'f'+str(k):>8s}" for k in folds)
            print(hdr)
            for key in keys:
                cells = []
                for k in folds:
                    s = [r for r in sub if int(r["fold"]) == k]
                    cells.append(100 * sum(1 for r in s if r[col] == key) / max(len(s), 1))
                print(f"    {key:32s}" + "".join(f"{c:8.1f}" for c in cells))


if __name__ == "__main__":
    summary()
