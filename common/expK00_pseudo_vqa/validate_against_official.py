"""dump アノテーションの網羅性を公式 QA で検算する.

公式 lapchole FRAME train の質問のうち、**同一フレーム（±0.5s）に dump アノテーションが
存在する**ものについて、dump から導いた答えと公式の正解を突き合わせる。
一致率が低い family は擬似 QA の教師としてノイジー、という判定に使う。

2026-08-25 実測（dump 20260825 / 公式 5b3510c）:
  matched 317/5,748
  list(=list_all/combination) 59/86 = 0.686
  inst_count 26/37 = 0.703
  class_count 19/26 = 0.731
  class_div  17/19 = 0.895
  count diff (dump−公式): {-3:1, -2:1, -1:7, +1:3, +2:3, +3:2, +4:1}  # 両方向
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE = Path("/home/shunsuke/.cache/huggingface/hub/datasets--orena-dkfz--lapchole-focus-vqa/"
            "snapshots/5b3510cde3ba1135c56c4b4b25b50c4f948e235b/data")
DUMP = Path("/mnt/data/data4/input/survis-anno/focus-lapchole/focus-lapchole_20260825_001")
FPS = 30.0
CM = {"sponge": "Sponge", "clip": "Clip", "Specimen_Bag": "Specimen bag",
      "Silicon_Loop": "Silicone loop", "External_Drain": "External drain",
      "Needle": "Needle", "Gallstone": "Gallstone", "Specimen": "Specimen"}
INV_PLURAL = {"Clips": "Clip", "Sponges": "Sponge", "Specimens": "Specimen",
              "Specimen bags": "Specimen bag", "Silicone loops": "Silicone loop",
              "External drains": "External drain", "Needles": "Needle",
              "Gallstones": "Gallstone"}


def to_sec(t: str) -> int:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def main() -> None:
    import argparse
    global DUMP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default=str(DUMP))
    DUMP = Path(ap.parse_args().dump)
    print(f"dump = {DUMP}")
    df = pd.read_parquet(BASE / "frame/train.parquet")
    d = json.load(open(DUMP / "coco/instances.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    img = {im["id"]: (im["video"].rsplit(".", 1)[0], im["frame_number"]) for im in d["images"]}
    per: dict = defaultdict(list)
    for a in d["annotations"]:
        per[img[a["image_id"]]].append(CM[cats[a["category_id"]]])

    rows = []
    for r in df.itertuples():
        stem = str(r.video).rsplit(".", 1)[0]
        fn = int(round(to_sec(r.timestamp_start) * FPS))
        key = next(((stem, fn + off) for off in range(-15, 16) if (stem, fn + off) in per), None)
        if key:
            rows.append((r.question, r.answer, per[key]))
    print(f"matched {len(rows)}/{len(df)}")

    agree: Counter = Counter()
    tot: Counter = Counter()
    diffs = []
    for q, ans, classes in rows:
        uniq = sorted(set(classes))
        if q.startswith("How many different foreign object instances"):
            fam, mine = "inst_count", str(len(classes))
        elif q.startswith("How many different foreign object classes"):
            fam, mine = "class_div", str(len(uniq))
        elif q.startswith("How many "):
            fam = "class_count"
            c = INV_PLURAL.get(q.split("How many ")[1].split(" appear")[0])
            if c is None:
                continue
            mine = str(sum(1 for x in classes if x == c))
        elif q.startswith("List all") or q.startswith("Which combination"):
            fam, mine = "list", ", ".join(uniq)
        else:
            continue
        tot[fam] += 1
        if str(mine).lower() == str(ans).lower():
            agree[fam] += 1
        elif fam in ("inst_count", "class_count"):
            diffs.append(int(mine) - int(ans))
    for f in tot:
        print(f"{f}: {agree[f]}/{tot[f]} = {agree[f] / tot[f]:.3f}")
    print("count diff (dump-official):", dict(sorted(Counter(diffs).items())))


if __name__ == "__main__":
    main()
