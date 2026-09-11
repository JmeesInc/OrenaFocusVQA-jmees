"""instseg リーク対策の **arm 表**を作る（動画 → DUAL / CONTROL）.

検出器（expF40 / expF39）は `instseg_v004` fold1 の 104 動画で学習している。
その動画に重畳を見せると **検出器が GT を暗記した状態の重畳**を学習することになり、
テスト時（未知動画・AP50 0.55）とは別物の入力を学ぶ。よって:

  * 検出器が学習に使っていない動画 → **DUAL**（原画像 + 重畳）
  * 検出器が学習に使った動画       → **CONTROL**（原画像のみ）
  * 擬似 QA（instseg 注釈由来）    → 全部 CONTROL（生成元が必ず det-train 動画）

★`instseg_v004` は qa-aligned に切ってあるので、**qa fold0（VLM の val）26 動画には
  det-train 動画が 1 本も無い**。CV は重畳込みで leak-free に測れる。
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def stem(s) -> str:
    return os.path.splitext(str(s))[0]


def main() -> None:
    v = pd.read_csv(ROOT / "workspace/fold/v003/folds.csv")
    ins = pd.read_csv(ROOT / "workspace/fold/instseg_v004/folds.csv")
    ins["stem"] = ins.video.map(stem)
    det_train = set(ins[ins.fold == 1].stem)
    det_val = set(ins[ins.fold == 0].stem)
    v["stem"] = v.videoID.map(stem)
    v["det_status"] = v.stem.map(
        lambda s: "det_train" if s in det_train else ("det_val" if s in det_val else "no_annot"))
    v["arm"] = v.det_status.map(lambda s: "CONTROL" if s == "det_train" else "DUAL")

    out = v[["videoID", "dataset", "fold", "det_status", "arm"]].rename(
        columns={"fold": "qa_fold"})
    p = HERE / "overlay_arm_v001.csv"
    out.to_csv(p, index=False)

    qa = pd.read_csv(ROOT / "workspace/fold/qa_v004/qa_split.csv")
    qa["stem"] = qa.videoID.map(stem)
    qa = qa.merge(v[["stem", "arm", "det_status"]], on="stem", how="left")
    f = qa[qa.track == "FRAME"]
    tr, va = f[f.fold != 0], f[f.fold == 0]

    print(f"書き出し: {p}  ({len(out)} 動画)")
    print("\n== FRAME train (fold1-4) ==")
    print(pd.crosstab(tr.arm, tr.dataset, margins=True))
    print("\n== 動画数 ==")
    print(v[v.fold != 0].groupby(["arm", "dataset"]).size())
    print("\n== FRAME val (fold0) ==  ★det_train は 0 本であること")
    print(pd.crosstab(va.det_status, va.dataset, margins=True))
    assert (va.det_status == "det_train").sum() == 0, "val に det-train 動画が混じっている"
    assert len(tr[tr.arm == "DUAL"]) + len(tr[tr.arm == "CONTROL"]) == len(tr)


if __name__ == "__main__":
    main()
