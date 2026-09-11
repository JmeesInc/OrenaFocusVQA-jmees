r"""質問文テンプレート → capability group の対応表を作って `resources/group_templates.json` に書く.

## なぜテンプレート表なのか
`focus.Request` には **capability group も answer format も入っていない**（qID / videoID /
start_time / end_time / procedure_type / question のみ）。一方 SEGMENT の最良構成は
**group ごとに入力を変える振り分け**で、これは group が分からないと実行できない。

QA のテンプレは生成器由来で規則的なので、`hh:mm:ss` と数値と FO 名を伏せると
**829 種のテンプレに畳める**（SEGMENT 20,000 問）。テンプレ→最頻 group の表で引く。

## 検証（fold v003 fold0 val, N=1958, judge 込み）
val 動画を除いた 18,042 問だけで表を作って測った:

| 振り分け | SCORE |
|---|---|
| 64f@560 単独 | 0.7113 |
| 32f@768 単独 | 0.7197 |
| オラクル group（GT を使う。実行不可能）| 0.7345 |
| **テンプレート推定 group** | **0.7345** |

GT group との一致率 0.978 / 未知テンプレ 3.1%。誤振り分けした分は両 run で正誤が
一致していたためバケット精度が1つも動かず、**オラクルと同点**になった。

## 提出用の表は全データから作る
上の数値は「val を見ていない」ことを示すための対照。**実提出では test 動画が別物なので
リークは存在せず、全 50,000 問（FRAME/SEGMENT/PROCEDURE）から作った方が
未知テンプレ率が下がる**。既定はこちら。

Usage:
  .venv/bin/python submit/v005_segment_hybrid/build_router_table.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]

from router import GROUP_A, template_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-csv", default=str(ROOT / "workspace/expE00_segproc_eda/all_qa.csv"))
    ap.add_argument("--tracks", nargs="+", default=["SEGMENT", "FRAME", "PROCEDURE"],
                    help="表に含めるトラック。SEGMENT を先頭に置くこと（衝突時に優先される）")
    ap.add_argument("--exclude-uids", default="",
                    help="この CSV の uid を除外する（val リーク検証用）")
    ap.add_argument("--out", default=str(HERE / "resources/group_templates.json"))
    a = ap.parse_args()

    qa = pd.read_csv(a.qa_csv)
    if a.exclude_uids:
        drop = set(pd.read_csv(a.exclude_uids)["uid"])
        qa = qa[~qa.uid.isin(drop)]

    # ★SEGMENT を最後に上書きさせる。同一テンプレが複数トラックに出るとき、
    #   SEGMENT の割り当てが正になる（この表は SEGMENT でしか使わない）。
    order = [t for t in a.tracks if t != "SEGMENT"] + ["SEGMENT"]
    table: dict[str, str] = {}
    for track in order:
        sub = qa[qa.track == track]
        if sub.empty:
            continue
        counts: dict[str, Counter] = {}
        for q, g in zip(sub.question, sub.group):
            counts.setdefault(template_key(q), Counter())[g] += 1
        got = {k: c.most_common(1)[0][0] for k, c in counts.items()}
        table.update(got)
        print(f"{track:10s}: {len(sub):6d} 問 → {len(got):4d} テンプレ")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(table, ensure_ascii=False, indent=1, sort_keys=True))
    n_a = sum(1 for v in table.values() if v in GROUP_A)
    print(f"\nwrote {a.out}  {len(table)} テンプレ"
          f"（64f@560 側 {n_a} / 32f@768 側 {len(table)-n_a}）")


if __name__ == "__main__":
    main()
