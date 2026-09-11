"""PROCEDURE の clean な評価 run（val 全量 N=1960）だけを集めてバケット表を出す.

## なぜ必要か
`results/` には **N=984 の旧 run（`--limit 1000`）** と **汚染 run（expD11 流用）** が混ざっており、
そのまま並べると誤読する（実測: expD11 流用 32f は val 25本中21本＝**問題単位 92.2%** が学習動画、
event_understanding は **19問中16問**で 0.8421）。**N を揃えた行だけを出す**のがこのスクリプトの役目。

Usage:
  .venv/bin/python workspace/expE01_segproc_baseline/proc_table.py [--n 1960]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2] / "reference/src"))

GROUPS = ["object_recognition", "temporal_grounding", "aggregation",
          "event_understanding", "complex_reasoning"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1960, help="この問題数の run だけを出す")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    a = ap.parse_args()

    from focus.evaluation import Evaluator
    ev = Evaluator()

    out = []
    for d in sorted(a.results.glob("*")):
        f, m = d / "results_merged.csv", d / "meta.json"
        if not (f.exists() and m.exists()):
            continue
        meta = json.loads(m.read_text())
        if meta.get("track") != "PROCEDURE":
            continue
        df = pd.read_csv(f)
        if len(df) != a.n:
            continue
        score, buckets = ev.pre_evaluation_score(df)   # ★タプルで返る
        acc = {r["group"]: r["accuracy"] for _, r in buckets.iterrows()}
        out.append({
            "run": d.name,
            "n_frames": meta.get("n_frames"),
            "grid": meta.get("grid"),
            "SCORE": round(float(score), 4),
            **{g: round(float(acc.get(g, float("nan"))), 4) for g in GROUPS},
            "latency": round(float(meta.get("latency_mean", 0)), 2),
        })

    if not out:
        raise SystemExit(f"N={a.n} の PROCEDURE run が見つからない")
    t = pd.DataFrame(out).sort_values("SCORE").reset_index(drop=True)
    print(f"# PROCEDURE — clean 比較（N={a.n} に揃えた run のみ / fold v003 = qa_v004 fold0 val）\n")
    print(t.to_markdown(index=False))


if __name__ == "__main__":
    main()
