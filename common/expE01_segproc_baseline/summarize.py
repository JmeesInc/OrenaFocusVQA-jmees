"""全 run の SCORE とバケット別内訳を1枚の表に集約する.

`eval_summary.md` を1本ずつ読むのは辛いので、`results_merged.csv`（公式 judge 済み）から
直接集計し直す。**リーク（expD11 の学習動画が val に入っている）も同時に判定して印を付ける**。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]  # workspace/expE01_.../ から見て 2つ上がリポジトリ直下
ROOT = ROOT.parent if ROOT.name == "workspace" else ROOT
RES = HERE / "results"

# expD11(FRAME LoRA) は `splits.cv_folds()` の既定 = fold **v001** の fold0 以外で学習している。
# その重みを使った run は、val に v001 fold≠0 の動画が入っていれば汚染。
_v1 = pd.read_csv(ROOT / "workspace/fold/v001/folds.csv")
D11_HELDOUT = set(_v1[_v1.fold == 0].videoID)

GROUPS = ["object_recognition", "temporal_grounding", "aggregation",
          "event_understanding", "complex_reasoning"]


def load(d: Path):
    f = d / "results_merged.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f)
    meta = json.loads((d / "meta.json").read_text())
    resp = json.loads((d / "responses.json").read_text())
    vid = {(r["dataset"], str(r["qID"])): r["videoID"] for r in resp}
    m["videoID"] = [vid.get((r.dataset, str(r.qID)), "") for r in m.itertuples()]
    m["clean"] = m["videoID"].isin(D11_HELDOUT)
    m["grp"] = m["primary"].str.lower().map(_to_group)
    return m, meta


def _to_group(p: str) -> str:
    t = {"object_identification": "object_recognition", "instance_matching": "object_recognition",
         "object_attributes": "object_recognition",
         "spatial_localization_camera": "object_recognition",
         "spatial_localization_situs": "object_recognition",
         "temporal_localization": "temporal_grounding", "duration_estimation": "temporal_grounding",
         "object_aggregation": "aggregation", "event_aggregation": "aggregation",
         "fo_interaction_recognition": "event_understanding",
         "fo_usage_purpose": "event_understanding", "temporal_ordering": "event_understanding",
         "functional_reasoning": "complex_reasoning",
         "causal_consequence_reasoning": "complex_reasoning",
         "multi_step_reasoning": "complex_reasoning"}
    return t.get(str(p).lower(), str(p).lower())


def score(m: pd.DataFrame) -> float:
    """公式 SCORE = バケット非加重平均（ID のみ。学習データに ood は無い）。"""
    g = m.groupby("grp")["correct"].mean()
    return float(g.mean())


def main() -> None:
    rows, buckets = [], {}
    for d in sorted(RES.iterdir()):
        if not d.is_dir() or d.name.startswith("SMOKE") or "SMOKE" in d.name:
            continue
        got = load(d)
        if got is None:
            continue
        m, meta = got
        uses_d11 = bool(meta.get("adapter", "")) and "expD11" in str(meta.get("adapter", ""))
        contam = (1 - m["clean"].mean()) if uses_d11 else 0.0
        rows.append({
            "run": d.name, "track": meta["track"], "n": len(m),
            "frames": meta["n_frames"], "px": meta["size"],
            "SCORE": round(score(m), 4),
            "汚染率": f"{contam:.1%}" if uses_d11 else "—",
            "latency": round(meta.get("latency_mean", 0), 2),
        })
        b = m.groupby("grp")["correct"].agg(["mean", "size"])
        buckets[d.name] = b

    df = pd.DataFrame(rows).sort_values(["track", "SCORE"])
    print("# 全 run の SCORE\n")
    print(df.to_markdown(index=False))

    for track in ("SEGMENT", "PROCEDURE"):
        names = [r["run"] for r in rows if r["track"] == track]
        if not names:
            continue
        print(f"\n\n# {track} バケット別 accuracy（括弧内は n）\n")
        tbl = {}
        for n in names:
            b = buckets[n]
            tbl[n] = {g: (f"{b.loc[g,'mean']:.4f} ({int(b.loc[g,'size'])})"
                          if g in b.index else "—") for g in GROUPS}
            tbl[n]["**SCORE**"] = f"**{df[df.run==n].SCORE.iloc[0]:.4f}**"
        print(pd.DataFrame(tbl).T.to_markdown())


if __name__ == "__main__":
    main()
