"""映像を一切見ない prior の回答を `responses.json` 形式で書き出す（= judge で正式採点できる）.

## なぜ必要か
`vs_prior.py` は judge 不要形式（77%）でしか比較できず、**open_ended / multiple_choice が抜けている**。
そこは SEGMENT の COMPLEX_REASONING / EVENT_UNDERSTANDING の中身そのもので、
**配点の 40% を占める2バケット**なので、抜けたままでは「prior が下限」と言い切れない。

prior を VLM の出力と同じ器（`responses.json`）に入れれば、`eval_seg.py` にそのまま食わせて
**公式 Evaluator + judge で同一条件の SCORE** が出る。

## prior の作り方（train fold だけを見る）
- `time`  : 与えられた区間の中点（正解の 96.6%/100% が区間内にある）
- その他  : 質問テンプレ（先頭8語）→ train fold の最頻回答。未知テンプレは形式ごとの最頻回答

latency は 0.0（映像を見ないので計算不要）。**latency 0 は上限判定を必ず通る**ので、
「prior は latency で有利」という交絡は無い（VLM 側も 1.29s で上限 15s に対し余裕）。

Usage:
  .venv/bin/python workspace/expE01_segproc_baseline/make_prior_responses.py \
     --track SEGMENT --fold 0 --limit 4006 --out-tag prior_seg
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from dataset_seg import ROOT, _group, _qa_index, load_qa_rows  # noqa: E402
from prompts_seg import detect_format  # noqa: E402
from time_postproc import hhmmss  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
for n in ("httpx", "httpcore", "filelock", "fsspec", "datasets", "huggingface_hub"):
    logging.getLogger(n).setLevel(logging.WARNING)
log = logging.getLogger("expE01.prior")


def build_prior(track: str, val_fold: int, version: str):
    idx = _qa_index(track)
    tmpl: dict[str, Counter] = {}
    fmt: dict[str, Counter] = {}
    with (ROOT / f"workspace/fold/qa_{version}/qa_split.csv").open() as f:
        for r in csv.DictReader(f):
            if r["track"] != track or int(r["fold"]) == val_fold:
                continue                      # ★val は絶対に見ない
            if r["answer_format"] == "time":
                continue                      # 絶対時刻なので最頻値は無意味
            got = idx.get((r["dataset"], r["qID"]))
            if got is None:
                continue
            q, ans = got[0].question, str(got[1].answer)
            tmpl.setdefault(" ".join(q.split()[:8]), Counter())[ans] += 1
            fmt.setdefault(r["answer_format"], Counter())[ans] += 1
    return ({k: c.most_common(1)[0][0] for k, c in tmpl.items()},
            {k: c.most_common(1)[0][0] for k, c in fmt.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="SEGMENT")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--version", default="v004")
    ap.add_argument("--out-tag", required=True)
    a = ap.parse_args()

    tmpl, fmt = build_prior(a.track, a.fold, a.version)
    log.info(f"prior: テンプレ {len(tmpl)} / 形式 {len(fmt)}（train fold のみ）")

    idx = _qa_index(a.track)
    rows = load_qa_rows(a.version, a.track, a.fold, "val", a.limit)
    out = []
    n_unk = 0
    for r in rows:
        got = idx.get((r["dataset"], r["qID"]))
        if got is None:
            continue
        req, ref = got
        f = detect_format(req.question)
        if f == "time":
            content = hhmmss((req.start_time + req.end_time) / 2)
        else:
            key = " ".join(req.question.split()[:8])
            if key in tmpl:
                content = tmpl[key]
            else:
                n_unk += 1
                content = fmt.get(f, "none")
        out.append({
            "qID": req.qID, "uid": r["uid"], "dataset": r["dataset"], "videoID": req.videoID,
            "content": content, "raw": content, "latency": 0.0, "n_input_tokens": 0,
            "fmt": f, "primary": r["primary"], "group": _group(r["primary"]),
            "answer": str(ref.answer), "question": req.question, "n_frames": 0,
            "start_time": req.start_time, "end_time": req.end_time,
        })

    d = HERE / "results" / a.out_tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "responses.json").write_text(json.dumps(out, indent=1))
    (d / "meta.json").write_text(json.dumps({
        "track": a.track, "fold": a.fold, "part": "val", "n": len(out),
        "n_frames": 0, "size": 0, "adapter": "(prior: 映像を見ない)",
        "base_model": "(none)", "dtype": "-", "clamp": True,
        "latency_mean": 0.0, "latency_max": 0.0, "input_tokens_mean": 0.0}, indent=1))
    log.info(f"wrote {d}/responses.json  n={len(out)}  未知テンプレ {n_unk}")


if __name__ == "__main__":
    main()
