"""responses.json を公式 Evaluator + LLM judge で採点し、バケット別内訳まで出す.

## 既知の落とし穴を最初から回避している
- **`qID` はグローバルに一意でない**（heico と lapchole で衝突）→ **dataset ごとに Evaluator を回す**
- **`DatasetSplit('train')` で突合すると responses の一部が黙って落ちる** → `DatasetSplit.ALL` を使う
- **SCORE は自作せず `Evaluator.pre_evaluation_score(results_df)` を使う**
  （`results_df.primary` は小文字、`Capability.name` は大文字で、自作マップは静かに壊れる）
- **track を渡さないと latency 上限が効かない**（SEGMENT 15s / PROCEDURE 30s）

Usage:
  PYTHONPATH=$PWD/reference/src CUDA_VISIBLE_DEVICES=3 \
    .venv/bin/python workspace/expE01_segproc_baseline/eval_seg.py \
      workspace/expE01_segproc_baseline/results/zeroshot_16f448 --track SEGMENT
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import transformers

import focus.evaluation.judges as _J
from focus import Evaluator, FocusConfig, FocusDataset, Response, set_config
from focus.enums import DatasetSplit, Track

# 公式 toolkit のバグ対処: judges.py が AutoTokenizer/AutoModelForCausalLM を import しておらず
# transformers 5.x で NameError になる。vendored コードは触らず名前空間へ注入する。
_J.AutoTokenizer = transformers.AutoTokenizer
_J.AutoModelForCausalLM = transformers.AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
for n in ("httpx", "httpcore", "filelock", "fsspec", "urllib3", "datasets", "huggingface_hub"):
    logging.getLogger(n).setLevel(logging.WARNING)
log = logging.getLogger("expE01.eval")

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--track", default="SEGMENT", choices=["SEGMENT", "PROCEDURE", "FRAME"])
ap.add_argument("--no-judge", action="store_true",
                help="judge 無し（open_ended/MC が全滅するので**比較にも使えない**。デバッグ専用）")
ap.add_argument("--no-latency", action="store_true",
                help="latency 上限を無効化（GPU 混雑下でのモデル比較専用。提出判断に使うな）")
ap.add_argument("--keep-dup-videos", action="store_true",
                help="重複動画を val に残す（既定は除外）。下の DUP_VAL_VIDEOS を参照")
ap.add_argument("--time-postproc", default="on", choices=["on", "off"],
                help="time の個数補正＋区間クランプを raw から再適用する（既定 on）。"
                     "★推論側でなくここで掛けるのは、**アーム間で必ず同一の後処理**にするため")
ap.add_argument("--out", default="")
a = ap.parse_args()

# ★fold v003 の fold0 val に入っている「学習側動画のバイト完全一致コピー」。
#   `0027 - Laparoscopic Cholecystectomy.mp4` は `0017`(fold2=train) と md5 まで同一。
#   除外しないと fold0 val の **2.0%（SEGMENT 81問 / PROCEDURE 40問 / FRAME 80問）**が
#   「学習済み映像を当てているだけ」の楽観バイアスになる。
#   ※他3ペア(0137/0183, 0140/0158, 0234/0242)は fold0 val を汚染しない（確認済み）。
#   根本対処は fold 生成の原子単位を「動画」→「重複グループ」に変えること（未実施）。
DUP_VAL_VIDEOS = {"0027 - Laparoscopic Cholecystectomy.mp4"}

DATA_ROOT = Path(__file__).resolve().parents[2] / "data/focus"
set_config(FocusConfig(root_dir=str(DATA_ROOT)))
track = Track[a.track]

raw = json.loads((a.run_dir / "responses.json").read_text())
log.info(f"{len(raw)} responses from {a.run_dir}")
if not a.keep_dup_videos:
    n0 = len(raw)
    raw = [r for r in raw if r.get("videoID") not in DUP_VAL_VIDEOS]
    log.info(f"dropped {n0-len(raw)} responses on duplicate-of-train videos "
             f"({DUP_VAL_VIDEOS}) → {len(raw)}")

# ── time の後処理を **raw から** 再適用する ────────────────────────────
# ★推論スクリプト側でなくここで掛ける理由: フレーム数などを変えたアーム同士を比較するとき、
#   後処理が実行時点のコード次第で変わると **matched 比較が壊れる**。
#   `responses.json` は `raw`（後処理前）を保存しているので、採点時に一括で揃えられる。
if a.time_postproc == "on":
    sys.path.insert(0, str(Path(__file__).parent))
    from answer_norm import normalize_all
    from time_postproc import TimePostproc

    # ── 形式の正規化（`verify()` を通す形へ整える）──
    # ★公式 verify は厳格で、`'1.'`(Number) や `'Yes.'`(Binary) は**例外＝自動的に不正解**。
    #   FRAME LoRA を SEGMENT に当てると `'1.'` が51件・`'Yes.'` 系が多発し、
    #   **正解を言っているのに句読点1つで 25問 落としていた**（zero-shot と prior では変化 0）。
    #   意味を変えない範囲でだけ直す。open_ended / multiple_choice は judge 採点なので触らない。
    n_norm = normalize_all(raw)
    log.info(f"回答形式の正規化: {n_norm} 件変更")
    tp = TimePostproc.load()
    n_ch = 0
    for r in raw:
        if r.get("fmt") != "time":
            continue
        src = r.get("raw", r["content"])
        new = tp.apply(src, r["question"], r["start_time"], r["end_time"])
        n_ch += (new != r["content"])
        r["content"] = new
    log.info(f"time 後処理を適用: {n_ch} 件変更（個数補正 + 区間クランプ）")

judge = None if a.no_judge else _J.TransformersJudge(device="cuda")
ev = Evaluator() if a.no_judge else Evaluator(judges=[judge])

frames = []
for dsn in ("heico", "lapchole"):
    sub = [r for r in raw if r["dataset"] == dsn]
    if not sub:
        continue
    qids = {r["qID"] for r in sub}
    # ★DatasetSplit.ALL。'train' で引くと test split 側の質問が突合できず黙って落ちる
    ds = FocusDataset(dsn, DatasetSplit.ALL, track)
    keep = [(rq, rf) for rq, rf in zip(ds.requests, ds.references) if rq.qID in qids]
    reqs = [x[0] for x in keep]
    refs = [x[1] for x in keep]
    resp = [Response(qID=r["qID"],
                     content=r["content"],
                     latency=(0.0 if a.no_latency else r["latency"])) for r in sub]
    log.info(f"  {dsn}: responses={len(resp)} matched refs={len(refs)}")
    res, _ = ev.run(reqs, refs, resp, track=None if a.no_latency else track)
    res["dataset"] = dsn
    # ★`pre_evaluation_score` は **(score, buckets_df) のタプル**を返す（scalar ではない）。
    #   列に代入しようとすると `Length of values (2) does not match length of index` で落ちる。
    score_ds, _buckets = ev.pre_evaluation_score(res)
    res["score_dataset"] = score_ds
    frames.append(res)

results = pd.concat(frames, ignore_index=True)
results.to_csv(a.run_dir / "results_judge.csv", index=False)

# 応答側のメタ（形式・入力トークン等）を結合して分析できるようにする
meta = pd.DataFrame([{k: r[k] for k in ("qID", "dataset", "uid", "fmt", "group", "primary",
                                        "latency", "n_input_tokens", "n_frames")} for r in raw])
merged = results.merge(meta, on=["qID", "dataset"], how="left", suffixes=("", "_resp"))
# ★公式の正誤列は `correctness`（`correct` ではない）、timeout 列は `timed_out`。
#   下流（compare_runs.py / make_qa_cards.py）が `correct` を見るので別名を用意する。
merged["correct"] = merged["correctness"]
merged.to_csv(a.run_dir / "results_merged.csv", index=False)

lines = [f"# {a.run_dir.name} — {a.track}", ""]
pooled, pooled_buckets = ev.pre_evaluation_score(results)
lines += ["## SCORE (pre-evaluation, バケット非加重平均)", "",
          "| 集合 | SCORE | n |", "|---|---|---|"]
for dsn, g in results.groupby("dataset"):
    lines.append(f"| {dsn} | **{float(g['score_dataset'].iloc[0]):.4f}** | {len(g)} |")
lines.append(f"| pooled | **{pooled:.4f}** | {len(results)} |")
lines += ["",
          "★公式 SCORE は 5 group × ID/OOD の10バケット平均だが、"
          "**学習データに ood=1 は0件**なので ID 側5バケットのみが埋まる（警告は正常）。", "",
          "## 公式バケット内訳（`Evaluator.pre_evaluation_score` の buckets_df）", "",
          pooled_buckets.round(4).to_markdown(index=False), ""]

if "group" in merged.columns:
    t2 = merged.groupby(["group", "fmt"])["correct"].agg(["mean", "size"]).round(4)
    lines += ["## バケット×回答形式", "", t2.to_markdown(), ""]
    t3 = merged.groupby("fmt")["correct"].agg(["mean", "size"]).round(4)
    lines += ["## 回答形式別", "", t3.to_markdown(), ""]
    t4 = merged.groupby(["dataset", "group"])["correct"].agg(["mean", "size"]).round(4)
    lines += ["## dataset×バケット", "", t4.to_markdown(), ""]

if "timed_out" in results.columns:
    lines += [f"- timeout: {results['timed_out'].mean():.4f} "
              f"({int(results['timed_out'].sum())}/{len(results)})", ""]
lat = [r["latency"] for r in raw if r["latency"] > 0]
if lat:
    lines += [f"- latency mean {sum(lat)/len(lat):.2f}s / max {max(lat):.2f}s "
              f"（上限 {a.track} = {'15' if a.track=='SEGMENT' else '30' if a.track=='PROCEDURE' else '5'}s）", ""]

out = Path(a.out) if a.out else a.run_dir / "eval_summary.md"
out.write_text("\n".join(lines))
print("\n".join(lines))
log.info(f"wrote {out}")
