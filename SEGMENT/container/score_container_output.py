"""コンテナが出した `answer.json` を公式 Evaluator + judge で採点し、CV と突き合わせる.

## これで何を確かめたいか
CV は**元動画から直接抽出した JPEG** で測っている。本番は **5fps へ再エンコードされた H.264
クリップ**（高さ<=576, キーフレーム5秒）。この**再エンコード劣化の影響が唯一の未検証項目**なので、
本番仕様クリップを食わせたコンテナ出力を、同じ qID の CV 正誤と並べて見る。

★後処理はコンテナ内で既に掛かっている（`answer.json` の content は採点対象そのもの）ので、
  ここでは何も足さない。

Usage (repo ルートから):
  PYTHONPATH=reference/src CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
    submit/v005_segment_hybrid/score_container_output.py \
      --input submit/v005_segment_hybrid/test/input/val200 \
      --answers submit/v005_segment_hybrid/test/output_val200/answer.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import transformers

import focus.evaluation.judges as _J
from focus import Evaluator, FocusConfig, FocusDataset, Response, load_requests, set_config
from focus.enums import DatasetSplit, Track

# 公式 toolkit のバグ対処（eval_seg.py と同じ）
_J.AutoTokenizer = transformers.AutoTokenizer
_J.AutoModelForCausalLM = transformers.AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
for n in ("httpx", "httpcore", "filelock", "fsspec", "urllib3", "datasets", "huggingface_hub"):
    logging.getLogger(n).setLevel(logging.WARNING)
log = logging.getLogger("score_container")

ROOT = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, help="/input として渡したディレクトリ")
ap.add_argument("--answers", required=True, help="コンテナが出した answer.json")
ap.add_argument("--track", default="SEGMENT")
ap.add_argument("--cv-run", default="workspace/expE01_segproc_baseline/results",
                help="CV 側の判定済み run の親ディレクトリ")
ap.add_argument("--no-latency", action="store_true",
                help="latency 上限を無効化（コンテナは latency=0 を書くので既定でも無効に近い）")
a = ap.parse_args()

set_config(FocusConfig(root_dir=str(ROOT / "data/focus")))
track = Track[a.track]

requests = load_requests(Path(a.input) / "request.json")
ans = {r["qID"]: r for r in json.loads(Path(a.answers).read_text())}
log.info(f"{len(requests)} requests / {len(ans)} answers")

judge = _J.TransformersJudge(device="cuda")
ev = Evaluator(judges=[judge])

frames = []
for dsn in ("heico", "lapchole"):
    ds = FocusDataset(dsn, DatasetSplit.ALL, track)
    qids = {q.qID for q in ds.requests} & {r.qID for r in requests}
    if not qids:
        continue
    keep = [(rq, rf) for rq, rf in zip(ds.requests, ds.references) if rq.qID in qids]
    reqs = [x[0] for x in keep]
    refs = [x[1] for x in keep]
    # ★latency はコンテナ側で 0 を書いている（プラットフォームが実測するので情報用）。
    #   ここで track を渡すと 0s 扱いで必ず通るため、latency 判定は別途 wall time で見る。
    resp = [Response(qID=q.qID, content=str(ans[q.qID]["content"]), latency=0.0) for q in reqs]
    res, _ = ev.run(reqs, refs, resp, track=None)
    res["dataset"] = dsn
    res["uid"] = res["qID"].map(lambda q: f"{dsn}:{q}")
    frames.append(res)

res = pd.concat(frames, ignore_index=True)
score, buckets = ev.pre_evaluation_score(res)
print(f"\n=== コンテナ出力（本番仕様クリップ経由） N={len(res)} ===")
print(f"SCORE {score:.4f}")
print(buckets.to_string(index=False))

# ── CV（元動画から直接抽出した JPEG）の同一 qID と突き合わせ ─────────────
cvdir = Path(a.cv_run)
cv = {}
for tag, cfg in (("eval_expE06g_seg64f560_anchor_n2000", "A"),
                 ("eval_expE06g_seg32f768_anchor_n2000", "B")):
    p = cvdir / tag / "results_merged.csv"
    if p.exists():
        cv[cfg] = pd.read_csv(p).set_index("uid")["correctness"].astype(bool)
if len(cv) == 2:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from router import CONFIG_A, GroupRouter
    rt = GroupRouter.load(Path(__file__).parent / "resources/group_templates.json")
    qmap = {r.qID: r.question for r in requests}
    common = [u for u in res["uid"] if u in cv["A"].index and u in cv["B"].index]
    rows = res.set_index("uid").loc[common]
    cv_corr = []
    for u in common:
        qid = u.split(":", 1)[1]
        use_a = rt.config_for(qmap[qid]) == CONFIG_A
        cv_corr.append(cv["A"][u] if use_a else cv["B"][u])
    got = rows["correctness"].astype(bool).values
    cvv = pd.Series(cv_corr).values
    agree = (got == cvv).mean()
    print(f"\n=== 同一 {len(common)} 問での CV（直接抽出JPEG）との比較 ===")
    print(f"CV 正答率        {cvv.mean():.4f}")
    print(f"コンテナ正答率   {got.mean():.4f}   （差 {got.mean()-cvv.mean():+.4f}）")
    print(f"正誤の一致率     {agree:.4f}")
    w = int(((~cvv) & got).sum()); l = int((cvv & (~got)).sum())
    print(f"コンテナが勝ち {w} / 負け {l}")
    try:
        from scipy.stats import binomtest
        if w + l:
            p = binomtest(w, w + l, 0.5).pvalue
            print(f"McNemar p={p:.4f}" + ("  ★有意な劣化なし" if p > 0.05 else "  ⚠有意差あり"))
    except ImportError:
        pass
else:
    print("\n（CV 側の run が見つからないので突き合わせは省略）")
