"""`time` 回答の後処理: ①区間クランプ ②タイムスタンプ個数をテンプレ事前分布に合わせる.

## なぜ必要か
`Time.compare`（`formats.py:298`）は **①個数の完全一致 ②各ペアが ±5s 以内** の両方を要求する。
つまり**時刻が合っていても個数が違えば即不正解**。

16フレーム zero-shot（SEGMENT fold0 val, time N=624）の実測:

| 症状 | n | 割合 |
|---|---|---|
| GT=1個 なのに PRED=5個 | 64 | 11.0% |
| 個数不一致 合計 | 104 | **17.9%** |

→ **個数を直すだけで +37問（悪化 0）**、time 精度 0.2564 → **0.3157**。

## 個数をどう決めるか — val を見ないこと
「先頭/末尾/中央値のどれを残すか」を val 上で比べて勝ったものを選ぶのは **val への過学習**。
GT の個数は**質問テンプレでほぼ決まる**（`at what time ...` は1個、
`at which time points ... for each individual clip instance` は複数）ので、
**train fold の GT 個数の最頻値**をテンプレ（質問の先頭8語）ごとに持つ。
これは train だけで作れるので val を汚さない。実測でも val 調律版と同じ +37/−0 に到達した。

## 提出コンテナでの使い方
事前分布は `time_count_prior.json` に**書き出して同梱**する（コンテナ内で QA parquet は引けない）。
未知テンプレは「個数を変えない」で素通しする（安全側）。

    from time_postproc import TimePostproc
    tp = TimePostproc.load("time_count_prior.json")
    answer = tp.apply(raw_text, question, start_time, end_time)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
PRIOR_PATH = HERE / "time_count_prior.json"

_TS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")


def hhmmss(sec: float) -> str:
    s = int(round(max(sec, 0)))
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"


def parse(text: str) -> list[int] | None:
    """`hh:mm:ss[, hh:mm:ss...]` を秒のリストへ。1つでも形式不正なら None（触らない）。"""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        return None
    out = []
    for p in parts:
        m = _TS.fullmatch(p)
        if not m:
            return None
        out.append(int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]))
    return sorted(out)


def tmpl_key(question: str) -> str:
    return " ".join(question.split()[:8])


class TimePostproc:
    def __init__(self, prior: dict[str, int]):
        self.prior = prior

    @classmethod
    def load(cls, path: str | Path = PRIOR_PATH) -> TimePostproc:
        return cls(json.loads(Path(path).read_text()))

    def apply(self, text: str, question: str, start: float, end: float) -> str:
        """個数をテンプレ事前分布に合わせ、[start, end] にクランプして返す。"""
        secs = parse(text)
        if secs is None:
            return text                       # 形式が壊れているものは触らない
        k = self.prior.get(tmpl_key(question))
        if k and len(secs) > k:
            # ★多く返しすぎている場合だけ削る（足りない場合は増やしようが無い）。
            #   **中央に寄せて k 個**を残す = k=1 なら中央値。
            #
            #   どれを残すかの実測（SEGMENT fold0 val, time N=656, 全て悪化0）:
            #     先頭 +20 / 末尾 +39 / 中央値 +36 / 区間中点に最近 +36 / 質問文の語で分岐 +36
            #   **末尾が3問だけ勝っているが、これは val 上の差でありノイズ**（N=656 で3問）。
            #   val で勝った規則を選ぶのは過学習なので、原理で選ぶ:
            #   許容誤差 ±5s は実質的に絶対誤差損失なので、**中央値が期待絶対誤差を最小化する**。
            #   明確に劣る「先頭」だけは採らない。
            lo = (len(secs) - k) // 2
            secs = secs[lo:lo + k]
        # ★正解が [start,end] 内にある割合は SEGMENT 0.966 / PROCEDURE 1.000（expE00 実測）
        #   → 区間外の予測は確実に外れなので端に寄せる。期待値は常に非負。
        secs = [min(max(s, start), end) for s in secs]
        return ", ".join(hhmmss(s) for s in sorted(secs))


def build_prior(track: str, val_fold: int = 0, version: str = "v004") -> dict[str, int]:
    """train fold（= val_fold 以外）の GT から テンプレ→個数の最頻値を作る。"""
    import csv
    import sys
    sys.path.insert(0, str(HERE))
    from dataset_seg import ROOT, _qa_index

    idx = _qa_index(track)
    counts: dict[str, Counter] = {}
    with (ROOT / f"workspace/fold/qa_{version}/qa_split.csv").open() as f:
        for r in csv.DictReader(f):
            if r["track"] != track or r["answer_format"] != "time":
                continue
            if int(r["fold"]) == val_fold:
                continue                       # ★val は絶対に見ない
            got = idx.get((r["dataset"], r["qID"]))
            if got is None:
                continue
            s = parse(str(got[1].answer))
            if s is None:
                continue
            counts.setdefault(tmpl_key(got[0].question), Counter())[len(s)] += 1
    return {k: c.most_common(1)[0][0] for k, c in counts.items()}


if __name__ == "__main__":
    import argparse
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    for n in ("httpx", "httpcore", "filelock", "fsspec", "datasets", "huggingface_hub"):
        logging.getLogger(n).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["SEGMENT", "PROCEDURE"])
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--out", default=str(PRIOR_PATH))
    a = ap.parse_args()
    prior: dict[str, int] = {}
    for t in a.tracks:
        p = build_prior(t, a.val_fold)
        # 同じテンプレが両トラックに出るが GT 個数の分布は同じなので統合してよい
        prior.update(p)
        print(f"{t}: {len(p)} テンプレ  "
              f"（個数の内訳 {Counter(p.values()).most_common()}）")
    Path(a.out).write_text(json.dumps(prior, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}  ({len(prior)} テンプレ)")
