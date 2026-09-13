"""answer format 判定 v2 + SEGMENT/PROCEDURE 用の system prompt.

## なぜ expB01 の `detect_format` をそのまま使えないか
expB01 の判定器は **FRAME で 99.76%** だが、**SEGMENT 81.5% / PROCEDURE 84.0%** まで落ちる。
形式判定を外すと「fo_class 問題に open_ended の規則を渡す」等で verify に落ちて **0点**になるので、
先にここを潰す。実測した誤判定の内訳（SEGMENT+PROCEDURE 30,000問）:

| GT → 誤判定 | n | 原因 |
|---|---|---|
| fo_class → open_ended | 3355 | `class name(s)` が open_ended 側の判定に先に当たる。**引用符なし `none` が fo_class の印** |
| multiple_choice → open_ended | 1482 | `Please select one or multiple` 形式に `select one answer` の規則が当たらない |
| open_ended → number | 328 | `provide a single integer` は **open_ended**（`provide a number` が number）|
| percentage → number | 135 | `In %, how many ...` の `how many` が number に先に当たる |

判定に使う語尾は**テンプレ由来で規則的**なので、順序を正せば決定的に分離できる（下の PRIORITY 順）。
"""
from __future__ import annotations

import re

from focus import FO_DEFINITIONS_FILE

# ★提出コンテナでは FO 定義を `/input/FO_definitions.json` から受け取る（本番はそちらが正）。
#   パッケージ同梱の定義はローカル検証用のフォールバック。
try:
    FO_DEFINITIONS = FO_DEFINITIONS_FILE.read_text()
except Exception:  # pragma: no cover - コンテナ内で同梱ファイルが無い場合
    FO_DEFINITIONS = ""


def set_fo_definitions(text: str) -> None:
    """`/input/FO_definitions.json` の本文で上書きする（推論開始時に1回呼ぶ）。"""
    global FO_DEFINITIONS
    FO_DEFINITIONS = text

FO_CLASS_NAMES = [
    "Sponge", "Clip", "Specimen Bag", "Silicone Loop", "External Drain",
    "Needle", "Gallstone", "Specimen", "Mesh",
]

# ── 形式判定（**この順に評価する**。順序そのものが仕様） ────────────────
# 実測した弁別力（50,000問全体）:
#   'provide a single integer' → open_ended 192/192
#   'provide a/the number'     → number     9862/9862（number の全件）
#   "'none'"(引用符あり)        → open_ended 263/263
#   'answer (with) none'(裸)   → fo_class   11531/11554
#   'in %' | 'xx%' | 'percentage' → percentage 135/135
#   'please select'            → multiple_choice
_RULES: list[tuple[str, str]] = [
    # 1. パーセント: "In %, how many ..." は `how many` より先に判定する
    ("percentage", r"\bin\s*%|xx%|percentage"),
    # 2. `single integer` は open_ended（自由記述で整数を書かせるテンプレ）。
    #    `provide a number` の number と紛らわしいので先に抜く
    ("open_ended", r"provide a single integer"),
    # 3. 引用符つき 'none' は open_ended（FO 語彙外を問う）。裸の none は fo_class
    ("open_ended", r"answer (with )?['‘’\"]none['‘’\"]"),
    # 4. yes/no のあとに理由を求めるものは open_ended（binary の verify を通らない）
    ("open_ended", r"yes or no.{0,60}(reason|explain|explanation|sentence)"),
    # 5. 明示的な選択肢提示
    ("multiple_choice", r"please select"),
    # 6. 時刻
    ("time", r"hh:mm:ss|provide the time"),
    # 7. 計数
    ("number", r"provide (a|the) number"),
    # 8. 裸の none / class name を求める → fo_class
    ("fo_class", r"answer (with )?none\b|provide (a|the) class name"),
    # 9. yes/no のみ
    ("binary", r"answer with yes or no"),
]
_COMPILED = [(f, re.compile(p, re.I | re.S)) for f, p in _RULES]


def detect_format(question: str) -> str:
    for fmt, pat in _COMPILED:
        if pat.search(question):
            return fmt
    return "open_ended"


def parse_mc_options(question: str) -> list[str]:
    """`Please select one answer: a; b; c` / `Please select one or multiple ...: a; b; c` から選択肢を抽出。"""
    m = re.search(r"please select[^:]*:\s*(.+)$", question, re.I | re.S)
    if not m:
        return []
    return [o.strip() for o in m.group(1).strip().rstrip(".").split(";") if o.strip()]


def is_fo_question(question: str, fmt: str) -> bool:
    if fmt == "fo_class":
        return True
    return bool(re.search(
        r"foreign object|\bfo\b|sponge|clip|specimen|silicone loop|external drain|"
        r"needle|gallstone|\bmesh\b|instrument", question, re.I))


# ── system prompt ────────────────────────────────────────────────────
PERSONA = (
    "You are a surgical assistant analyzing endoscopic video from a "
    "minimally invasive procedure."
)
LOOK_CAREFULLY = (
    "You are shown frames sampled from the video in chronological order; each frame is "
    "labelled with its absolute timestamp. Base your answer strictly on what you can "
    "actually see in these frames. Small objects (clips, needle tips, drain tips) are "
    "easy to miss — inspect every frame before deciding. Do not fall back on a default "
    "or most-likely answer."
)


def _fo_class_rule() -> str:
    names = ", ".join(FO_CLASS_NAMES)
    return (
        "OUTPUT RULE: Answer with the exact foreign-object class name(s), taken "
        f"verbatim from this list: {names}. If several appear at any point in the "
        "shown frames, separate them with commas (e.g. `Clip, Sponge`). If none are "
        "present, answer exactly `none`. Check every class in the list one by one. "
        "Output only the class name(s) or `none`, with no other words."
    )


FORMAT_RULES = {
    "number": (
        "OUTPUT RULE: Answer with a single non-negative integer and nothing else "
        "(e.g. `2`). If there are none, answer `0`. No words, no units, no punctuation."
    ),
    "binary": (
        "OUTPUT RULE: Answer with exactly one word, either `yes` or `no`. "
        "No punctuation and no explanation."
    ),
    "percentage": (
        "OUTPUT RULE: Answer with a single number optionally followed by `%` "
        "(e.g. `12.5%`). No other words."
    ),
    "time": (
        "OUTPUT RULE: Answer with absolute timestamp(s) in `hh:mm:ss` format, using the "
        "same clock as the timestamps labelling the frames. If the question asks for "
        "several time points, separate them with commas (e.g. `00:12:30, 00:47:05`). "
        "Give exactly as many timestamps as the question asks for. No other words."
    ),
    "open_ended": (
        "OUTPUT RULE: Answer as briefly as possible — a single short phrase or sentence. "
        "Follow any answer format stated in the question exactly. No preamble, no reasoning."
    ),
    "multiple_choice": (
        "OUTPUT RULE: Answer using only the option text(s) offered in the question, "
        "copied verbatim. If several apply, separate them with commas. No other words."
    ),
}


def build_system_prompt(question: str, with_fo_definitions: bool = True) -> tuple[str, str]:
    """(system_prompt, detected_format) を返す。"""
    fmt = detect_format(question)
    parts = [PERSONA, LOOK_CAREFULLY]
    if fmt == "fo_class":
        parts.append(_fo_class_rule())
    else:
        parts.append(FORMAT_RULES[fmt])
    if with_fo_definitions and is_fo_question(question, fmt):
        parts.append("Foreign object class definitions:\n" + FO_DEFINITIONS)
    return "\n\n".join(parts), fmt


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import pandas as pd

    csv = Path(__file__).resolve().parents[1] / "expE00_segproc_eda/all_qa.csv"
    d = pd.read_csv(csv)
    d["det"] = d.question.map(detect_format)
    for t, g in d.groupby("track"):
        acc = (g.det == g.answer_format).mean()
        print(f"{t:10s} acc={acc:.4f}  n={len(g)}")
        bad = g[g.det != g.answer_format]
        if len(bad):
            print(pd.crosstab(bad.answer_format, bad.det).to_string())
            for q in bad.question.drop_duplicates().head(3):
                print("   ex:", q[:150])
