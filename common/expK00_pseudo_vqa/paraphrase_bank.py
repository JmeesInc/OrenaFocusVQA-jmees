"""質問パラフレーズ銀行 — 答えを変えない質問文の言い換え変種.

## 生成方式の決定（2026-08-25 ユーザ相談への回答）

- **答え（事実）は必ずアノテーションから機械導出**し、LLM には作らせない
  （LLM が答えを作ると ~30% の注釈ノイズにさらに幻覚が乗る）。
- **質問文の言い換えはテンプレ単位**なので件数は 12 family × 十数変種しかない。
  → API/27B を回すまでもなく Claude Code が直接執筆した（このファイル）。
  問題単位の LLM 生成が必要になるのは、注釈から導出できない属性系（1c/1e）を
  視覚 LLM に作らせる場合のみで、それは answer 検証ループとセットで v2 以降。

## 執筆時の不変条件

- **意味の同一性**: instances（個体数）と classes（種類数）の区別を全変種で明示的に保つ
- **回答形式の指定文は保持**（number / class name / yes-no / 選択肢列挙 / hh:mm:ss）。
  multiple_choice の選択肢列挙と positions_all のフォーマット規定文は**一言一句変えない**
- プレースホルダはテンプレ本体と同じ（{plural}, {cls}, {quad}, {pa}, {pb}, {hms}）
- 各 family の先頭要素は公式テンプレそのもの（= paraphrase 率の制御は利用側で行う）

利用例（学習側）:
    from paraphrase_bank import BANK, sample_question
    q = sample_question("inst_count", rng)                 # 引数なし family
    q = sample_question("class_count", rng, plural="Clips")
"""
from __future__ import annotations

import random

_FMT_POSITIONS = (" Please provide the answer in the following format: “number. object type: quadrant”, "
                  "where number represents an enumeration starting with 1, object type is the type of the "
                  "foreign object and quadrant is one of the following options: top/left, top/right, "
                  "bottom/left, bottom/right. Respond “none” in case there are no foreign objects present "
                  "at timepoint {hms}. For example: 1. Sponge: top/left 2. Sponge: top/right 3. Needle: bottom/left")

BANK: dict[str, list[str]] = {
    "inst_count": [
        "How many different foreign object instances appear in this frame? Please provide a number.",
        "Count the individual foreign object instances visible in this frame. Please provide a number.",
        "What is the total number of separate foreign object instances shown in this frame? Please provide a number.",
        "How many distinct foreign object instances can be seen in this image? Please provide a number.",
        "Counting each instance separately, how many individual foreign objects are present in this frame? Please provide a number.",
        "In this video frame, how many foreign object instances are visible in total? Please provide a number.",
        "State the count of all foreign object instances appearing in this frame. Please provide a number.",
        "Considering every instance separately, how many foreign objects are present in this frame? Please provide a number.",
    ],
    "class_diversity": [
        "How many different foreign object classes appear in this frame? Please provide a number.",
        "How many distinct types of foreign objects are visible in this frame? Please provide a number.",
        "Count the number of different foreign object categories present in this frame. Please provide a number.",
        "What is the number of unique foreign object classes shown in this image? Please provide a number.",
        "In this frame, how many kinds of foreign objects can be seen (counting each class once)? Please provide a number.",
        "State how many separate foreign object types appear in this video frame. Please provide a number.",
    ],
    "class_count": [
        "How many {plural} appear in this frame? Please provide a number.",
        "Count the {plural} visible in this frame. Please provide a number.",
        "What is the number of {plural} present in this image? Please provide a number.",
        "How many {plural} can be seen in this video frame? Please provide a number.",
        "State the total count of {plural} shown in this frame. Please provide a number.",
        "In this frame, how many {plural} are visible? Please provide a number.",
    ],
    "list_all": [
        "List all foreign objects that are visible in this video frame. Please provide the class names or answer with none.",
        "Name every foreign object class that can be seen in this frame. Please provide the class names or answer with none.",
        "Which foreign objects are present in this video frame? Please provide the class names or answer with none.",
        "Enumerate the foreign object classes visible in this image. Please provide the class names or answer with none.",
        "What foreign objects appear in this frame? Please provide the class names or answer with none.",
        "Please identify all foreign object classes shown in this video frame. Provide the class names or answer with none.",
    ],
    "single_object": [
        "There is one surgical foreign object visible in the frame. What surgical foreign object is visible in this video frame? Please provide a class name.",
        "Exactly one surgical foreign object is visible in this frame. Which class is it? Please provide a class name.",
        "A single surgical foreign object appears in this video frame. What is its class? Please provide a class name.",
        "One surgical foreign object can be seen in this frame. Identify it. Please provide a class name.",
        "This frame contains exactly one surgical foreign object. What type of foreign object is it? Please provide a class name.",
    ],
    "combination": [
        "Which combination of foreign object classes is visible in this frame? Please provide the class names or answer with none.",
        "What set of foreign object classes appears together in this frame? Please provide the class names or answer with none.",
        "Which foreign object classes co-appear in this video frame? Please provide the class names or answer with none.",
        "Name the combination of foreign object types present in this image. Please provide the class names or answer with none.",
    ],
    "same_class": [
        "Are all visible foreign objects in this frame of the same class? Please answer with yes or no.",
        "Do all foreign objects shown in this frame belong to a single class? Please answer with yes or no.",
        "Is every foreign object visible in this frame of one and the same type? Please answer with yes or no.",
        "In this frame, are the visible foreign objects all of identical class? Please answer with yes or no.",
    ],
    "co_occur": [
        "Do {pa} and {pb} co-occur in this frame? Please answer with yes or no.",
        "Are {pa} and {pb} both visible in this frame? Please answer with yes or no.",
        "Can {pa} and {pb} be seen together in this video frame? Please answer with yes or no.",
        "Does this frame contain both {pa} and {pb} at the same time? Please answer with yes or no.",
    ],
    "quad_to_class": [
        "What class is the foreign object located in the {quad} relative to the image center? Please provide a class name.",
        "Which foreign object class occupies the {quad} quadrant relative to the image center? Please provide a class name.",
        "Relative to the image center, a foreign object lies in the {quad}. What is its class? Please provide a class name.",
        "Identify the class of the foreign object found in the {quad} region relative to the image center. Please provide a class name.",
    ],
    "class_to_quad": [
        # 選択肢列挙は一言一句固定
        "Where is the center of the {cls} located relative to the image center in this frame? Please select one answer: top/left; top/right; bottom/left; bottom/right",
        "Relative to the image center, in which quadrant does the center of the {cls} lie in this frame? Please select one answer: top/left; top/right; bottom/left; bottom/right",
        "Considering the image center as origin, where is the {cls} centered in this frame? Please select one answer: top/left; top/right; bottom/left; bottom/right",
    ],
    "closest_center": [
        "Which of the visible foreign objects has its centre closest to the centre of the image? Please provide a class name.",
        "Among the foreign objects visible in this frame, which one lies nearest to the image center? Please provide a class name.",
        "Which foreign object class is positioned closest to the middle of the frame? Please provide a class name.",
    ],
    "positions_all": [
        # フォーマット規定文は固定。リード文のみ言い換え
        "At timepoint {hms} please provide all relative central positions of foreign objects present in the frame." + _FMT_POSITIONS,
        "For the frame at timepoint {hms}, report the quadrant of every visible foreign object relative to the image center." + _FMT_POSITIONS,
        "At timepoint {hms}, list each foreign object present together with the quadrant of its center relative to the image center." + _FMT_POSITIONS,
    ],
}


import re as _re

# 正準テンプレ（parquet に保存された question）からパラメータを回収する regex
_PARAM_RX = {
    "class_count": _re.compile(r"^How many (?P<plural>.+?) appear in this frame\?"),
    "co_occur": _re.compile(r"^Do (?P<pa>.+?) and (?P<pb>.+?) co-occur in this frame\?"),
    "quad_to_class": _re.compile(r"located in the (?P<quad>top/left|top/right|bottom/left|bottom/right) relative"),
    "class_to_quad": _re.compile(r"^Where is the center of the (?P<cls>.+?) located relative"),
    "positions_all": _re.compile(r"^At timepoint (?P<hms>\d{2}:\d{2}:\d{2}) "),
}
# cutpaste の needle 特化 family → 基底 family
_ALIAS = {"class_count_needle": "class_count", "class_to_quad_needle": "class_to_quad",
          "co_occur_needle": "co_occur"}


def rephrase(family: str, question: str, rng: random.Random,
             paraphrase_prob: float = 0.5) -> str:
    """parquet の正準 question を、パラメータを保ったまま言い換える。

    正準文からパラメータを regex で回収して変種へ再充填する。
    回収に失敗した場合（未知の family / 文面）は**元の質問をそのまま返す**（安全側）。
    """
    fam = _ALIAS.get(family, family)
    if fam not in BANK or rng.random() >= paraphrase_prob:
        return question
    rx = _PARAM_RX.get(fam)
    kw: dict[str, str] = {}
    if rx is not None:
        m = rx.search(question)
        if m is None:
            return question
        kw = m.groupdict()
    t = rng.choice(BANK[fam][1:]) if len(BANK[fam]) > 1 else BANK[fam][0]
    try:
        return t.format(**kw) if kw else t
    except (KeyError, IndexError):
        return question


def sample_question(family: str, rng: random.Random, paraphrase_prob: float = 0.5, **kw) -> str:
    """family のテンプレを抽選して穴埋めする。先頭=公式文。paraphrase_prob で言い換え率を制御。"""
    variants = BANK[family]
    if rng.random() < paraphrase_prob and len(variants) > 1:
        t = rng.choice(variants[1:])
    else:
        t = variants[0]
    return t.format(**kw) if kw else t
