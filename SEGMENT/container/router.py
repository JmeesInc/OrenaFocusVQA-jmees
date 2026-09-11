r"""capability group を質問文から推定し、入力構成（枚数・解像度）へ振り分ける.

## 根拠（SEGMENT fold v003 fold0 val, N=1958, judge 込み）

同じ vision token 予算 `frames × (size/448)^2 ≒ 96` 上でも、**どの点が最適かは
capability group ごとに違う**:

| group | N | 64f@560 | 32f@768 | 採用 |
|---|---|---|---|---|
| object_recognition | 850 | **0.8247** | 0.8047 | 64f@560 |
| temporal_grounding | 795 | **0.5962** | 0.5421 | 64f@560 |
| aggregation | 186 | 0.5000 | **0.5376** | 32f@768 |
| event_understanding | 74 | 0.8243 | **0.8649** | 32f@768 |
| complex_reasoning | 53 | 0.8113 | **0.8491** | 32f@768 |

→ **解像度が効くのは集計・イベント理解・複雑推論、枚数が効くのは時間定位**。
振り分けた SCORE は **0.7345**（単独ベスト 32f@768 の 0.7197 に +0.0148）。

## group は Request に入っていない
`focus.Request` は qID / videoID / start_time / end_time / procedure_type / question のみ。
そこで**質問文テンプレート**（`hh:mm:ss`・数値・FO 名を伏せ字にした文字列）から引く。
表は `build_router_table.py` が作る。未知テンプレは 64f@560 側へ倒す
（SEGMENT の 84.7% が object_recognition + temporal_grounding なので多数派が安全）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ── 振り分け先（枚数, 幅px）─────────────────────────────────────────────
# ★`frames × (size/448)^2 <= 96` の容量線上の2点。probe_capacity.py の実測に対応。
CONFIG_A = (64, 560)   # object / temporal / aggregation / event  ← expR00 の A/B で実測
CONFIG_B = (32, 768)   # complex_reasoning のみ

# ★★2026-09-05 expR00 の router A/B 実測（val 3,924問・同一 adapter・同一機体・judge込み）で
#   **v013/v015 の割当が 2 点間違っていた**ことが分かったので改めた。
#   64f@560(A) vs 32f@768(B) の matched 比較:
#     object_recognition  n=1693  A 0.8447 / B 0.8358  Δ+0.0089  p=0.235   n.s.（A 寄り）
#     temporal_grounding  n=1653  A 0.7084 / B 0.6673  Δ+0.0411  p=2.4e-05 ★A
#     aggregation         n= 369  A 0.5637 / B 0.5230  Δ+0.0407  p=0.032   ★A  ← v015 は B に割当
#     event_understanding n= 111  A 0.8198 / B 0.8018  Δ+0.0180  p=0.69    n.s.（A 寄り）
#     complex_reasoning   n=  98  A 0.8469 / B 0.8776  Δ−0.0306  p=0.38    n.s.（B 寄り）
#   合成 SCORE: この割当 0.7628 / v015 の割当 0.7511 / 全問A 0.7567 / 全問B 0.7411
#   ★機序: 64枚だと 119s クリップで間隔 1.9s となり temporal の許容誤差 ±2.3s を下回る
#     （32枚は 3.7s で許容より粗い）。aggregation(計数) も見落としが減る。
#   ⚠️complex は n=98・不一致 5 問のみの n.s.。小バケットの単発差はノイズになりうる
#     （[[small-bucket-single-run-is-noise]]）。B に置くのは実測に従っただけで確信は薄い。
#   ⚠️val は学習に含まれる（train_part: all）ので**絶対値は無意味**。相対差のみが根拠。
GROUP_A = frozenset({"OBJECT_RECOGNITION", "TEMPORAL_GROUNDING",
                     "AGGREGATION", "EVENT_UNDERSTANDING"})
DEFAULT_GROUP = "OBJECT_RECOGNITION"   # 未知テンプレの既定（= CONFIG_A）

_TS = re.compile(r"\d{2}:\d{2}:\d{2}")
_NUM = re.compile(r"\b\d+(\.\d+)?\b")

# ★長い名前から先に置換する（`Specimen Bag` を `Specimen` より先に潰す）
_FO_NAMES = [
    "Specimen Bags", "Specimen Bag", "Silicone Loops", "Silicone Loop",
    "External Drains", "External Drain", "Gallstones", "Gallstone",
    "Specimens", "Specimen", "Sponges", "Sponge", "Needles", "Needle",
    "Clips", "Clip", "Meshes", "Mesh",
]


def template_key(question: str) -> str:
    """質問文を「テンプレート」へ畳む（可変部を伏せ字にする）。"""
    s = _TS.sub("<TS>", str(question))
    s = _NUM.sub("<N>", s)
    for name in _FO_NAMES:
        s = s.replace(name, "<FO>")
    return s.strip()


class GroupRouter:
    def __init__(self, table: dict[str, str]):
        self.table = table
        self.n_unknown = 0

    @classmethod
    def load(cls, path: str | Path) -> GroupRouter:
        return cls(json.loads(Path(path).read_text()))

    def group_of(self, question: str) -> str:
        g = self.table.get(template_key(question))
        if g is None:
            self.n_unknown += 1
            return DEFAULT_GROUP
        return g

    def config_for(self, question: str) -> tuple[int, int]:
        """(n_frames, width_px) を返す。"""
        return CONFIG_A if self.group_of(question) in GROUP_A else CONFIG_B
