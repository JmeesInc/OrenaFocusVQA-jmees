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

# ── 振り分け先（PROCEDURE）──────────────────────────────────────────────
# ★SEGMENT は「同じ token 予算の上でどの点が最適か」の振り分けだったが、
#   PROCEDURE は **予算が余っている**（1バッチ 720s に対し 64f は ~140s）ので、
#   「group ごとにどこまで枚数を増やすと得か」の振り分けになる。
#
# 根拠（PROCEDURE fold v003 fold0 val, N=1960, judge 込み。claudeSummary の実測表）:
#
# | group | N | 64f | 96f | Δ | 問数換算 | 採用 |
# |---|---|---|---|---|---|---|
# | temporal_grounding | 769 | 0.1782 | **0.2406** | +0.0624 | **+48問** | **96f** |
# | aggregation        | 544 | 0.2463 | **0.2702** | +0.0239 | **+13問** | **96f** |
# | object_recognition | 560 | **0.6089** | 0.5929 | −0.0160 | −9問 | 64f |
# | event_understanding|  37 | **0.7568** | 0.7027 | −0.0541 | −2問 | 64f |
# | complex_reasoning  |  50 | **0.5400** | 0.5200 | −0.0200 | −1問 | 64f |
#
# ★**64f/96f 単独はほぼ同点（0.4660 / 0.4653）なのに、振り分けると 0.4833（+0.0173）**。
# ⚠️ event / complex の「64f が上」は **1〜2問の差＝ノイズ**。ただし 64f は確立した既定なので
#    そちらへ倒すのは安全側。**信頼できる差は temporal(+48問) と aggregation(+13問) だけ**。
# ★★temporal は枚数に**単調**（16f 0.0676 → 32f 0.1092 → 64f 0.1782 → 96f 0.2406）＝**未飽和**。
#    予算が余るので **128f まで伸ばす梯子**を用意し、実測スループットで届く所まで登る。
#
# 各 group の「枚数の梯子」。左が第一希望で、予算が足りなければ右へ降りる。
# ★★v016: **最上段を 64 に揃えた**。
#   v010 までは 128/96 が最上段だったが、`ClipContext.times()` が枚数を受け取らないバグで
#   索引経路では**常に 64 枚**になっており、梯子は事実上死んでいた（＝ 64 固定で動いていた）。
#   v016 でそのバグを直した結果、放置すると 96〜128 枚が実際に送られてしまう。
#   expP00C は **64 枚で学習している**ので、通常運転は 64 枚が正しい条件。
#   → 上へは登らせず、**予算が足りないときに下へ降りるためだけ**に梯子を使う。
#   （128/96 が有利という過去の実測は「学習が一様16枚だった頃」のもので、
#     学習と推論を揃えた今は前提が変わっている。上げ直すなら CV を取り直すこと）
#   ★★★2026-09-05 実測で **単段(64固定)に戻した**。理由:
#     `ctx.times()` が枚数を無視していたため梯子は v010 まで**一度も動いていない**
#     ＝ `Ladder.est()` のコストモデルは**未校正**。実際に通してみると
#     `est(64, 16189s) = 0.0035*16189 + 0.08*64 = 61.8s > 予算 30.6s` で即座に最下段へ落ち、
#     4.5h クリップの 1 問目が **15 枚**になった（dl2 RTX4090 実測）。
#     だがデコードは**必要な index だけ読む**ので実測 64 枚 2.6s であり、
#     「デコード時間 ∝ クリップ長」という仮定自体が誤っている。
#     v010 は常に 64 枚を送って LB 11.9s/問（予算の 41%）で回っており、
#     expP00C も 64 枚で学習している。**64 固定が検証済みの構成**。
#     → 梯子を使うなら先に `est()` を実測で校正すること（本番前日に触る変更ではない）。
FRAME_LADDER = {
    "TEMPORAL_GROUNDING":  (64,),
    "AGGREGATION":         (64,),
    "OBJECT_RECOGNITION":  (64,),
    "EVENT_UNDERSTANDING": (64,),
    "COMPLEX_REASONING":   (64,),
}
WIDTH_PX = 448                  # PROCEDURE の CV は全て 448 で取っている。動かさない
DEFAULT_GROUP = "TEMPORAL_GROUNDING"   # 未知テンプレの既定（PROCEDURE の最大 group）

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

    def ladder_for(self, question: str) -> tuple[int, ...]:
        """その問の「枚数の梯子」（第一希望から順に降りる）。"""
        return FRAME_LADDER.get(self.group_of(question), FRAME_LADDER[DEFAULT_GROUP])

    def config_for(self, question: str, level: int = 0) -> tuple[int, int]:
        """(n_frames, width_px)。`level` だけ梯子を降りた構成を返す。"""
        lad = self.ladder_for(question)
        return lad[min(level, len(lad) - 1)], WIDTH_PX
