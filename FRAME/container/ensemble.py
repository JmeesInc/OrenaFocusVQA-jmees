r"""3モデル多数決と、予算に応じた打ち切り（anytime）の制御.

## なぜ「パス単位」で回すのか — 打ち切り設計の中心

素朴には「1問ごとに member1→2→3 を回す」が、それだと**途中で予算が尽きたとき
後半の問が member1 すら通っていない**状態になる。公式のペナルティは

  allowed = 120s + B×5s（FRAME）／超過分に応じて**一部の質問が forfeit**／
  **20%超過でバッチ全問 forfeit**／**forfeit される質問はこちらで選べない**

なので「一部の問が未回答」は致命的。そこで**パス単位**にする:

    pass 1: 全問を member1（= 単独ベスト）で回答  ← ここで提出物として完成する
    pass 2: 余裕があれば全問を member2 で追加
    pass 3: 余裕があれば全問を member3 で追加
    最後に多数決

- **pass 1 完了時点で全問に有効な回答がある**。以降は純粋な上積み
- pass 2/3 は**途中で止めてよい**。到達しなかった問は前パスの回答のまま残る
- **pass 1 の実測から pass 2/3 が入るかを判断できる**（見積もりでなく実測で決める）
- adapter 切り替えが**パスあたり1回**で済む（問ごとに3回切り替えるより安い）

## 多数決の規則
- 同数のときは**メンバー順（= 単独 SCORE の高い順）で先勝ち**
- したがって **k=1 は member1 単独と完全に一致**し、k=2 も実質 member1 に一致する
  （実測: 2本 0.6392 vs member1 単独 0.6396 = 誤差）。**利得が出るのは k=3 から**

## 実測（fold v003 fold0 val, N=1962, judge 込み）
| 規則 | SCORE | McNemar vs member1 |
|---|---|---|
| member1 単独（V05 1024px）| 0.6396 | — |
| **多数決 3本** | **0.6548** | 83勝53敗 **p=0.0126 \*** |
| 多数決 5本 | 0.6483 | p=0.26 n.s. |
| 多数決 7本 | 0.6490 | p=0.27 n.s. |
"""
from __future__ import annotations

import logging
import time
from collections import Counter

log = logging.getLogger(__name__)


class Budget:
    """バッチでプールされた予算を管理する（実測スループットで判断する）。

    ★プラットフォームが測るのは**実行全体の wall time**（デコード・I/O・モデルロード込み）。
      `Response.latency` は情報用でスコアに使われない。
    """

    def __init__(self, n_questions: int, setup_s: float, per_q_s: float,
                 safety: float = 0.85):
        self.t0 = time.monotonic()
        self.allowed = setup_s + n_questions * per_q_s
        # ★20%超過でバッチ全滅なので、その半分以下のところに自前の締切を置く
        self.deadline = self.allowed * safety
        self.n = n_questions

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    @property
    def left(self) -> float:
        return self.deadline - self.elapsed

    def can_afford(self, cost: float) -> bool:
        return self.left > cost

    def log_state(self, tag: str) -> None:
        log.info("[budget] %s: elapsed %.1fs / soft %.0fs (hard %.0fs), left %.1fs",
                 tag, self.elapsed, self.deadline, self.allowed, self.left)


def majority(answers: list[str]) -> str:
    """メンバー順で先勝ちの多数決。`answers[0]` が最優先メンバーの回答。"""
    cnt = Counter(answers)
    top = max(cnt.values())
    for a in answers:                 # 入力順 = メンバー順なので先勝ちになる
        if cnt[a] == top:
            return a
    return answers[0]


def vote_all(per_member: list[dict[str, str]], qids: list[str]) -> dict[str, str]:
    """メンバーごとの {qID: answer} を多数決に畳む。

    ★**そのメンバーが到達しなかった問は欠落している**（打ち切りで途中終了した pass）。
      欠落は投票から除外する ＝ その問だけ少ないメンバーで投票することになる。
    """
    out = {}
    for q in qids:
        cands = [m[q] for m in per_member if q in m]
        out[q] = majority(cands) if cands else ""
    return out
