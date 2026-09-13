r"""ORena SAVE FOCUS — PROCEDURE track container（QLoRA + group 別「枚数の梯子」/ anytime）.

Source: workspace/expM03_round1_external/results/expP05_warm_proc64f_alldata/fold0/adapter
        （★expE03f から warm start → **PROCEDURE のみ 2ep 追加学習**。
          学習データ = fold0 train 8,000 + **val 2,000** = 10,000 問（train_part=all）。
          LoRA r16/α32 / lr 3e-5 / 外部データなし・instseg 重畳なし。
          ★**学習時のフレーム選択を推論と同一**にした（combo2 c9w + anchor + 64枚 + C2_SKIP_ANON）
            ＝ v010 までは学習が一様16枚で、推論だけ索引を使うズレがあった）
CV: ★P05 自体は測れない（val が学習データ）。根拠は **fold 版 expP02 = 0.5934**
    （対照 `eval_expP00ctrl_E03f_skipanon` 0.5386 に対し ALL 正答率 +0.0327, 207/143, p=0.0007。
     主因は object_recognition +0.0750, 560問, p=0.0001）。P05 は train_part だけを変えた版。
動画側: workspace/expM03_round1_external/results/expP06_warm_video128_alldata/fold0/adapter
        （★expP03 の all-data 版 = `train_part` だけを変えた版。動画128枚@448。
          md5 f7faf42bd06b8de277b32ea8936aacc6 / 1,250 step / 4×RTX5090 6h47m）
⚠️**動画入力は単体では画像入力に負ける**（fold 版 expP03 0.5605 < expP02 0.5934、
   問単位 frame 勝ち235/video 勝ち119, p=7.0e-10。SEGMENT expS00 −0.0612, p=3.8e-21 と同じ向き）。
   採用しているのは**単体で強いから**ではなく、**2パスの選択器の一方として**である。

## v010（LB 0.5105）からの差分

1. **adapter を expP05 に差し替え**（上記）。v010 は expE03f そのままで、
   学習が一様16枚・推論だけ combo2 64枚という**ズレ**を抱えていた。
2. **匿名(単色)フレームをデコード後の画素から検出して置き換える**（`_replace_flat`）。
   学習は `C2_SKIP_ANON=1` だがコンテナは匿名区間表を持てない（本番動画に無い）。
   判定はチャンネル毎の面内 std < 2.0（1/8 間引き、1080p 64枚で 178ms）。
   置換は探索半径を倍々にして匿名区間を跨ぐ。再デコード上限は枚数の 2 倍。
   `FOCUS_FLAT_SKIP=0` で無効化できる（対照用）。
3. **EOS 対策**: 生成側 `stop_strings=["\n"]` + `answer_norm` の1行目切断。
   expP05 は warm start なので改行 0 件だが、base scratch 版は 90% が偽ターンを続けた
   実績があるため無害な保険として残す。

## 枚数は 64 固定（`FRAME_LADDER` 全 group `(64,)`）

⚠️v010 までの梯子（temporal 128f まで登る等）は **`ClipContext.times()` が枚数を
受け取らないバグで一度も動いておらず、実質 64 固定**だった。LB 0.5105 はその条件で出た値。
根拠だった 64f/96f の比較は**学習が一様16枚だった頃**の測定で、いまは前提が違う:

| 実測（2026-09-10） | 結果 |
|---|---|
| 動画128枚 vs 画像64枚（同一レシピ） | TEMPORAL 0.3173 vs **0.4148**（p=0.0022 で負け）|
| combo2 の推薦優位性（一様比） | 64枚 **+5.3pt** → 128枚 +0.3pt（**消える**）|
| 画像列128枚 | 16,651 tok で `max_seq_len` 超過（そもそも載らない）|

→ **学習が 64 枚なので 64 固定が正しい条件**。`select_frames.py` は枚数を受け取るよう
直してあるが、梯子は単段 `(64,)` にして上へ登らせない。

## 予算（2026-08-25 に SEGMENT 提出の metrics から確定）

```
batches 100 / questions 2000 / setup_allowance_s 120 / latency_per_question_s 30.0(PROCEDURE)
```
→ **1バッチ 20問、予算 120 + 20×30 = 720s**。
★SEGMENT 提出の実測は 173.7s / 420s ＝ **予算の 41% しか使っていなかった**
（1位ですら 46%）。**時間を使い切る設計は未開拓**なので、ここでは使い切りに行く。

## 何を「アンサンブル」するか — ★多数決ではなく **group 別の枚数振り分け**

⚠️ SEGMENT / PROCEDURE では **多数決は全構成で振り分けに負ける**（claudeSummary）。
そこで**同じモデルを1回だけ走らせ、group ごとに枚数を変える**。

| group | N | 64f | 96f | Δ(問数換算) | 採用 |
|---|---|---|---|---|---|
| temporal_grounding | 769 | 0.1782 | **0.2406** | **+48問** | 96f→**128f まで登る** |
| aggregation | 544 | 0.2463 | **0.2702** | **+13問** | 96f |
| object_recognition | 560 | **0.6089** | 0.5929 | −9問 | 64f |
| event_understanding | 37 | **0.7568** | 0.7027 | −2問(ノイズ) | 64f |
| complex_reasoning | 50 | **0.5400** | 0.5200 | −1問(ノイズ) | 64f |

★**64f/96f 単独はほぼ同点（0.4660/0.4653）なのに振り分けると 0.4833（+0.0173）**。
★★**temporal は枚数に単調で未飽和**（16f 0.0676→32f 0.1092→64f 0.1782→96f 0.2406）
  ＝ 予算が余る分をここに突っ込む（128f）。

## anytime（打ち切りではなく「梯子を降りる」）

実測スループットから **1問あたりに使える残り時間**を出し、各問で
`FRAME_LADDER` を上から降りて**入る枚数を選ぶ**。予算が尽きても
**必ず全問に回答を書く**（無回答はバッチごと Failed になる）。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ─────────────────────────────────────────────────────────────────────
# ★★★最優先の fail-safe: **重い import より前**に空の answer.json を置く。
#   grand-challenge は「出力なし」も「非ゼロ終了」も **提出ごと Failed** にする
#   （= その提出枠と image digest を消費する）。torch/transformers の import や
#   モデルロード中に SIGKILL（OOM・ハード時間切れ・基盤側の障害）を受けると
#   try/except は貫通されるので、**import 前に置いておくことだけが効く**。
#   本処理が正常に進めば同じパスへ上書きされるので、副作用は無い。
#   2026-08-27: FRAME(v008)/PROCEDURE(v009) がともに "The algorithm failed on
#   one or more cases" で落ちたことを受けて追加。
# ─────────────────────────────────────────────────────────────────────
def _prewrite_empty_answers() -> None:
    import json as _json
    import os as _os
    from pathlib import Path as _Path
    try:
        _in = _Path(_os.environ.get("FOCUS_INPUT", "/input")) / "request.json"
        _out = _Path(_os.environ.get("FOCUS_OUTPUT", "/output"))
        _out.mkdir(parents=True, exist_ok=True)
        _reqs = _json.loads(_in.read_text())
        _recs = [{"qID": str(r["qID"]), "content": "", "latency": 0.0} for r in _reqs]
        (_out / "answer.json").write_text(_json.dumps(_recs, indent=2))
    except Exception:                      # ここで落ちても本処理は続ける
        try:
            _out = _Path(_os.environ.get("FOCUS_OUTPUT", "/output"))
            _out.mkdir(parents=True, exist_ok=True)
            (_out / "answer.json").write_text("[]")
        except Exception:
            pass


_prewrite_empty_answers()

import torch
from focus import Request, Response, load_requests, save_items
from qwen_vl_utils import process_vision_info
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from answer_norm import normalize  # noqa: E402
from prompts_seg import build_system_prompt, set_fo_definitions  # noqa: E402
from router import GroupRouter, WIDTH_PX  # noqa: E402
from sampling import hhmmss, sample_times  # noqa: E402
from clip_reader import ClipReader  # noqa: E402
from select_frames import ClipContext  # noqa: E402
import fo_index  # noqa: E402  # noqa: E402
from time_postproc import TimePostproc  # noqa: E402
from video import (extract_all_keyframes, extract_frames_keyframes,  # noqa: E402
                   find_clip, keyframe_times)

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

INPUT_PATH = Path(os.environ.get("FOCUS_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("FOCUS_OUTPUT", "/output"))
RESOURCES = Path(__file__).parent / "resources"
BASE_MODEL_DIR = Path(os.environ.get("FOCUS_BASE_MODEL", RESOURCES / "base_model"))
ADAPTER_DIR = RESOURCES / "adapter"
# ★★2パス構成（frame 入力 / 動画入力の 2 モデルを走らせ confidence の高い方を採る）。
#   【fold0 val の実測 2026-09-11】両者が食い違う 1,097 問のうち「どちらか一方だけ正解」は 354 問。
#     常に frame を採ると 0.6412(227/354)、conf の高い方を採ると **0.6610(234/354)**。
#     1960 問の公式 SCORE は **0.5907 → 0.5982（+0.0075）**。
#   ⚠️**有意ではない**（McNemar 改善53/悪化46, p=0.547）。しかも SCORE 増分の 72% は
#     EVENT_UNDERSTANDING(n=37) の **1 問**。実際に選択器が効いている形式は fo_class
#     （0.596→0.697, +9問/89）だが、**fo_class だけに限ると SCORE は −0.0005**
#     ＝「効く領域」と「SCORE が動く領域」が一致していない。
#   ⚠️提出の組は P05(全データ・frame) × P03(fold・video) で**非対称**。全データ版の
#     動画モデルは存在しないので、実運用では測定時より差が開き選択器の効きは縮む。
#   → 期待値プラス側として**ユーザ判断で採用**（2026-09-11）。
ADAPTER_VIDEO_DIR = RESOURCES / "adapter_video"
TWO_PATH = os.environ.get("FOCUS_TWO_PATH", "1") == "1"
# 動画パスの枚数（トークンが約半分なので frame 64 と同程度の系列長になる）
VIDEO_FRAMES = int(os.environ.get("FOCUS_VIDEO_FRAMES", "128"))
# 動画パスの追加費用を frame パス実測の何倍とみなすか（索引は共有するので生成+デコードのみ）
VIDEO_COST_RATIO = float(os.environ.get("FOCUS_VIDEO_COST_RATIO", "0.8"))
ROUTER_TABLE = RESOURCES / "group_templates.json"
TIME_PRIOR = RESOURCES / "time_count_prior.json"

MAX_NEW_TOKENS = 48
MAX_NEW_TOKENS_RUSH = 16
SETUP_BUDGET_S = 120.0
# ★PROCEDURE の公式上限。**env は検証専用**（遅い GPU で 2 パス経路を踏ませるため）。
#   本番は env を渡さないので 30.0 のまま。
PER_QUESTION_BUDGET_S = float(os.environ.get("FOCUS_PER_Q_BUDGET_S", "30.0"))
SAFETY = float(os.environ.get("FOCUS_SAFETY", "0.85"))
GRID_S = 5.0                        # 本番クリップのキーフレーム間隔と一致（CV も 5.0）
USE_OVERLAY_FOR_TIME = os.environ.get("FOCUS_OVERLAY_TIME", "1") == "1"
# ★procedure_type を user ターン先頭に足す。CV では推論時のみ +0.023（学習に入れると −0.074）。
USE_PROCEDURE_TYPE = os.environ.get("FOCUS_PROCEDURE_TYPE", "1") == "1"
# 梯子を1段降りるかの判定に使う「枚数あたりコスト」の初期値（実測で上書きされる）
INIT_DECODE_S_PER_CLIP_S = 0.0035   # クリップ1秒あたりのデコード秒（実測 ~90 kf/s から）
INIT_GEN_S_PER_FRAME = 0.08         # 1枚あたりの生成秒


def write_answers(records: list[dict]) -> None:
    """★何があっても answer.json を残す（出力なし＝バッチごと Failed）。"""
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_items([Response(**r) for r in records], OUTPUT_PATH / "answer.json")


class Ladder:
    """実測スループットから「いま何枚まで払えるか」を決める。"""

    def __init__(self, n_questions: int):
        self.t0 = time.monotonic()
        self.allowed = SETUP_BUDGET_S + n_questions * PER_QUESTION_BUDGET_S
        self.deadline = self.allowed * SAFETY   # 20%超過でバッチ全滅なので手前に置く
        self.n = n_questions
        self.done = 0
        self.max_frames = 10**9                 # OOM を観測したら下げる
        # ★2項のコストモデル。どちらも実測で上書きされる
        self.dec_per_s = INIT_DECODE_S_PER_CLIP_S   # デコード: クリップ1秒あたり秒
        self.gen_per_frame = INIT_GEN_S_PER_FRAME   # 生成: 1枚あたり秒

    @property
    def left(self) -> float:
        return self.deadline - (time.monotonic() - self.t0)

    def est(self, nf: int, duration: float) -> float:
        """コストを **デコード（クリップ長に比例）+ 生成（枚数に比例）** の2項で見る。

        ★枚数だけでコストを測ると外れる。実測では同じ 64 枚でも 10.6s〜36.0s と
          3倍以上ばらついた。**支配項はクリップ長**（テストバッチで 619s〜16,189s）で、
          枚数ではない。1項モデルだと長尺クリップの費用を短尺にも見積もってしまい、
          梯子が不必要に降りる（実測: 96f が 20問中 1問しか出せなかった）。
        """
        return self.dec_per_s * duration + self.gen_per_frame * nf

    def observe(self, nf: int, duration: float, took: float) -> None:
        """1問の実測から2項を分離して更新する（生成側を先に引く）。"""
        gen = self.gen_per_frame * nf
        dec = max(took - gen, 0.0)
        if duration > 0:
            r = dec / duration
            self.dec_per_s = 0.7 * self.dec_per_s + 0.3 * r
        # 生成側は「短いクリップの問」からしか綺麗に取れないので緩やかに寄せる
        if duration < 900 and nf > 0:
            g = max(took - self.dec_per_s * duration, 0.0) / nf
            self.gen_per_frame = 0.8 * self.gen_per_frame + 0.2 * g
        self.done += 1

    def pick(self, ladder: tuple[int, ...], duration: float) -> tuple[int, float]:
        """残り予算・クリップ長・OOM 実績から、この問に使う枚数を選ぶ。"""
        remaining = max(self.n - self.done, 1)
        # ★★採点は **per-question**（30s 超の問は回答があっても不正解）。残り予算 ÷ 残り問数
        #   をそのまま使うと、前の問が速かったぶんを繰り越して **1 問に 30s 超を配る**
        #   （2026-09-11 の dl2 実測で 39〜44s を配っていた ＝ その問を確実に 0 点にする動き）。
        #   予算のプールは「バッチ全体を落とさない」ためには正しいが、**1 問の上限としては誤り**。
        per_q = min(self.left / remaining, PER_QUESTION_BUDGET_S)
        for nf in ladder:
            if nf > self.max_frames:             # ★OOM 実績のある枚数は最初から選ばない
                continue
            if self.est(nf, duration) <= per_q:
                return nf, per_q
        feasible = [nf for nf in ladder if nf <= self.max_frames]
        return (feasible[-1] if feasible else ladder[-1]), per_q

    def note_oom(self, nf: int) -> None:
        """★OOM した枚数以上は二度と使わない。

        128f は 24GB 級では載らない可能性があり、本番 L40S(48GB) でも未検証。
        1問 OOM しただけで temporal の全問を落とさないよう、**上限を学習して降りる**。
        """
        self.max_frames = min(self.max_frames, max(16, nf // 2))
        log.warning("OOM at %d frames → 以後の上限を %d 枚に下げる", nf, self.max_frames)


def load_model():
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(str(BASE_MODEL_DIR))
    model = AutoModelForImageTextToText.from_pretrained(
        str(BASE_MODEL_DIR), quantization_config=bnb, device_map="cuda:0",
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), adapter_name="frame")
    # ★★2パス構成: 動画入力で学習した adapter も同じ base に載せ、`set_adapter` で切り替える。
    #   base（19GB）は1回しか読まないので、増えるのは LoRA 分（225MB）だけ。
    if TWO_PATH and ADAPTER_VIDEO_DIR.exists():
        model.load_adapter(str(ADAPTER_VIDEO_DIR), adapter_name="video")
        log.info("★2パス構成: frame + video の adapter を両方ロードした")
    else:
        log.info("単一パス（frame のみ）")
    model.set_adapter("frame")
    model.eval()
    return model, processor


def build_messages(req: Request, times: list[float], images: list[Image.Image],
                   system_prompt: str) -> list[dict]:
    """★CV（`dataset_seg.build_messages`）と**同じ並び**にする。

    ⚠️冒頭の説明文（"Video frames sampled from ... (N frames):"）が抜けていると、
      LoRA が学習時と違う文脈を受け取り、**思考過程を出力し始める**
      （2026-09-01 実測: 20 問中 18 問が "The user wants to ..." で始まり、
       同じ問の CV 正答率 0.60 に対しコンテナは 0.10 まで落ちた）。
    ★`procedure_type` は **user ターンの先頭**に置く。LoRA は system の指示をほぼ無視する。
    """
    if len(images) == 1:
        head = f"Frame at [{hhmmss(times[0])}]:"
    else:
        head = (f"Video frames sampled from {hhmmss(req.start_time)} to "
                f"{hhmmss(req.end_time)} ({len(images)} frames):")
    if USE_PROCEDURE_TYPE and getattr(req, "procedure_type", ""):
        head = f"Procedure: {req.procedure_type}\n" + head
    content: list[dict] = [{"type": "text", "text": head}]
    for t, im in zip(times, images):
        # ★1枚のときは head が既に "Frame at [t]:" なので**時刻テキストを重ねない**（CV と同一）。
        if len(images) > 1:
            content.append({"type": "text", "text": f"[{hhmmss(t)}]"})
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": req.question})
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": content}]


@torch.no_grad()
def generate(model, processor, messages: list[dict], max_new_tokens: int) -> tuple[str, float]:
    # ★★`enable_thinking=False` が要る。省くと Qwen3.5 が思考モードで答え、
    #   "The user wants to ..." という**推論過程がそのまま回答本文になる**
    #   （2026-09-01 実測: 同じ 20 問で CV 0.60 に対しコンテナ 0.10）。
    #   CV(`run_infer.gen`) と**同じ 2 段構え**（テキスト化 → process_vision_info）に揃える。
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    ii, vi = process_vision_info(messages)
    inp = processor(text=[text], images=ii, videos=vi, return_tensors="pt").to(model.device)
    # ★★v016: EOS を打たずに偽ターンを続けるモデルなので **改行で生成を止める**。
    #   クリーンな答えには無影響で、暴走時だけ latency と末尾ゴミを断つ
    #   （answer_norm の1行目切断と二重の防御）。tokenizer を渡さないと
    #   stop_strings は**黙って無視される**。
    return _gen_with_conf(model, processor, inp, max_new_tokens)


def _gen_with_conf(model, processor, inp, max_new_tokens: int):
    """生成して `(text, conf)` を返す。conf は **1行目のトークンの平均 logprob**。

    ★偽ターンや余剰トークンを混ぜると confidence が薄まって 2 パスの比較が壊れるので、
      実際に採点される 1 行目だけで測る。greedy なので生成そのものは変わらない。
    """
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                         stop_strings=["\n"], tokenizer=processor.tokenizer,
                         output_scores=True, return_dict_in_generate=True)
    seq = out.sequences[0][inp.input_ids.shape[1]:]
    txt = processor.decode(seq, skip_special_tokens=True).strip()
    lps = []
    for i, sc in enumerate(out.scores):
        if i >= len(seq):
            break
        tok = int(seq[i])
        lps.append(float(torch.log_softmax(sc[0].float(), dim=-1)[tok]))
        if "\n" in processor.tokenizer.decode([tok]) and len(lps) > 1:
            break
    return txt, (sum(lps) / len(lps) if lps else float("-inf"))


FLAT_STD = float(os.environ.get("FOCUS_FLAT_STD", "2.0"))   # 単色判定の閾値
FLAT_SKIP = os.environ.get("FOCUS_FLAT_SKIP", "1") == "1"   # 0 で無効化（対照用）


def _is_flat(a: np.ndarray) -> bool:
    """匿名(単色)フレームか. ★**チャンネルごとの面内 std** で見る.

    平均色では判別できない（暗い術野の平均は青ブロックに近い）。
    画素は 1/8 に間引く（単色フレームはどこを見ても単色なので情報を失わない）。
    """
    s = a[::8, ::8, :].reshape(-1, 3).astype(np.float32).std(0)
    return bool((s < FLAT_STD).all())


FLAT_ROUNDS = int(os.environ.get("FOCUS_FLAT_ROUNDS", "8"))


def _replace_flat(reader, times: list[float], secs: list[float], arr: np.ndarray,
                  lo: float, hi: float, grid: float):
    """単色フレームを**同じ格子上の最も近い可視時刻**へ置き換える（枚数を保つ）.

    ★学習と CV は `C2_SKIP_ANON=1`（`dataset_seg.drop_anon_times`）で匿名時刻を外している。
      コンテナは匿名区間表を持てない（本番動画には無い）ので、**デコード後の画素から判定**して
      同じことをする。入れないと「学習は匿名を見ていない / 本番は見る」というズレが残る。
    ★★**1 巡では足りない**（2026-09-04 の合成テストで実測）: 匿名は**連続区間**なので、
      隣の格子点もほぼ確実に匿名で、10 枚中 9 枚が「置換先も単色」で落ちた。
      → **見つけた単色時刻を `bad` に貯めながら数巡回す**。区間幅を跨ぐまで外へ抜ける。
      再デコードは巡ごとに「まだ単色の枚数」だけなので、64 枚の再読み込みより安い。
    ★それでも抜けられない（クリップがほぼ全部匿名）ときだけ落とす。
    """
    flat = [i for i in range(len(secs)) if _is_flat(arr[i])]
    n_flat = len(flat)
    if not flat:
        return secs, arr, 0, 0
    secs = [float(t) for t in secs]
    arr = np.array(arr, copy=True)
    taken = {round(t, 3) for t in secs}
    bad = {round(secs[i], 3) for i in flat}     # 単色と分かっている時刻
    k_max = int((hi - lo) / grid) if grid > 0 else 0

    pending = list(flat)
    radius = {i: 1 for i in flat}    # ★次に探し始める格子距離（失敗するたび倍にする）
    # ★latency 保護: 再デコードの総枚数に上限を置く（実測 最悪 143 枚 = 元の 2.2 倍）。
    #   予算は 41% しか使っていないが、青が長いクリップで青天井にはしない。
    budget = 2 * len(secs)
    for _ in range(max(1, FLAT_ROUNDS)):
        if not pending or budget <= 0:
            break
        want: dict[int, float] = {}
        for i in pending:
            k0 = int(round((secs[i] - lo) / grid)) if grid > 0 else 0
            # ★★線形に 1 格子ずつ外へ出ても匿名区間は抜けられない（区間は連続で、
            #   数百秒に及ぶ。2026-09-04 の合成テストで一様64枚が 8 枚落ちた）。
            #   → 失敗するたび探索半径を**倍**にして、区間幅を対数回で跨ぐ。
            for d in range(radius[i], k_max + 1):
                hit = None
                for k in (k0 - d, k0 + d):
                    if not (0 <= k <= k_max):
                        continue
                    c = round(lo + k * grid, 3)
                    if c in taken or c in bad:
                        continue
                    hit = c
                    break
                if hit is not None:
                    taken.discard(round(secs[i], 3))
                    taken.add(hit)
                    want[i] = hit
                    radius[i] = max(2 * d, d + 1)
                    break
        if not want:
            break                                # 候補が尽きた
        idx = list(want)[:max(1, budget)]
        budget -= len(idx)
        new_secs, new_arr = reader.read([want[i] for i in idx])
        nxt = [i for i in want if i not in set(idx)]   # 予算で切った分は次巡へ
        for j, i in enumerate(idx):
            if j >= len(new_arr):
                nxt.append(i)
                continue
            secs[i] = float(new_secs[j])
            arr[i] = new_arr[j]
            if _is_flat(new_arr[j]):
                bad.add(round(float(new_secs[j]), 3))
                nxt.append(i)                    # まだ単色 → 次の巡で更に外へ
        pending = nxt

    if pending:                                  # 抜けられなかった枚だけ落とす
        keep = [i for i in range(len(secs)) if i not in set(pending)]
        secs = [secs[i] for i in keep]
        arr = arr[keep]
    return secs, arr, n_flat, len(pending)


# ─────────────────────────────────────────────────────────────────────
# ★★動画入力パス（expP01 の `video_input.py` と**同じ規約**にする）
#   ① `temporal_patch_size=2` で 2 枚が 1 時間パッチに畳まれ、トークンが約半分になる
#   ② フレームは (i,i+1) で対にされ、プロンプトの時刻は**対の中点**
#      → **必ず時刻昇順で渡す**（バラバラだと無意味な時刻になる）
#   ③ `do_sample_frames=False` を渡さないと processor が **fps=24 で勝手に間引く**
#      （64枚が実質数枚になり、トークンが 375 まで落ちる）
#   ④ `VideoMetadata.frames_indices` に元動画のフレーム番号を渡すと
#      combo2 の**非一様な選択のまま**正しい絶対時刻が出る
# ─────────────────────────────────────────────────────────────────────
META_FPS = 1000.0        # 時刻の分解能を決めるだけ。実 fps と一致する必要はない


def _read_video_frames(reader, ctx, req, fmt):
    """動画パス用に **VIDEO_FRAMES 枚**を読む（frame パスとは枚数が違う）。

    ★索引(`ClipContext`)と `ClipReader` は frame パスが作ったものを**そのまま使い回す**。
      以前は自前で `ClipContext.from_reader` を呼んでおり、**予算の大半を占める検出器を
      1 問につき 2 回**走らせていた（日報 2026-09-10「latency 予算は検出器が大半」）。
      デコードだけなら追加費用は枚数に比例する分で済む。
    """
    times = (ctx.times(fmt, VIDEO_FRAMES) if ctx is not None
             else sample_times(req.start_time, req.end_time, VIDEO_FRAMES, GRID_S, None))
    times = sorted(float(t) for t in times)      # ★昇順が必須（上記②）
    secs, arr = reader.read(times)
    if FLAT_SKIP and len(secs):
        secs, arr, n_flat, n_lost = _replace_flat(
            reader, times, secs, arr, float(req.start_time), float(req.end_time), GRID_S)
        if n_flat:
            log.info("匿名(単色) video qID=%s: 検出 %d 落とし %d", req.qID, n_flat, n_lost)
    return list(secs), arr


def _video_inputs(processor, req, secs, arr, system_prompt: str, width: int):
    """動画としての processor 入力を作る。head/質問の並びは frame パスと同一にする。"""
    from transformers.video_utils import VideoMetadata
    ims = []
    for a in arr:
        im = Image.fromarray(a)
        if im.width != width:
            im = im.resize((width, max(2, round(im.height * width / im.width))), Image.BICUBIC)
        ims.append(np.asarray(im))
    vid = np.stack(ims)
    n = len(secs)
    head = (f"Video frames sampled from {hhmmss(req.start_time)} to "
            f"{hhmmss(req.end_time)} ({n} frames):")
    if USE_PROCEDURE_TYPE and getattr(req, "procedure_type", ""):
        head = f"Procedure: {req.procedure_type}\n" + head
    mids = [(secs[i] + secs[i + 1]) / 2 for i in range(0, n - 1, 2)]
    if n % 2:
        mids.append(secs[-1])
    content = [{"type": "text", "text": head}, {"type": "video"},
               {"type": "text", "text": "Frame timestamps in hh:mm:ss order: "
                                        + ", ".join(hhmmss(m) for m in mids)},
               {"type": "text", "text": req.question}]
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": content}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    idx = [int(round(float(t) * META_FPS)) for t in secs]
    meta = VideoMetadata(
        total_num_frames=max(int(float(req.end_time) * META_FPS), (idx[-1] if idx else 0) + 1),
        fps=META_FPS, width=int(vid.shape[2]), height=int(vid.shape[1]),
        duration=float(req.end_time), video_backend="decord", frames_indices=idx)
    return processor(text=[text], videos=[vid], video_metadata=[meta],
                     return_tensors="pt", do_sample_frames=False)


def answer_one(model, processor, req: Request, n_frames: int, width: int,
               system_prompt: str, fmt: str, tp: TimePostproc,
               max_new_tokens: int, det=None, index_budget_s: float | None = None,
               two_path_room_s: float = 0.0) -> tuple[str, int]:
    """★v010: 一様サンプリングを **FO 索引 + 質問テンプレ別ルール**に差し替えた。

    デコードは 1 回（全キーフレーム）だけ行い、索引・工程・VLM 入力で使い回す。
    索引が作れなければ従来どおり一様に倒れる（壊れずに CV 相当へ落ちる）。
    """
    # ★`time` 形式は**時計が焼き込まれた overlay 版**を使う（env で切替可）。
    #   ⚠️学習・CV は素の映像＋テキスト時刻なので、入力分布が変わる。効果は CV で確認すること。
    use_ov = USE_OVERLAY_FOR_TIME and fmt == "time"
    clip = find_clip(INPUT_PATH, req.qID, overlay=use_ov)
    if clip is None:
        raise FileNotFoundError(f"no clip for qID={req.qID}")
    # ★2 段構成（ユーザ設計 2026-09-01）
    #   ① 推薦段階: 15 秒刻みの粗いフレームを **1 回だけ**読み、索引(M2F)と工程(CNN)で共有
    #   ② VLM 段階: ルールが 64 枚の時刻を決めてから、その時刻だけを**もう 1 回**読む
    #   decord の一括読みは「要る index だけ」なので、全キーフレームを展開するより安い。
    #   時刻は index/fps で厳密に決まるので、キーフレーム時刻の推測が要らない。
    reader = ClipReader(clip, req.start_time)
    try:
        _t0 = time.monotonic()
        ctx = ClipContext.from_reader(req, det, reader, index_budget_s, fmt) if det is not None else None
        if ctx is not None:
            log.info("推薦段階 qID=%s: %.1fs", req.qID, time.monotonic() - _t0)
            # ★v016: 梯子が選んだ枚数を索引経路へも通す（v010 は渡しておらず常に 64 枚だった）
            times = ctx.times(fmt, n_frames)
        else:
            times = sample_times(req.start_time, req.end_time, n_frames, GRID_S, None)
        _t1 = time.monotonic()
        secs, arr = reader.read(times)          # ★VLM に渡す分だけを読む
        if FLAT_SKIP and len(secs):
            secs, arr, n_flat, n_lost = _replace_flat(
                reader, times, secs, arr, float(req.start_time), float(req.end_time), GRID_S)
            if n_flat:
                log.info("匿名(単色)フレーム qID=%s: 検出 %d 置換 %d 落とし %d",
                         req.qID, n_flat, n_flat - n_lost, n_lost)
        log.info("VLM用フレーム qID=%s: %d枚 %.1fs", req.qID, len(secs), time.monotonic() - _t1)
        keep = []
        for t, a in zip(secs, arr):
            im = Image.fromarray(a)
            if im.width != width:               # VLM へは学習時と同じ幅で渡す
                im = im.resize((width, max(2, round(im.height * width / im.width))), Image.BICUBIC)
            keep.append((t, im))
        if not keep:
            raise RuntimeError(f"no frames decoded for qID={req.qID}")
        msgs = build_messages(req, [t for t, _ in keep], [im for _, im in keep], system_prompt)
        _t1 = time.monotonic()
        model.set_adapter("frame")
        raw, conf_f = generate(model, processor, msgs, max_new_tokens)
        log.info("VLM(frame) qID=%s: %d枚 %.1fs conf=%.3f",
                 req.qID, len(keep), time.monotonic() - _t1, conf_f)

        # ★★2パス: 動画入力で学習した adapter でも解き、**confidence の高い方**を採る。
        #   ⚠️2 モデルの logprob は較正されていないので、素の比較が当たる保証はない。
        #     fold0 val の実測では 354 問で 0.6412 → 0.6610（有意ではない。冒頭の注記を参照）。
        #   ★予算管理: `two_path_room_s` に「この問に残っている秒数」を渡す。frame パスに
        #     使った時間より残りが少なければ**動画パスに入らない**（anytime を壊さない）。
        #     ★reader と ctx は frame パスのものを使い回す（検出器を2回走らせない）。
        spent = time.monotonic() - _t0
        if TWO_PATH and "video" in getattr(model, "peft_config", {}):
            # ★既定(0.0)と OOM 再試行(-1.0)はどちらも「入らない」側に倒す
            if two_path_room_s <= 0 or two_path_room_s - spent < spent * VIDEO_COST_RATIO:
                log.info("2パス省略 qID=%s: 予算 %.1fs 経過 %.1fs 見積 %.1fs",
                         req.qID, two_path_room_s, spent, spent * VIDEO_COST_RATIO)
            else:
                try:
                    _t2 = time.monotonic()
                    v_secs, v_arr = _read_video_frames(reader, ctx, req, fmt)
                    if v_secs:
                        model.set_adapter("video")
                        inp = _video_inputs(processor, req, v_secs, v_arr, system_prompt, width)
                        raw_v, conf_v = _gen_with_conf(model, processor, inp.to(model.device),
                                                       max_new_tokens)
                        log.info("VLM(video) qID=%s: %d枚 %.1fs conf=%.3f",
                                 req.qID, len(v_secs), time.monotonic() - _t2, conf_v)
                        if conf_v > conf_f:
                            raw = raw_v
                            log.info("  → video を採用 (%.3f > %.3f)", conf_v, conf_f)
                except Exception as e:              # 2パスの失敗で1問も落とさない
                    log.warning("video パス失敗 qID=%s: %s（frame の答えを使う）", req.qID, e)
                finally:
                    model.set_adapter("frame")
    finally:
        reader.close()
    # ★CV の採点直前処理をそのまま移植する（入れないと CV と LB がズレる）:
    #   time は個数をテンプレ事前分布に合わせて中央へ寄せ、[start,end] にクランプ
    if fmt == "time":
        raw = tp.apply(raw, req.question, req.start_time, req.end_time)
    return normalize(fmt, raw), len(keep)


def warmup(model, processor) -> None:
    im = Image.new("RGB", (448, 252), (32, 32, 32))
    msgs = [{"role": "system", "content": "You are a surgical assistant."},
            {"role": "user", "content": [{"type": "text", "text": "[00:00:00]"},
                                         {"type": "image", "image": im},
                                         {"type": "text", "text": "Answer yes or no. Ready?"}]}]
    generate(model, processor, msgs, 4)


def run() -> int:
    log.info("=== ORena FOCUS PROCEDURE — group 別 枚数梯子 / anytime start ===")
    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json に問が無い")
        write_answers([])
        return 0
    n = len(requests)

    # ★何が起きても回答が残るよう、最初に空回答を書いておく
    records = [{"qID": r.qID, "content": "", "latency": 0.0} for r in requests]
    write_answers(records)

    lad = Ladder(n)
    log.info("%d問 / 予算 %.0fs（自前締切 %.0fs, safety %.2f）",
             n, lad.allowed, lad.deadline, SAFETY)

    set_fo_definitions(json.loads((INPUT_PATH / "FO_definitions.json").read_text()))
    router = GroupRouter.load(ROUTER_TABLE)
    tp = TimePostproc.load(TIME_PRIOR)
    model, processor = load_model()
    warmup(model, processor)
    # ★FO 索引の検出器はバッチで 1 回だけ読む（TRT があれば TRT、無ければ PyTorch）。
    #   読めなければ det=None ＝ 従来どおりの一様サンプリングへ静かに倒れる。
    det = None
    if os.environ.get("FOCUS_USE_INDEX", "1") == "1":
        try:
            det = fo_index.Detector()
        except Exception as e:                       # noqa: BLE001
            log.warning("索引の検出器を読めない（一様サンプリングで続行）: %s", e)
    log.info("model ready: elapsed %.1fs / 残り %.1fs", lad.allowed * SAFETY - lad.left, lad.left)

    used: dict[int, int] = {}
    for i, req in enumerate(requests):
        t_q = time.monotonic()
        ladder = router.ladder_for(req.question)
        dur = max(float(req.end_time) - float(req.start_time), 1.0)
        nf, per_q = lad.pick(ladder, dur)
        rush = lad.left < per_q * 0.5
        try:
            sysmsg, fmt = build_system_prompt(req.question)
            ans, got = answer_one(model, processor, req, nf, WIDTH_PX, sysmsg, fmt, tp,
                                  MAX_NEW_TOKENS_RUSH if rush else MAX_NEW_TOKENS,
                                  det=det, index_budget_s=max(per_q * 0.5, 3.0),
                                  # ★rush 時は 2 パスに入らない（frame の答えを確実に出す）
                                  two_path_room_s=0.0 if rush else per_q)
            records[i]["content"] = ans
        except torch.cuda.OutOfMemoryError:
            # ★枚数を下げて**同じ問をもう一度**試す（そのまま空回答にしない）
            lad.note_oom(nf)
            torch.cuda.empty_cache()
            try:
                nf2, _ = lad.pick(ladder, dur)
                sysmsg, fmt = build_system_prompt(req.question)
                ans, got = answer_one(model, processor, req, nf2, WIDTH_PX, sysmsg, fmt, tp,
                                      MAX_NEW_TOKENS_RUSH, det=det,
                                      index_budget_s=max(per_q * 0.3, 2.0),
                                      two_path_room_s=-1.0)   # ★OOM 後は 2 パス禁止
                records[i]["content"] = ans
                nf = nf2
            except Exception as e2:
                log.warning("qID=%s OOM 後の再試行も失敗: %s", req.qID, e2)
                got = 0
        except Exception as e:                    # 1問の失敗でバッチを落とさない
            log.warning("qID=%s 失敗: %s", req.qID, e)
            got = 0
        took = time.monotonic() - t_q
        records[i]["latency"] = round(took, 3)
        lad.observe(nf, dur, took)
        used[nf] = used.get(nf, 0) + 1
        if (i + 1) % 5 == 0 or i == n - 1:
            write_answers(records)                # 途中経過も落とさない
            log.info("[%d/%d] %d枚 %.1fs | 残り %.0fs | 枚数内訳 %s",
                     i + 1, n, nf, took, lad.left, dict(sorted(used.items())))

    write_answers(records)
    filled = sum(1 for r in records if r["content"])
    log.info("done: %d/%d 回答  経過 %.0fs / 予算 %.0fs（%.0f%%） 枚数内訳 %s",
             filled, n, lad.allowed * SAFETY - lad.left + 0.0, lad.allowed,
             100 * (lad.allowed * SAFETY - lad.left) / lad.allowed, dict(sorted(used.items())))
    if filled < n:
        log.warning("★空回答が %d 件ある（不正解扱い）", n - filled)
    return 0


def main() -> int:
    try:
        return run()
    except Exception:
        log.exception("致命的エラー — 空回答で終了する（出力なしは Failed になる）")
        try:
            reqs = load_requests(INPUT_PATH / "request.json")
            write_answers([{"qID": r.qID, "content": "", "latency": 0.0} for r in reqs])
        except Exception:
            OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
            (OUTPUT_PATH / "answer.json").write_text("[]")
        return 0


if __name__ == "__main__":
    sys.exit(main())
