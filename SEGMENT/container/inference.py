"""ORena SAVE FOCUS — SEGMENT v019: 2モデル confidence 選択 (expN04 + [expR00C | expR00], 重畳なし).
★構成: 1st=expN04(v015) 共通 / 2nd=ID→expR00C(FOCUS-ft all-data), OOD→expR00 / 全問 2 パス, logp_mean 大小比較
★元の v018 docstring:

CV (fold v003 fold0 val, N=1958, judge 込み, 重複動画 0027 除外):
  64f@560 単独 0.7113 / 32f@768 単独 0.7197 / **group 振り分け 0.7345**

group ごとに入力構成を変えるのが要点（同じ vision token 予算でも最適点が違う）:

  | group               | N   | 64f@560 | 32f@768 |
  |---------------------|-----|---------|---------|
  | object_recognition  | 850 | 0.8247  | 0.8047  |
  | temporal_grounding  | 795 | 0.5962  | 0.5421  |
  | aggregation         | 186 | 0.5000  | 0.5376  |
  | event_understanding |  74 | 0.8243  | 0.8649  |
  | complex_reasoning   |  53 | 0.8113  | 0.8491  |

group は `Request` に入っていないので**質問文テンプレートから推定**する（`router.py`）。
val 動画を除いた表で検証したところ **オラクル振り分けと同じ 0.7345** に到達した
（一致率 0.978 / 未知テンプレ 3.1%）。

Source:
  - LoRA: workspace/expE01_segproc_baseline/results/expE06g_joint_frame_segment_FULL/fold0/adapter
          （FRAME 15,992 + SEGMENT 15,994 の joint 全量 / 448px / 1ep / fold v003）
  - プロンプト: workspace/expE01_segproc_baseline/prompts_seg.py（同梱）
  - 後処理:     同 answer_norm.py / time_postproc.py + time_count_prior.json（同梱）

★後処理をコンテナ内に入れてある理由: ローカル CV では `eval_seg.py` が採点直前に
  正規化と time 個数補正を掛けている。**同じ処理を提出側でも掛けないと CV と LB がズレる**
  （time は SEGMENT の 39% を占める）。

★★fail-safe（2026-08-15 追加。2026-08-14 の Failed を受けて）:
  grand-challenge は `answer.json` が無いと "The algorithm failed on one or more cases."
  ＝ **提出そのものが Failed** になる。空回答なら 0 点で済むので、**必ず answer.json を残す**:

    1. **重い import より前に**全 qID の空回答 `answer.json` を書く
       （8/14 の実害は最初の CUDA 演算だったが、import 時に落ちる経路も同じ扱いにする）
    2. 1問終わるたびに書き直す（SIGKILL / OOM-kill は捕まえられないので**逐次保存**が要る）
    3. 例外は全部飲んで **exit code 0** で終える。非ゼロで終わると answer.json があっても Failed
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

INPUT_PATH = Path(os.environ.get("FOCUS_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("FOCUS_OUTPUT", "/output"))
RESOURCES = Path(__file__).parent / "resources"
BASE_MODEL_DIR = Path(os.environ.get("FOCUS_BASE_MODEL", RESOURCES / "base_model"))
# ★v019: 2モデル confidence 選択。1st は共通（expN04 = v015）、2nd は術式で切替
#   ID（学習済み4術式）→ expR00C（expR00 + FOCUS-ft all-data）/ OOD → expR00（外部データ込み）
ADAPTER_1ST  = Path(os.environ.get("FOCUS_ADAPTER_1ST",  RESOURCES / "adapter_n04"))
ADAPTER_ID   = Path(os.environ.get("FOCUS_ADAPTER_ID",   RESOURCES / "adapter_r00c"))
ADAPTER_OOD  = Path(os.environ.get("FOCUS_ADAPTER_OOD",  RESOURCES / "adapter_r00"))
# ★学習データに存在する procedure_type（fold0 全 20,000 問で実測、この4種のみ）。これ以外は OOD 扱い
ID_PROCEDURES = frozenset({"Laparoscopic Cholecystectomy", "Proctocolectomy",
                           "Rectal Resection", "Sigmoid Resection"})
# ★選択規則: 長さ正規化 log-prob（logp_mean）の大小比較。閾値なし＝較正不要（expR00 A/B で 65.9% 的中）
CONF_KEY = os.environ.get("FOCUS_CONF_KEY", "logp_mean")
TWO_PASS = os.environ.get("FOCUS_TWO_PASS", "1") != "0"   # 0 なら 2nd model 単独（予算逼迫の退避）
ROUTER_TABLE = RESOURCES / "group_templates.json"
TIME_PRIOR = Path(__file__).parent / "time_count_prior.json"
ANSWER_PATH = OUTPUT_PATH / "answer.json"

# ── ★★instseg 重畳（expN00, 2026-09-03 採用）────────────────────────────
#   CV 実測（SEGMENT val 3,925問, 32f@768+anchor, judge込み, 対照は同一マシン再走）:
#     ctrl(expE06g) 0.6842 / s0 0.6381 / **s1 0.7047 (+0.0205 vs ctrl, p=0.0015)**
#     s1rules 0.7065 (+0.0223, p=0.0002) / s2rules 0.6828（塗りは害）
#   → **s1（bbox+ラベル・塗りなし）を in-place で描く**。学習で見た描画そのもの。
# ★expR00 は **s3**（bbox・クラス色・conf・番号なし）で学習した（`two_way` で s0:s3 = 1:1）。
#   v013 は s1（単色 bbox+ラベル）だったので既定を変える。"s0" で無効化（対照用）。
# ★★検出を間引く stride。**既定 1 = 全フレームに両検出器**（cascade も不採用）。
#   本番 L40S の実測から: 両検出器 125 ms/枚（v013 が 59.0 枚平均で 14.25 s/問だったことから逆算）。
#   v018 は平均 63.1 枚（A 97.3% / B 2.7%）→ 検出器 7.87 + モデル 6.89 = **14.76 s/問 = 予算 96.9%**。
#   ★rush（16f@448 への退避）は「残り予算÷残り問数 < 11.25s」で発動するが、14.76 < 15 なので
#     per_q_left は単調に増え **発動しない**。実効 15.2 s/問 を超えて初めて後半が rush に落ちる。
#   ★v013 は 93.5% で forfeit 0 で着地している。緊急時のみ 2 に上げれば 71% まで落とせる。
# ★質問文ルール（expN00 §segrules）。`between T1 and T2` / `at T ... When is it retrieved`
#   の窓へフレーム予算を寄せる。CV: s0 で +0.0174 / s1 で +0.0018。発火 367/4,006 問。
USE_SEGRULES = os.environ.get("FOCUS_SEGRULES", "1") != "0"

MAX_NEW_TOKENS = 48            # run_infer.py の既定と一致（CV と同条件）
MAX_NEW_TOKENS_RUSH = 16
RUSH_CONFIG = (16, 448)        # 予算逼迫時に落とす構成（枚数・幅）
SETUP_BUDGET_S = 120.0
PER_QUESTION_BUDGET_S = 15.0   # SEGMENT
SAFETY_MARGIN = 0.90           # 2 パス継続の下限（予算比）。下回ったら 2nd model 単独へ
SAFETY_MARGIN_SINGLE = 0.50    # さらに下回ったら rush 構成へ


# ── fail-safe I/O（`focus` にも `torch` にも依存しない。壊れていても書けること） ──────
def read_qids_raw(path: Path) -> list[str]:
    """request.json から qID だけを拾う.

    ★`focus.load_requests` は `Request(**row)` なので**未知キーが1つ増えただけで落ちる**。
      fail-safe の入口でそれに巻き込まれたくないので、素の json で読む。
    """
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("items", rows.get("requests", []))
    return [str(r["qID"]) for r in rows if isinstance(r, dict) and "qID" in r]


def write_answers(records: list[dict]) -> None:
    """answer.json をアトミックに書く（tmp → rename）.

    ★書式は `focus.save_items` の Response 出力と同一（`indent=2` / `ensure_ascii=False` /
      キーは qID・content・latency）。**`focus` が import できない状況でも書きたい**ので
      save_items は使わない。
    ★rename にするのは、逐次保存中に kill されても**半端な JSON を残さない**ため。
    """
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH / ".answer.json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp.replace(ANSWER_PATH)


# ★★重い import の**前に**空回答を書く。ここから下の import が落ちても提出は成立する。
_QIDS: list[str] = []
if __name__ == "__main__":
    try:
        _QIDS = read_qids_raw(INPUT_PATH / "request.json")
        write_answers([{"qID": q, "content": "", "latency": 0.0} for q in _QIDS])
        log.info("fail-safe: wrote %d empty answer(s) to %s before loading anything",
                 len(_QIDS), ANSWER_PATH)
    except Exception:
        log.exception("fail-safe bootstrap failed — answer.json may be missing")

# ★import 自体の失敗も Failed にしない（ここで raise させると exit code が非ゼロになる）
_IMPORT_ERROR: BaseException | None = None
try:
    import torch
    from focus import Request, load_requests
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent))
    from answer_norm import normalize  # noqa: E402
    from prompts_seg import build_system_prompt, set_fo_definitions  # noqa: E402
    from router import GroupRouter  # noqa: E402
    from sampling import (SEGMENT_GRID_S, hhmmss, question_anchor_times,  # noqa: E402
                          sample_times, segrules_times)
    from time_postproc import TimePostproc  # noqa: E402
    from video import extract_frames, find_clip  # noqa: E402
except BaseException as exc:  # noqa: BLE001  — ここで死ぬと提出ごと落ちる
    _IMPORT_ERROR = exc


def load_model():
    """base(4bit NF4) を1回ロードし、LoRA adapter 3本を**名前付き**で載せる（set_adapter で切替）."""
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(str(BASE_MODEL_DIR))
    model = AutoModelForImageTextToText.from_pretrained(
        str(BASE_MODEL_DIR), quantization_config=bnb, device_map="cuda",
        dtype=torch.bfloat16).eval()
    # ★4bit ベースに載せた LoRA を merge_and_unload してはいけない（黙って捨てられる）
    for name, d in (("first", ADAPTER_1ST), ("id", ADAPTER_ID), ("ood", ADAPTER_OOD)):
        if not d.exists():
            raise FileNotFoundError(f"adapter '{name}' が無い: {d}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_1ST), adapter_name="first")
    model.load_adapter(str(ADAPTER_ID), adapter_name="id")
    model.load_adapter(str(ADAPTER_OOD), adapter_name="ood")
    model.set_adapter("first"); model.eval()
    log.info("LoRA adapters loaded: first=%s id=%s ood=%s", ADAPTER_1ST, ADAPTER_ID, ADAPTER_OOD)
    return model, processor


def build_messages(req: Request, times: list[float], images: list[Image.Image],
                   system_prompt: str, overlay_note: str = "") -> list[dict]:
    """`dataset_seg.build_messages` と同一のレイアウト（時刻テキスト → 画像 の交互）。"""
    head = (f"Video frames sampled from {hhmmss(req.start_time)} to "
            f"{hhmmss(req.end_time)} ({len(images)} frames):")
    content: list[dict] = [{"type": "text", "text": head}]
    for t, im in zip(times, images):
        content.append({"type": "text", "text": f"[{hhmmss(t)}]"})
        content.append({"type": "image", "image": im})
    # ★重畳を描いたときは説明文を**質問の直前**に置く（`dataset_seg.build_messages` の
    #   note-only 分岐と同一レイアウト。画像は増やさない = in-place なので token 増ゼロ）
    if overlay_note:
        content.append({"type": "text", "text": overlay_note})
    content.append({"type": "text", "text": req.question})
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": content}]


def generate(model, processor, messages: list[dict], max_new_tokens: int) -> str:
    return generate_conf(model, processor, messages, max_new_tokens)[0]


def generate_conf(model, processor, messages: list[dict], max_new_tokens: int
                  ) -> tuple[str, dict]:
    """greedy 生成 + 生成トークンの log-prob 統計（`run_infer.py --with-conf` と同一の計算）."""
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    ii, vi = process_vision_info(messages)
    inp = processor(text=[text], images=ii, videos=vi, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                           return_dict_in_generate=True, output_scores=True)
        ts = model.compute_transition_scores(o.sequences, o.scores, normalize_logits=True)[0]
    ts = ts[torch.isfinite(ts)]
    n = int(ts.numel())
    conf = {"logp_sum": float(ts.sum()) if n else 0.0,
            "logp_mean": float(ts.mean()) if n else 0.0,
            "logp_min": float(ts.min()) if n else 0.0, "n_gen_tokens": n}
    txt = processor.decode(o.sequences[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return txt, conf


def postprocess(fmt: str, raw: str, req: Request, tp: TimePostproc) -> str:
    """CV の採点直前に掛けている処理と**同一**にする（`eval_seg.py` 参照）。

    - time: 個数をテンプレ事前分布に合わせ、[start, end] にクランプ
    - それ以外: `verify()` を通す形へ正規化（`'1.'` → `'1'`, `'Yes.'` → `'yes'` 等）
      open_ended / multiple_choice は judge 採点なので触らない
    """
    if fmt == "time":
        return tp.apply(raw, req.question, req.start_time, req.end_time)
    return normalize(fmt, raw)


def is_ood(req: Request) -> bool:
    return str(getattr(req, "procedure_type", "") or "").strip() not in ID_PROCEDURES


def answer_one(model, processor, req: Request, n_frames: int, width: int,
               max_new_tokens: int, tp: TimePostproc, two_pass: bool = True
               ) -> tuple[str, int, dict]:
    system_prompt, fmt = build_system_prompt(req.question)
    # ★質問文ルール（segrules）: 発火したら窓を絞る。非発火なら従来と**ビット同一**
    times = None
    if USE_SEGRULES:
        times = segrules_times(req.question, req.start_time, req.end_time,
                               n_frames, SEGMENT_GRID_S)
    if times is None:
        times = sample_times(req.start_time, req.end_time, n_frames, grid=SEGMENT_GRID_S,
                             anchors=question_anchor_times(req.question))
    clip = find_clip(INPUT_PATH, req.qID)
    if clip is None:
        raise FileNotFoundError(f"no clip for qID={req.qID} under {INPUT_PATH}")
    # ★クリップは窓に切り出し済み。絶対時刻 → クリップ相対へ直してから読む
    rel = [round(t - req.start_time, 3) for t in times]
    frames = extract_frames(clip, rel, width)          # ★デコードは 1 回（2 パスで共有）
    keep = [(t, frames[r]) for t, r in zip(times, rel) if r in frames]
    if not keep:
        raise RuntimeError(f"no frames decoded for qID={req.qID} ({clip})")
    msgs = build_messages(req, [t for t, _ in keep], [im for _, im in keep], system_prompt)
    second = "ood" if is_ood(req) else "id"
    info = {"second": second, "picked": second}
    if not two_pass:
        model.set_adapter(second)
        raw, c2 = generate_conf(model, processor, msgs, max_new_tokens)
        info.update(conf2=c2)
        return postprocess(fmt, raw, req, tp), len(keep), info
    model.set_adapter("first")
    raw1, c1 = generate_conf(model, processor, msgs, max_new_tokens)
    model.set_adapter(second)
    raw2, c2 = generate_conf(model, processor, msgs, max_new_tokens)
    # ★閾値なしの大小比較。回答が一致していれば選択は無関係（どちらでも同じ）
    pick_first = c1.get(CONF_KEY, 0.0) > c2.get(CONF_KEY, 0.0)
    raw = raw1 if pick_first else raw2
    info.update(picked="first" if pick_first else second, agree=(raw1.strip() == raw2.strip()),
                conf1=c1, conf2=c2, raw1=raw1, raw2=raw2)
    return postprocess(fmt, raw, req, tp), len(keep), info


def warmup(model, processor) -> None:
    """初回生成はカーネルコンパイルで遅い。setup 予算のうちに済ませる。"""
    im = Image.new("RGB", (448, 252), (32, 32, 32))
    msgs = [{"role": "system", "content": "You are a surgical assistant."},
            {"role": "user", "content": [{"type": "text", "text": "[00:00:00]"},
                                         {"type": "image", "image": im},
                                         {"type": "text", "text": "Answer with yes or no. Ready?"}]}]
    generate(model, processor, msgs, 4)


def run() -> int:
    t_start = time.monotonic()
    log.info("=== ORena FOCUS SEGMENT — QLoRA group-routed inference start ===")
    log.info("CUDA available: %s (%s)", torch.cuda.is_available(),
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json contains no requests — nothing to answer")
        return 0
    n = len(requests)
    budget_total = SETUP_BUDGET_S + n * PER_QUESTION_BUDGET_S
    log.info("Batch of %d question(s); pooled budget %.0fs", n, budget_total)

    fo_text = json.loads((INPUT_PATH / "FO_definitions.json").read_text())
    set_fo_definitions(fo_text)
    log.info("FO definitions: %d chars", len(fo_text))

    router = GroupRouter.load(ROUTER_TABLE)
    tp = TimePostproc.load(TIME_PRIOR)
    log.info("router: %d templates / time prior: %d templates", len(router.table), len(tp.prior))

    # ★★instseg 重畳（s1）。ロード失敗は ERROR に残して s0 相当で続行（提出は Failed にしない）
    n_ood = sum(1 for r in requests if is_ood(r))
    log.info("procedure gate: ID %d / OOD %d (two_pass=%s, conf_key=%s)",
             n - n_ood, n_ood, TWO_PASS, CONF_KEY)

    model, processor = load_model()
    warmup(model, processor)
    log.info("Model ready (setup %.1fs)", time.monotonic() - t_start)

    # ★qID 順は request.json のまま保つ。未処理分は空回答で埋まっているので、
    #   途中で SIGKILL されても「そこまでの回答 + 残りは空」の answer.json が残る。
    records: list[dict] = [{"qID": r.qID, "content": "", "latency": 0.0} for r in requests]
    n_failed = n_rushed = n_single = 0
    stat = {"agree": 0, "pick_first": 0, "pick_id": 0, "pick_ood": 0}
    for i, req in enumerate(requests, start=1):
        remaining = budget_total - (time.monotonic() - t_start)
        left = n - i + 1
        if remaining <= 0:
            log.warning("Budget exhausted at %d/%d — leaving the rest empty", i, n)
            break  # 残りは既に空回答で埋まっている
        per_q_left = remaining / max(left, 1)
        # ★退避は 2 段: まず 2 パスをやめて 2nd model 単独（1 パス）、それでも足りなければ rush 構成
        two_pass = TWO_PASS
        if per_q_left >= PER_QUESTION_BUDGET_S * SAFETY_MARGIN:
            n_frames, width = router.config_for(req.question)
            max_new = MAX_NEW_TOKENS
        elif per_q_left >= PER_QUESTION_BUDGET_S * SAFETY_MARGIN_SINGLE:
            n_frames, width = router.config_for(req.question)
            max_new = MAX_NEW_TOKENS
            two_pass = False; n_single += 1
        else:
            n_frames, width = RUSH_CONFIG
            max_new = MAX_NEW_TOKENS_RUSH
            two_pass = False; n_rushed += 1

        t0 = time.monotonic()
        try:
            content, n_got, info = answer_one(model, processor, req, n_frames, width,
                                              max_new, tp, two_pass=two_pass)
            if two_pass:
                stat["agree"] += int(bool(info.get("agree")))
                stat["pick_" + info["picked"]] += 1
        except Exception:
            n_failed += 1
            log.exception("qID=%s failed; emitting empty answer", req.qID)
            content, n_got, info = "", 0, {}
        latency = time.monotonic() - t0
        records[i - 1] = {"qID": req.qID, "content": content, "latency": latency}
        # ★1問ごとに保存する。OOM-kill / タイムアウトは例外として捕まえられないので、
        #   「最後にまとめて書く」設計だとそこまでの回答が全部消える。
        #   B=20 で数十KB の書き込みなので 15s/問の予算に対して無視できる。
        try:
            write_answers(records)
        except Exception:
            log.exception("incremental save failed at %d/%d (continuing)", i, n)
        if i <= 5 or i % 20 == 0:
            log.info("[%d/%d] %s %df@%d %.2fs %s/%s agree=%s -> %r", i, n, req.qID, n_got,
                     width, latency, info.get("second"), info.get("picked"),
                     info.get("agree"), content[:60])

    write_answers(records)
    total = time.monotonic() - t_start
    n_empty = sum(1 for r in records if not r["content"].strip())
    log.info("Wrote %d responses (empty=%d failed=%d single=%d rushed=%d unknown-template=%d) "
             "in %.1fs / budget %.0fs  select: agree=%d pick_first=%d pick_id=%d pick_ood=%d",
             len(records), n_empty, n_failed, n_single, n_rushed, router.n_unknown, total,
             budget_total, stat["agree"], stat["pick_first"], stat["pick_id"], stat["pick_ood"])
    # ★★重畳が本当に効いたかを**数字で**残す（v008 の「静かに重畳なし」事故の再発防止）。
    #   drawn=0 なら s0 相当で走っている＝CV 換算で −0.067 相当の取りこぼし。
    if total > budget_total:
        log.warning("OVER BUDGET by %.1fs (%.0f%%)", total - budget_total,
                    100 * (total / budget_total - 1))
    log.info("=== done ===")
    return 0


def main() -> int:
    """★どんな落ち方をしても answer.json を残して exit 0 で終える.

    非ゼロ終了・出力なしのどちらでも grand-challenge は
    "The algorithm failed on one or more cases." にする（2026-08-14 に実害）。
    空回答は 0 点だが**提出は成立する**ので、常にそちらへ倒す。
    """
    if _IMPORT_ERROR is not None:
        log.error("import failed — answering nothing", exc_info=_IMPORT_ERROR)
    else:
        try:
            run()
        except BaseException:  # noqa: BLE001 — KeyboardInterrupt/SystemExit も飲む
            log.exception("run() crashed — keeping the answers written so far")

    if not ANSWER_PATH.exists():
        # ここまで来て出力が無い ＝ bootstrap も失敗している。最後にもう一度だけ試す。
        try:
            write_answers([{"qID": q, "content": "", "latency": 0.0} for q in _QIDS])
        except Exception:
            log.exception("could not write %s — the job will fail", ANSWER_PATH)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
