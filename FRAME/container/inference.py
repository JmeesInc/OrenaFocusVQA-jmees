r"""ORena SAVE FOCUS — FRAME track container v008 (OOD hardening)
(Qwen3.5-9B QLoRA **3モデル多数決 / anytime** + **instseg 重畳ヒント** + **procedure_type**).

## v006 からの差分（3つ）

1. ★**v006 のバグ修正: user ターンの head が欠けていた**。
   学習/CV は `dataset_seg.build_messages` で
   `["Frame at [HH:MM:SS]:", <画像>, 質問]` を渡しているのに、v006 のコンテナは
   `[<画像>, 質問]` しか渡していなかった（SEGMENT の v005 は正しく再現している）。
   **v006 の CV 0.6548 は head ありで測った値**なので、コンテナは測った条件と別物だった。
2. **instseg 重畳（dual）**: Mask2Former(expF23) のマスクを青で重畳した画像を
   **2枚目として追加**する（原画像は残す）。leak-aware 2,886問で
   SCORE 0.6589 → **0.6670（+0.0081, McNemar p=0.17 n.s.）**。
   ⚠️**有意ではない**。ユーザ判断で採用。
3. **`procedure_type`** を user ターン先頭に付ける（公式が request.json でくれる情報）。

CV (fold v003 fold0 val, N=1962, judge 込み):
  member1 単独 0.6396 → **多数決3本 0.6548**（83勝53敗, McNemar **p=0.0126 \***）

## なぜ FRAME だけアンサンブルするのか
SCORE は**バケットの非加重平均**。**FRAME は全 20,000 問が OBJECT_RECOGNITION (11,647) と
AGGREGATION (8,353) の 2 group しか持たない**（構造的にそう）ので、平均改善がそのまま
SCORE に乗る。一方 SEGMENT/PROCEDURE は 5 group あり complex 53 / event 74 のような
小バケットが同じ重みを持つため、**多数決は大バケットで勝って小バケットで負け、差し引き悪化する**
（SEGMENT 実測 0.7197 → 0.7153）。SEGMENT は v005 の group 振り分けが正解。

## メンバー（すべて同じ base Qwen3.5-9B 上の LoRA。base は1回だけロードする）
| 順 | adapter | 推論解像度 | 単独 SCORE |
|---|---|---|---|
| 1 | expV05_frame_1024px | **1024px** | 0.6396 |
| 2 | expV06_frame_r32 | 768px | 0.6354 |
| 3 | expE10_frame_specialist_768_v003 | 768px | 0.6289 |

⚠️ **この3本は val 上の SCORE 順で選んでいる**（＝ val への選択バイアスがある）。
5本・7本が n.s. なので「多いほど良い」ではない点も含め、**別 fold での追試が必要**。

## 打ち切り（anytime）
`ensemble.py` の docstring を参照。**パス単位**で回すので pass 1 完了時点で提出物が完成し、
以降の pass はどこで止めても損失ゼロ。詳細:
  - pass 前: 実測スループットから残り時間で入るかを判断
  - pass 中: 1問ごとに時計を見て、入らなくなったらその pass を打ち切る（残りは前パスの回答）
  - 生成中: `MaxTimeCriteria` が安全弁。★発火時の**部分出力は捨てる**
    （`00:12:` や `Clip, Spec` はパースに失敗して不正解になるため）

Source:
  workspace/expE01_segproc_baseline/results/{expV05_frame_1024px,expV06_frame_r32,
                                             expE10_frame_specialist_768_v003}/fold0/adapter
"""
from __future__ import annotations

import json
import logging
import os
import re
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from answer_norm import normalize  # noqa: E402
from detector import Detector  # noqa: E402
from ensemble import Budget, vote_all  # noqa: E402
from prompts import build_system_prompt, detect_format, set_fo_definitions  # noqa: E402

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

INPUT_PATH = Path(os.environ.get("FOCUS_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("FOCUS_OUTPUT", "/output"))
RESOURCES = Path(__file__).parent / "resources"
BASE_MODEL_DIR = Path(os.environ.get("FOCUS_BASE_MODEL", RESOURCES / "base_model"))

# (adapter ディレクトリ名, 推論時の幅px, 重畳を使うか)。**単独 SCORE の高い順**に並べること
#
# ★視点(control/dual)の組み合わせは 8 通りすべてを leak-aware 2,886問で採点した
#   （`workspace/expG00_vlm_grounding/sweep_views.py`。投票結果は必ずどれかのメンバーの
#     回答と文字列一致するので judge を回さずに再構成できる）:
#
#     D/C/D 0.6728 | D/C/C 0.6727 | D/D/C 0.6726 | D/D/D 0.6726
#     C/D/C 0.6711 | C/D/D 0.6684 | C/C/D 0.6684 | C/C/C 0.6679
#
#   ★**効くのは member1 が dual かどうかだけ**。m2/m3 の視点は何を選んでも動かない
#   （D/D/D vs D/C/D は 21改善/20悪化 p=1.000）。control と dual の不一致は 0.117 しかなく、
#   食い違ったときの正解率もほぼ対称なので、視点を混ぜても独立な情報が増えない。
#   D/C/D を採るのは **pass2 の入力トークンが 2020→1375 に減って速い**ため（精度は同点）。
# ★★v009 の構成（2026-09-02）: **重畳を学習に入れたモデル 2 本 + control 1 本**。
#   タプルは (adapter 名, 推論幅, 重畳バリアント)。**バリアント名 "" が control**。
#
#   ⚠️ **各メンバーは「自分が学習したときの描画」で重畳を受け取る**こと。
#      A は r1（青単色 + 番号）、E は r4（クラス色 + 番号 + confidence）で学習しており、
#      入れ替えると入力形式が学習と別物になる。expM00-A は重畳を外すだけで
#      **−0.0415** 落ちた（形式変化に非常に敏感）ので、ここは厳密に合わせる。
#
#   実測（FRAME val 3,928 / judge 込み / matched）:
#     m1 = expM00A @1024 dual(r1)  **0.6581**  ← 現時点の FRAME ベスト
#          （@768 dual は 0.6488。**解像度を上げる分には壊れない**ことを実測）
#     m2 = expM00A @768  dual(r1)  **0.6488**（同一 adapter・別解像度）
#     m3 = expV06_frame_r32 control（重畳を学習していないので **control でのみ使う**）
#
# ⚠️ **暫定構成（2026-09-03）**: 本来は m2 に expM00-E（r4 描画で学習した別モデル）を
#    置く設計だが、E は学習中（9/4 未明完了）。E が揃うまでは **A を 2 解像度で使う**。
#    同一 adapter なので投票の多様性は限定的（control/dual の不一致は 0.117 実測）。
#    E 完成後に m2 を差し替える（`stage_resources.sh` の 1 行 + MEMBERS の 1 行）。
# ★★v011（expM03 Round 1）: adapter を expM03B（外部 Surgical VQA 混合学習）に差し替えた以外
#   v009 と完全同一（解像度・重畳 r1・検出器・conf 0.25・投票）。LB 差分 = 外部データの寄与。
# ★★★v017: **バケット別ルーティング + 経路ごとの3本投票**
#
# LB のバケット分解（Validation 2,000問, N: obj_id 747 / agg_id 553 / obj_ood 512 / agg_ood 188）:
#   v011  obj_id 538 / agg_id 324 / obj_ood 262 / agg_ood 129  → SCORE 0.6260（4位）
#   v015  obj_id 515 / agg_id 338 / obj_ood 247 / agg_ood 118  → SCORE 0.6027
#   → **v015 が勝つのは aggregation_id ただ1つ**（+14問）。他3バケットは v011 が勝つ。
#   → 「ID × aggregation」だけ別モデルに振れば、実測バケットの組み替えで 0.6323 相当。
#
# 判定は2軸とも**質問文とリクエスト属性だけ**で決まる（モデルを走らせる前に確定する）:
#   ID/OOD    : procedure_type が学習4術式に含まれるか（[[2026-08-19 に確認]]）
#   capability: detect_format → number/binary = AGGREGATION / 他 = OBJECT_RECOGNITION
#               （FRAME val 実測で純度 1.000 / 1.000 / 0.980）
#
# ★経路Aは **v011 の3本をそのまま**（LB 0.6260 を出した実測構成に手を入れない）。
# ★経路Bは all-data 3本。**別々に学習した重み**であること（同一 adapter 2本の投票は負ける実測あり）。
ROUTES: dict[str, list[tuple[str, int, str]]] = {
    # OOD 全部 + ID×object
    "A": [
        ("m1_m03b_overlay", 1024, "r1"),
        ("m2_m03b_768", 768, "r1"),
        ("m3_v06_r32", 768, ""),
    ],
    # ID×aggregation（number / binary）
    "B": [
        ("b1_q02b_r1", 768, "r1"),    # v011 レシピ + all-data
        ("b2_q00b_r4", 768, "r4"),    # r4 + SurgAtlas + all-data（= v015 の中身）
        ("b3_q03b_r1sa", 768, "r1"),  # r1 + SurgAtlas + all-data
    ],
}
# 学習した4術式。これ以外は OOD（`procedure_type` は request.json に入っている）
KNOWN_PROCEDURES = {
    "proctocolectomy", "rectal resection", "sigmoid resection",
    "laparoscopic cholecystectomy",
}
# 経路Bへ送る回答形式（= AGGREGATION バケット）
AGG_FORMATS = {"number", "binary"}

# 全メンバー（load_model がここを見て adapter を全部ぶら下げる）
MEMBERS: list[tuple[str, int, str]] = [m for ms in ROUTES.values() for m in ms]
MAX_NEW_TOKENS = 64
# ★時間切れで投票から外した部分出力の退避先（どのメンバーも答えられなかった問の最後の砦）
SALVAGE: dict[str, str] = {}
DETECTOR_DIR = RESOURCES / "detector"
# 環境変数で個別に切れるようにする（回帰テストと、万一の切り戻し用）
USE_OVERLAY = os.environ.get("FOCUS_OVERLAY", "1") == "1"
USE_PROCEDURE = os.environ.get("FOCUS_PROCEDURE", "1") == "1"
# ★expM00-A は **conf 0.25** で学習した。学習と推論を必ず揃える。
DETECTOR_CONF = float(os.environ.get("FOCUS_DET_CONF", "0.25"))
# ★clip は specialist(expF39, 768x1344) が全クラス検出器の clip を置き換える
DETECTOR_CLIP_DIR = RESOURCES / "detector_clip"
OVERLAY_VARIANT = os.environ.get("FOCUS_OVERLAY_VARIANT", "r1")

# ── ヘルニア時の sponge→mesh 置換（★仮説であり検証不能。既定 OFF）──────────
# 公式 FO 定義は Mesh の説明に **わざわざ sponge との区別**を書いている:
#   "This differs from a sponge which appears more like a tightly woven cloth."
# しかし **Mesh は学習データ 50,000問に1件も無い**（実測）。学習済み9クラスのうち
# 見た目が最も近いのは sponge なので、ヘルニア動画の mesh は sponge と答えるはず。
# ⚠️ ヘルニア修復でもガーゼ/スポンジは実際に使われるので、全置換は真の sponge を落とす。
#   手元に mesh もヘルニア動画も1件も無く、**ローカルでは正誤の判定すらできない**。
#   → LB を実験装置として使う前提で、**既定は OFF**。
HERNIA_MESH = os.environ.get("FOCUS_HERNIA_MESH", "0") == "1"
HERNIA_RE = re.compile(r"hernia|\bTAPP\b|\bTEP\b|\bIPOM\b|inguinal|ventral|umbilical",
                       re.I)


def is_hernia(req: Request) -> bool:
    return bool(HERNIA_RE.search(getattr(req, "procedure_type", "") or ""))


def sponge_to_mesh(text: str) -> str:
    """`Sponge` を `Mesh` に置換する（語単位・大小文字を保つ）。重複は畳む。"""
    # ★`re.I` を忘れると `Sponge`（大文字始まり）に一致しない。実際に回帰テストで
    #   「対象 20問 / 置換 0問」を踏んだ（2026-08-24）。ラムダ側で大小を復元している以上、
    #   マッチは必ず case-insensitive にすること。
    out = re.sub(r"\bsponges?\b", lambda m: "Mesh" if m.group(0)[:1].isupper() else "mesh",
                 text, flags=re.I)
    parts = [p.strip() for p in out.split(",") if p.strip()]
    seen, uniq = set(), []
    for p in parts:                      # `Mesh, Mesh` → `Mesh`
        if p.lower() not in seen:
            seen.add(p.lower())
            uniq.append(p)
    return ", ".join(uniq) if len(parts) > 1 else out
# ★オフライン検証（m2f_dual）と**同一文言**。変えると検証した条件と別物になる。
DUAL_NOTE = ("The second image is the same frame with the object detector's foreign-object "
             "masks drawn in blue (green labels). Use the **first (clean) image** to read "
             "the scene; use the second only as a location hint. The detector is imperfect "
             "and often misses instances — do not simply count the blue overlays.")
# ★これは**プラットフォームが許す**セットアップ時間であって、こちらの都合で増やせない
#   （allowed = 120s + B×5s）。detector のロード時間は**この枠を消費する**側なので、
#   150 に増やすと「まだ余裕がある」と誤認して超過する。v006 と同じ 120 に戻す。
SETUP_BUDGET_S = 120.0
PER_QUESTION_BUDGET_S = 5.0        # FRAME
SAFETY = 0.85                      # 20%超過ラインの半分以下に自前の締切を置く
# 打ち切りテスト用: 予算を人工的に絞る（回帰テストで打ち切り経路を実際に踏むため）
BUDGET_SCALE = float(os.environ.get("FOCUS_BUDGET_SCALE", "1.0"))
MAX_MEMBERS = int(os.environ.get("FOCUS_MAX_MEMBERS", str(len(MEMBERS))))
# pass 全体が入らなくても、この割合の問が入るなら pass を走らせる（途中打ち切り前提）
MIN_PASS_FRACTION = 0.2


def route_of(req) -> str:
    """この問をどの経路に送るか。**モデルを走らせる前に**質問文と属性だけで決まる。

    ★誤判定の向きは非対称にしてある: 迷ったら経路A（= LB 0.6260 の実測構成）へ倒す。
      OOD を ID と誤れば agg_ood を v015 系で答えてしまい損（実測 −11問相当）なので、
      **procedure_type が読めない/未知のときは OOD 扱い＝経路A**にする。
    """
    if "B" not in ROUTES or not ROUTES["B"]:
        return "A"
    proc = (getattr(req, "procedure_type", "") or "").strip().lower()
    if proc not in KNOWN_PROCEDURES:
        return "A"                      # OOD（または不明）→ 実測構成
    fmt = detect_format(req.question)
    return "B" if fmt in AGG_FORMATS else "A"


def load_model():
    """base(4bit NF4) を1回ロードし、LoRA adapter を全メンバー分ぶら下げる。"""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(str(BASE_MODEL_DIR))
    model = AutoModelForImageTextToText.from_pretrained(
        str(BASE_MODEL_DIR), quantization_config=bnb, device_map="cuda",
        dtype=torch.bfloat16).eval()
    names = []
    for i, (name, _, _) in enumerate(MEMBERS[:MAX_MEMBERS]):
        d = RESOURCES / name
        if not d.exists():
            log.warning("adapter %s が無い — このメンバーは飛ばす", d)
            continue
        if i == 0 or not names:
            # ★4bit ベースに載せた LoRA を merge_and_unload してはいけない（黙って捨てられる）
            model = PeftModel.from_pretrained(model, str(d), adapter_name=name).eval()
        else:
            model.load_adapter(str(d), adapter_name=name)
        names.append(name)
    if not names:
        raise RuntimeError(f"resources/ に adapter が1つも無い: {RESOURCES}")
    log.info("adapters loaded: %s", names)
    return model, processor, names


def resize_to(img: Image.Image, width: int) -> Image.Image:
    """学習時と同じ前処理: 幅 width にアスペクト比維持でリサイズ。"""
    if img.width != width:
        h = max(2, round(img.height * width / img.width))
        h -= h % 2                       # ffmpeg の scale=W:-2 と同じく偶数に揃える
        img = img.resize((width, h), Image.BILINEAR)
    return img


def hhmmss(t: float) -> str:
    t = int(round(t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def build_user_content(req: Request, image: Image.Image,
                       overlay: Image.Image | None,
                       overlay_note: str = "") -> list[dict]:
    """★`dataset_seg.build_messages` と**同一レイアウト**にする（v006 はここが欠けていた）。

        [ "Procedure: X\nFrame at [HH:MM:SS]:", <原画像>, (<重畳画像>, 説明文), 質問 ]

    `procedure_type` を **system でなく user ターン先頭**に置くのは、LoRA が system の
    指示をほぼ無視するため（2026-08-14 expG00 で実測）。
    """
    head = f"Frame at [{hhmmss(req.start_time)}]:"
    proc = (getattr(req, "procedure_type", "") or "") if USE_PROCEDURE else ""
    if proc:
        head = f"Procedure: {proc}\n" + head
    content: list[dict] = [{"type": "text", "text": head},
                           {"type": "image", "image": image}]
    if overlay is not None:
        # ★検出ゼロのときは呼び出し側が overlay=None を渡す（= 2枚目を付けない）。
        content.append({"type": "image", "image": overlay})
        # ★説明文は **描画バリアントに紐づく**（r1 は番号付けの説明が入る）。
        #   overlay_render.note_for() が返した文字列をそのまま使う（学習と同一）。
        content.append({"type": "text", "text": overlay_note or DUAL_NOTE})
    content.append({"type": "text", "text": req.question})
    return content


def generate(model, processor, req: Request, image: Image.Image,
             overlay: Image.Image | None, deadline_s: float | None,
             overlay_note: str = "") -> str | None:
    """1問1メンバー分の生成。時間切れで打ち切られたら **None** を返す（部分出力は使わない）。"""
    from transformers import MaxTimeCriteria, StoppingCriteriaList
    system_prompt, _fmt = build_system_prompt(req.question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_content(req, image, overlay, overlay_note)},
    ]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    imgs = [image] if overlay is None else [image, overlay]
    inputs = processor(text=[text], images=imgs, return_tensors="pt").to(model.device)
    kw = {}
    if deadline_s is not None and deadline_s > 0:
        kw["stopping_criteria"] = StoppingCriteriaList([MaxTimeCriteria(max_time=deadline_s)])
    # ★v011: expM03B は EOS を打たずに偽ターンを続けることがある（46%実測、latency 1.8→6.8s）。
    #   FOCUS の答えは全形式1行なので**改行で生成を止める**。クリーンな答えには無影響で、
    #   暴走時のみ latency と末尾ゴミを断つ（answer_norm 側の1行目切断と二重の防御）。
    t0 = time.monotonic()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             stop_strings=["\n"], tokenizer=processor.tokenizer, **kw)
    took = time.monotonic() - t0
    n_new = gen.shape[1] - inputs.input_ids.shape[1]
    out = processor.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    # ★「時間切れで途中打ち切りされたか」は**経過時間で判定する**。
    #   最終トークンが EOS か否かで判定してはいけない: FRAME の回答は 4〜5 トークンと短く、
    #   正常終了でも `eos_token_id` と一致しない停止トークン（`<|im_end|>` 等）で終わるため、
    #   **正常な回答を全部「打ち切り」と誤判定して捨てた**（2026-08-14 に実害、20/20 が空回答）。
    #   `max_new_tokens` に達した場合は打ち切りではない（従来どおり採用する）。
    if kw and n_new < MAX_NEW_TOKENS and took >= deadline_s * 0.98:
        # ★打ち切られた出力は**投票には使わない**（不完全な文字列が多数派を汚す）。
        #   ただし **捨て切らずに SALVAGE へ退避**する。
        #   [[answer-beats-abstain-when-zero]]: 無回答は必ず 0 点なので、
        #   どのメンバーも答えられなかった問に限り、最後にこれを使う方が期待値が高い。
        #   （v008 は p1only で 0 件だったが、v009 は member1 が画像2枚で遅く、
        #     予算 16% の極端な絞り込みで 1 問取りこぼした）
        if out:
            SALVAGE[req.qID] = out
        log.warning("qID=%s: 生成が時間切れ（%.2fs / 上限 %.2fs, %d tok）— 投票からは除外"
                    "（%s）", req.qID, took, deadline_s, n_new,
                    "SALVAGE に退避" if out else "出力も空")
        return None
    return out


def run_pass(model, processor, detector, name: str, width: int, ov_variant: str,
             requests: list[Request], budget: Budget, per_q_est: float | None,
             is_first: bool) -> tuple[dict[str, str], float]:
    """1メンバーで全問を回す。

    ★**pass 1 と pass 2以降で方針が違う**:
      - `is_first=True`（pass 1）: **全問に必ず挑む**。ここで問を飛ばすと
        その問は**どのメンバーの回答も持たない＝空回答**になり、確実に不正解になる。
        1問ごとの上限は `MaxTimeCriteria` が押さえるので、時間超過は生成単位で頭打ちになる
      - `is_first=False`（pass 2以降）: 純粋な上積みなので、**残り時間で1問ぶん賄えなくなったら
        そこで打ち切る**。未到達の問は前パスの回答が残り、投票メンバーが減るだけ
        （同数は member1 先勝ちなので**単独より悪くならない**）
    """
    model.set_adapter(name)
    out: dict[str, str] = {}
    n_tried = 0
    t_pass = time.monotonic()
    for i, req in enumerate(requests, 1):
        if not is_first:
            # 「もう1問ぶん（+5割の余裕）入るか」で判断する。
            # ★「残り全問ぶん入るか」で判断してはいけない: 一部しか入らない pass が
            #   1問目で即 break して、走らせた意味が無くなる
            if per_q_est is not None and not budget.can_afford(per_q_est * 1.5):
                log.warning("[%s] 予算のため pass を %d/%d 問で打ち切り（残 %.1fs）",
                            name, i - 1, len(requests), budget.left)
                break
            if budget.left <= 0:
                log.warning("[%s] 予算切れ — pass 中断", name)
                break
        t0 = time.monotonic()
        try:
            # ★検出は**原寸フレーム**に対して行い、重畳してから member の幅に縮める。
            #   検出結果を pass 間でキャッシュしない（バッチが大きいとホスト RAM を食う。
            #   再計算は RTX4090 で 0.08s/問）。
            orig = Image.open(INPUT_PATH / "frames" / f"{req.qID}.png").convert("RGB")
            img = resize_to(orig, width)
            ov, ov_note = None, ""
            if ov_variant and detector is not None and detector.ok:
                # ★★expM00 と**同一手順**にする（v008 からの変更点）:
                #   1. **先に member の幅へ縮めてから描く**。v008 は原寸(960x540/1280x720)に
                #      描いてから縮めていたので、表示上の文字サイズが CV と違い、
                #      heico(0.8倍) と lapchole(0.6倍) でも違っていた
                #   2. クラス名は**公式表記**（`Silicon_Loop` ではなく `Silicone Loop`）
                #   3. **検出0件なら重畳を作らない**（None のまま = 単画像）。
                #      v008 は原画像と同一の2枚目を付けており expG00 §39 で −0.0429
                # ★**メンバーごとのバリアント**で描く。検出自体は (qID, 幅) で
                #   キャッシュされるので、2メンバーが同じ幅なら検出は1回で済む。
                ov, ov_note = detector.overlay(img, ov_variant, key=(req.qID, width))
            # 生成単体の上限は「1問あたり予算」か「残り時間」の小さい方
            # 生成単体の上限。★下限 2s を置く: 残り時間が細ったときに 1s まで縮むと
            #   正常な回答まで「時間切れ」で捨ててしまう
            hard = max(2.0, min(PER_QUESTION_BUDGET_S * 2.0, budget.left))
            ans = generate(model, processor, req, img, ov, hard, ov_note)
            if ans is not None:
                out[req.qID] = normalize(build_system_prompt(req.question)[1], ans)
        except Exception:
            log.exception("[%s] qID=%s 失敗 — このメンバーはこの問を棄権", name, req.qID)
        n_tried += 1
        if per_q_est is None:
            per_q_est = time.monotonic() - t0
    dt = time.monotonic() - t_pass
    # ★1問あたりコストは「試した問数」で割る（成功数で割ると、失敗が多いときに
    #   コストを過大評価して以降の pass を不当に諦める）
    cost = dt / max(n_tried, 1)
    log.info("[%s] pass 完了 回答 %d / 試行 %d / 全 %d 問 / %.1fs (%.2fs/問)",
             name, len(out), n_tried, len(requests), dt, cost)
    if n_tried and not out:
        log.error("★[%s] %d問すべてで回答が得られなかった。予算ではなく**バグ**の可能性が高い",
                  name, n_tried)
    return out, cost


def run() -> int:
    log.info("=== ORena FOCUS FRAME v007 — 3モデル多数決 + instseg 重畳 (anytime) start ===")
    log.info("CUDA available: %s (%s)", torch.cuda.is_available(),
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    requests = load_requests(INPUT_PATH / "request.json")
    if not requests:
        log.error("request.json contains no requests")
        return 1
    n = len(requests)
    # ★BUDGET_SCALE は**予算全体**（setup 項込み）を縮める。per-question 項だけを縮めると
    #   120s の setup 項が支配してしまい、いくら絞っても打ち切り経路を踏めない
    #   （2026-08-14: scale 0.12 でも 3パス完走してしまい、テストとして無意味だった）。
    budget = Budget(n, SETUP_BUDGET_S * BUDGET_SCALE,
                    PER_QUESTION_BUDGET_S * BUDGET_SCALE, SAFETY)
    log.info("Batch of %d question(s); hard %.0fs / soft %.0fs%s",
             n, budget.allowed, budget.deadline,
             f"  (BUDGET_SCALE={BUDGET_SCALE})" if BUDGET_SCALE != 1.0 else "")

    fo_text = json.loads((INPUT_PATH / "FO_definitions.json").read_text())
    set_fo_definitions(fo_text)

    model, processor, names = load_model()
    budget.log_state("model loaded")
    detector = None
    if USE_OVERLAY:
        detector = Detector(DETECTOR_DIR, conf=DETECTOR_CONF,
                            clip_ckpt=DETECTOR_CLIP_DIR, variant=OVERLAY_VARIANT)
        budget.log_state("detector loaded")
    log.info("overlay=%s procedure=%s hernia_mesh=%s views=%s", bool(detector and detector.ok), USE_PROCEDURE, HERNIA_MESH,
             " ".join(f"{k}:" + "/".join(o if o else "C" for _, _, o in ms)
                      for k, ms in ROUTES.items()))

    # ── ★経路ごとに問を振り分ける（モデルを走らせる前に確定する）──────────
    by_route: dict[str, list] = {k: [] for k in ROUTES}
    for req in requests:
        by_route[route_of(req)].append(req)
    log.info("★ルーティング: " + " / ".join(
        f"{k} {len(v)}問 [{','.join(n for n, _, _ in ROUTES[k])}]" for k, v in by_route.items()))

    # ── ★ラウンド単位で回す（経路をまたいで pass1 を先に全部終わらせる）────
    #   こうすると **ラウンド1 完了時点で全問に回答がある**（v011 の anytime 性質を維持）。
    #   ラウンド2/3 は純粋な上積みで、途中で止めても投票メンバーが減るだけ。
    per_member_by_route: dict[str, list[dict[str, str]]] = {k: [] for k in ROUTES}
    est = None
    n_round = max(len(v) for v in ROUTES.values())
    stop = False
    for k in range(n_round):
        for rk, reqs_r in by_route.items():
            if stop or not reqs_r or k >= len(ROUTES[rk]):
                continue
            name, width, ov_variant = ROUTES[rk][k]
            if name not in names:
                log.warning("★[%s] adapter %s が無いので飛ばす", rk, name)
                continue
            is_first = (k == 0)
            if not is_first:
                need = est * len(reqs_r) if est else 0.0
                if not budget.can_afford(need * MIN_PASS_FRACTION):
                    log.warning("★round %d [%s] (%s) は一部すら入らないので実行しない"
                                "（全体 %.0fs / 残 %.0fs）", k + 1, rk, name, need, budget.left)
                    stop = True
                    continue
                log.info("★round %d [%s] (%s) を実行（%d問 / 全体 %.0fs / 残 %.0fs%s）",
                         k + 1, rk, name, len(reqs_r), need, budget.left,
                         "" if budget.can_afford(need) else " — **途中打ち切り見込み**")
            got, est_r = run_pass(model, processor, detector, name, width, ov_variant,
                                  reqs_r, budget, est, is_first=is_first)
            if est_r:
                est = est_r
            per_member_by_route[rk].append(got)
            budget.log_state(f"after round {k+1} [{rk}] ({name})")

    qids = [r.qID for r in requests]
    # ★投票は**経路の中だけ**で行う（別経路のメンバーは担当外の問を持っていない）
    final: dict[str, str] = {}
    for rk, mems in per_member_by_route.items():
        if not mems:
            continue
        final.update(vote_all(mems, [r.qID for r in by_route[rk]]))
    if HERNIA_MESH:
        by_id = {r.qID: r for r in requests}
        n_h = n_sub = 0
        for q in qids:
            if not is_hernia(by_id[q]):
                continue
            n_h += 1
            new = sponge_to_mesh(final.get(q, ""))
            if new != final.get(q, ""):
                n_sub += 1
                final[q] = new
        log.info("★hernia sponge→mesh: 対象 %d問 / 置換 %d問（procedure_type で判定）",
                 n_h, n_sub)
    # ★★最後の砦: どのメンバーも答えられなかった問に、打ち切り出力を充てる。
    #   無回答は確実に 0 点、部分出力は当たる可能性がある（期待値で必ず勝つ）。
    n_salv = 0
    for q in qids:
        if not final.get(q) and SALVAGE.get(q):
            by_q = {r.qID: r for r in requests}
            final[q] = normalize(build_system_prompt(by_q[q].question)[1], SALVAGE[q])
            n_salv += 1
    if n_salv:
        log.info("★SALVAGE 適用: %d 問（無回答→打ち切り出力）", n_salv)
    n_empty = sum(1 for q in qids if not final.get(q))
    n_voted = sum(1 for rk, mems in per_member_by_route.items()
                  for q in (r.qID for r in by_route[rk])
                  if sum(q in m for m in mems) >= 3)
    n_pass = sum(len(m) for m in per_member_by_route.values())
    responses = [Response(qID=q, content=final.get(q, ""), latency=0.0) for q in qids]

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_items(responses, OUTPUT_PATH / "answer.json")
    log.info("Wrote %d responses / passes=%d / 3本投票できた問 %d/%d / 空 %d / "
             "%.1fs of hard %.0fs (%.0f%%)",
             len(responses), n_pass, n_voted, n, n_empty,
             budget.elapsed, budget.allowed, 100 * budget.elapsed / budget.allowed)
    if budget.elapsed > budget.allowed:
        log.error("OVER HARD BUDGET by %.1fs (%.0f%%) — forfeit の危険",
                  budget.elapsed - budget.allowed, 100 * (budget.elapsed / budget.allowed - 1))
    log.info("=== done ===")
    return 0


def main() -> int:
    """★例外で**非ゼロ終了すると提出ごと Failed** になる。必ず 0 で終える。
    answer.json は `_prewrite_empty_answers()` が import 前に置いてあるので、
    ここで落ちても「全問空回答（=不正解）」で採点される。Failed よりは遥かに良い。
    """
    try:
        return run()
    except Exception:
        log.exception("致命的エラー — 事前に書いた空回答のまま 0 で終了する")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
