"""expE01 — SEGMENT / PROCEDURE の QLoRA fine-tuning（複数フレーム入力, vision+language）.

expD00（FRAME）との差分は **入力が1枚の画像 → 時刻ラベル付きの複数フレーム** だけ。
それ以外（4bit NF4 / vision+language 両方に LoRA / answer トークンのみ loss /
best checkpoint / resume / VRAM 先取り確保）は expD00 の構成をそのまま踏襲する。

**補助損失（計数の距離損失・fo_class 集合損失）は載せていない。**
expD22 / expD24 で「損失関数としては正しく動くのに、最適化したい対象が1問も動かない」ことが
確定しており（|E[x]-gt| 0.755→0.750、多クラス fo_class 0.176→0.176）、
持ち込むと config が複雑になるだけで得るものが無い。

実行: bash workspace/expE01_segproc_baseline/run.sh <config.yaml> [--train-limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# ★`ROOT` は **dataset_seg のものを唯一の定義として使う**（= リポジトリルート .../Orena）。
#   ここには以前 `ROOT = HERE.resolve().parents[2]` があったが、`HERE` は**ディレクトリ**
#   なので `dataset_seg.ROOT`（`Path(__file__).resolve().parents[2]` = ファイル基準）より
#   1 階層上を指していた（dl1 では .../MICCAI2026、貸しGPU では /workspace）。
#   2026-09-01 まで誰も参照していなかったので露見せず、expM00 の重畳注入で初めて
#   `ModuleNotFoundError: overlay_attach` として出た。定義を 2 つ持たない形にする。
from dataset_seg import (ROOT, build_messages, build_multitrack_samples,  # noqa: E402
                         build_samples)

# ★学習側も `procedure_type` の有無を切り替えられるようにする（config: data.with_procedure）。
#   ⚠️**学習と推論で必ず揃える**こと。片方だけ入れると入力分布がズレる。
WITH_PROCEDURE = False  # noqa: E402

log = logging.getLogger("expE01.train")


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    root.addHandler(ch); root.addHandler(fh)
    for noisy in ("httpcore", "httpx", "filelock", "fsspec", "urllib3", "PIL",
                  "datasets", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class SeqDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class VideoCollator:
    """`VideoVQASample` → model 入力 + labels（prompt 部分は -100）。

    ★複数フレームでも構造は FRAME と同じ。違いは `build_messages` が
      「[hh:mm:ss] テキスト → 画像」を n_frames 回繰り返す点だけ。

    ★`per_device_batch_size` は 1 前提で書く。16フレーム/サンプルだと `pixel_values` が
      1サンプルで既に数千行あり、bs>1 は 24GB では載らない。
      有効バッチは `grad_accum` で作る（expD00 と同じ bs1×accum16）。
    """

    def __init__(self, processor, max_seq_len: int):
        from qwen_vl_utils import process_vision_info
        self.processor = processor
        self.max_seq_len = max_seq_len
        self._pvi = process_vision_info
        # ★Qwen 系以外（LFM2-VL 等）は `videos=` を受け付けず、`apply_chat_template` に
        #   `images=` を渡す経路も無い（transformers 5.x で TypeError）。
        #   **text を先に作って `processor(text=..., images=...)`** が唯一通る形。
        #   2026-08-17 に LFM2.5-VL-3B で5通り試して確定（probe_lfm_processor.py）。
        #   ⚠️Qwen 側の呼び出しは**一切変えない**（既存の全実験と bit 一致を保つため）。
        self._is_qwen = "Qwen" in type(processor).__name__

    def __call__(self, batch):
        if len(batch) == 1:
            return self._one(batch[0])
        return self._merge([self._one(s) for s in batch])

    # ── bs>1 のためのパディング結合 ──────────────────────────────
    # ★動機（2026-08-13 実測）: **この学習は重みのメモリ帯域に律速されている**。
    #   同じ AUG で bf16(重み18.5GB) は 150 s/it、4bit(4.7GB) は 133 s/it と、
    #   **読む量が少ない方が速い**＝帯域律速。帯域律速なら **バッチを増やすと
    #   重み読み込みが償却されてほぼ線形に速くなる**（計算律速になるまで）。
    #
    # ★パディングは **左詰め**にする。生成時の慣習に合わせるためではなく、
    #   `labels` を右端に揃えたいから: 右パディングだと答えの直後にパディングが入り、
    #   位置エンコーディングと `attention_mask` の整合を自前で取る必要が出る。
    #   左パディングなら「実データは常に右端」で揃うので破綻しにくい。
    #
    # ⚠️**`pixel_values` は連結、`image_grid_thw` も連結**する（サンプルごとに
    #   枚数が違ってよい設計。Qwen VL は grid_thw で画像境界を復元する）。
    #   ここを取り違えると**画像とテキストの対応が黙ってずれる**。
    SEQ_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids", "labels")

    def _merge(self, items: list[dict]) -> dict:
        import torch as _t
        maxlen = max(int(x["input_ids"].shape[1]) for x in items)
        out: dict = {}
        for k in self.SEQ_KEYS:
            if k not in items[0]:
                continue
            pad = -100 if k == "labels" else 0
            rows = []
            for x in items:
                v = x[k]
                n = maxlen - int(v.shape[1])
                if n > 0:                      # ★左パディング
                    v = _t.nn.functional.pad(v, (n, 0), value=pad)
                rows.append(v)
            out[k] = _t.cat(rows, dim=0)
        # 画像系はサンプル方向に連結（枚数が違ってよい）
        for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if k in items[0]:
                out[k] = _t.cat([x[k] for x in items], dim=0)
        # 取りこぼしがあれば気づけるようにする（黙って落とさない）
        known = set(self.SEQ_KEYS) | {"pixel_values", "image_grid_thw",
                                      "pixel_values_videos", "video_grid_thw"}
        missing = [k for k in items[0] if k not in known]
        if missing:
            raise RuntimeError(
                f"VideoCollator._merge が扱えないキー {missing}。"
                "bs>1 で黙って落とすと学習が壊れるので明示的に落とす。")
        return out

    def _one(self, s):
        msg_prompt = build_messages(s, with_procedure=WITH_PROCEDURE)
        msg_full = msg_prompt + [{"role": "assistant", "content": s.answer}]

        prompt_text = self.processor.apply_chat_template(
            msg_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full_text = self.processor.apply_chat_template(
            msg_full, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        image_inputs, video_inputs = self._pvi(msg_full)

        if self._is_qwen:
            full = self.processor(text=[full_text], images=image_inputs, videos=video_inputs,
                                  return_tensors="pt")
            prompt = self.processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                                    return_tensors="pt")
        else:
            # ★`videos=` を渡さない。LFM は画像列として受ける（1問の全フレームが images）。
            full = self.processor(text=[full_text], images=image_inputs,
                                  return_tensors="pt")
            prompt = self.processor(text=[prompt_text], images=image_inputs,
                                    return_tensors="pt")
        labels = full.input_ids.clone()
        plen = prompt.input_ids.shape[1]
        # ★prompt_text が full_text の接頭辞である保証はモデル依存（Gemma は崩れる）。
        #   崩れたまま labels[:, :plen] を潰すと**学習対象トークンが0個 = loss 不定**になる。
        f_ids, p_ids = full.input_ids[0], prompt.input_ids[0]
        n = min(len(f_ids), len(p_ids))
        if plen <= len(f_ids) and bool(torch.equal(f_ids[:plen], p_ids[:plen])):
            mask_len = plen
        else:
            nz = (~(f_ids[:n] == p_ids[:n])).nonzero()
            mask_len = int(nz[0]) if len(nz) else n
        labels[:, :mask_len] = -100
        if int((labels != -100).sum()) == 0:
            raise RuntimeError(
                f"学習対象トークンが 0 個（prompt_len={plen} full_len={len(f_ids)} "
                f"mask_len={mask_len}）。チャットテンプレートの接頭辞性が崩れている。"
                f"answer={s.answer!r}")

        out = {k: full[k] for k in full}
        out["labels"] = labels
        # ★seq 系だけ切り詰める。pixel_values / image_grid_thw は画像特徴なので切らない。
        #   ⚠️切り詰めが answer に届くと教師が消えるので、超過は警告して数える（下の統計）。
        seq_keys = ("input_ids", "attention_mask", "mm_token_type_ids", "labels")
        seq_len = full.input_ids.shape[1]
        if seq_len > self.max_seq_len:
            for k in seq_keys:
                if k in out and hasattr(out[k], "shape") and out[k].shape[-1] == seq_len:
                    out[k] = out[k][:, : self.max_seq_len]
            if int((out["labels"] != -100).sum()) == 0:
                raise RuntimeError(
                    f"max_seq_len={self.max_seq_len} の切り詰めで answer が全部落ちた "
                    f"(seq_len={seq_len})。**max_seq_len を上げること**")
        return out


def check_seq_lengths(samples, collator, n: int = 32) -> None:
    """先頭 n 件の系列長を実測して max_seq_len の妥当性を確認する.

    ★複数フレームでは系列長がフレーム数×解像度で大きく変わる。max_seq_len が足りないと
      **answer が切り落とされて静かに学習が壊れる**ので、学習開始前に必ず測る。
    """
    lens = []
    for s in samples[:n]:
        enc = collator([s])
        lens.append(int(enc["input_ids"].shape[1]))
    lens.sort()
    log.info(f"系列長 実測 (n={len(lens)}): min={lens[0]} p50={lens[len(lens)//2]} "
             f"max={lens[-1]}  / max_seq_len={collator.max_seq_len}")
    if lens[-1] > collator.max_seq_len:
        log.warning(f"★max_seq_len({collator.max_seq_len}) を超えるサンプルがある "
                    f"(最大 {lens[-1]})。切り詰めが起きる")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config_seg_16f.yaml"))
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--track", default=None)
    ap.add_argument("--uid-list", default=None,
                    help="expE23 量揃え対照: この JSON（uid のリスト）に train サンプルを絞る。"
                         "answer はそのまま（差し替えない）。CoT 学習と**同じ uid・答えのみ**の"
                         "対照を作り、ΔSCORE から「データ量」の交絡を分離するために使う")
    ap.add_argument("--cot-targets", default=None,
                    help="expE23: STaR で作った CoT 教師（uid -> {target,...}）の JSON。"
                         "渡すと train サンプルを **その uid に絞り**、system から OUTPUT RULE を"
                         "落とし、質問の末尾に自由記述の指示を足し、answer を "
                         "`Observations:...\\n\\nANSWER: <GT>` に差し替える。"
                         "★val サンプルにも同じ入力側の改変を当てる（学習中の eval loss を"
                         "同じ分布で測るため）。教師のかたち以外は対照 run と完全に同一")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.train_limit is not None:
        cfg["train"]["train_limit"] = args.train_limit
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.track is not None:
        cfg["data"]["track"] = args.track

    exp, dat, mc, tr = cfg["experiment"], cfg["data"], cfg["model"], cfg["train"]
    tag = f"fold{dat['fold']}"
    out_dir = HERE / "results" / exp["name"] / tag
    # 同名は _001.. で採番（上書き禁止）。**adapter の有無だけで完了判定する**
    # （checkpoint の有無で判定すると、完走した smoke の checkpoint から resume してしまう）
    if out_dir.exists() and (out_dir / "adapter").exists():
        i = 1
        while (p := out_dir.parent / f"{tag}_{i:03d}").exists():
            i += 1
        out_dir = p
    setup_logging(out_dir)
    shutil.copy(args.config, out_dir / "config.yaml")
    what = ("+".join(f"{s['track']}x{s['limit']}" for s in dat["tracks"])
            if dat.get("tracks") else dat.get("track", "?"))
    log.info(f"=== expE01 QLoRA  {what}  {tag}  out={out_dir} ===")

    import transformers
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    transformers.set_seed(exp["seed"])

    # ── VRAM 先取り確保（共有GPU対策）──
    # 重みロードの数分の間に他ジョブが同じ GPU へ入って OOM する事故が頻発した。
    # caching allocator は del してもドライバへ返さないので、起動直後に確保→即 del すると
    # 「他プロセスからは使用中、自分は後でそのプールを使える」状態が作れる。
    # ※ empty_cache() を呼ぶと返してしまうので絶対に呼ばない。
    reserve_mb = int(tr.get("reserve_mb", 0))
    min_reserve_mb = int(tr.get("min_reserve_mb", reserve_mb))
    if reserve_mb and torch.cuda.is_available() and int(os.environ.get("RANK", 0)) == 0:
        got = 0
        for mb in range(reserve_mb, min_reserve_mb - 1, -1024):
            try:
                _ballast = torch.empty(mb * 1024 * 1024, dtype=torch.uint8, device="cuda")
                del _ballast
                got = mb
                break
            except torch.OutOfMemoryError:
                continue
        log.info(f"VRAM 先取り確保 {got}MB（希望 {reserve_mb}MB）" if got
                 else f"VRAM 先取り確保に失敗（{min_reserve_mb}MB も取れず）。続行")

    # ── samples ──
    # `data.tracks` があれば **複数トラックを混ぜた joint 学習**、無ければ従来の単一 track。
    ver = dat.get("qa_version", "v004")
    global WITH_PROCEDURE
    WITH_PROCEDURE = bool(dat.get("with_procedure", False))
    log.info(f"with_procedure={WITH_PROCEDURE}")
    ext = dat.get("extract", True)
    if dat.get("tracks"):
        # ★`--train-limit N` を **トラックごとの上限**として効かせる。
        #   これが無いと multitrack では train_limit が黙って無視され、smoke が全量を構築してしまう
        #   （2026-08-07: 全量 joint の smoke が 1.5h かけて 39,986 件を作りに行った）。
        #   config の `limit: null`（全量）は書き換えず、**本番と同じ形のまま件数だけ絞れる**ようにする。
        _cap = tr.get("train_limit")
        _specs = dat["tracks"] if _cap is None else [
            dict(s, limit=(_cap if s.get("limit") is None else min(_cap, s["limit"])))
            for s in dat["tracks"]]
        if _cap is not None:
            log.info(f"train_limit={_cap} をトラックごとの上限として適用: "
                     + " / ".join(f"{s['track']} {s['limit']}" for s in _specs))
        # ★`data.seg_aug: true` で **SEGMENT の入力構成をサンプルごとにランダム化**する
        #   （`dataset_seg.SEG_AUG_CONFIGS`）。FRAME は 1 枚固定なので対象外。
        if dat.get("seg_aug"):
            from dataset_seg import SEG_AUG_CONFIGS
            for _sp in _specs:
                if _sp["track"] == "SEGMENT":
                    _sp["aug_configs"] = SEG_AUG_CONFIGS
            log.info(f"★SEGMENT 入力構成 augmentation: {SEG_AUG_CONFIGS}")
        train_samples = build_multitrack_samples(
            _specs, dat["fold"], "train", version=ver, extract=ext,
            shuffle_seed=exp["seed"])
        # eval も同じ混合比で作る（eval_loss がトラック構成で動くと best 選択が歪む）
        # ★比率は **実際に構築された件数** から取る。config の `limit` を使うと
        #   `limit: null`（＝全量）のときに None との除算で落ちる
        #   （2026-08-07 実害: 全量 joint で 1.5h かけて 39,986 件を構築した直後に TypeError）。
        from collections import Counter as _Counter
        _built = _Counter(s.track for s in train_samples)
        _tot = sum(_built.values()) or 1
        val_specs = [dict(s, limit=max(1, int(tr["eval_subset"] * _built[s["track"]] / _tot)))
                     for s in dat["tracks"]]
        log.info(f"eval 内訳（学習の混合比に合わせる）: "
                 + " / ".join(f"{s['track']} {s['limit']}" for s in val_specs))
        val_samples = build_multitrack_samples(
            val_specs, dat["fold"], "val", version=ver, extract=ext, shuffle_seed=exp["seed"])
    else:
        # ★`data.grid`（秒）。省略時はトラック既定（PROCEDURE=5.0 = 本番クリップの
        #   キーフレーム間隔 / 他=1.0）。対照を取るときだけ明示する。
        _grid = dat.get("grid")
        train_samples = build_samples(
            dat["track"], dat["fold"], "train", n_frames=dat["n_frames"],
            size=dat["frame_size"], limit=tr["train_limit"], version=ver, extract=ext,
            grid=_grid)
        val_samples = build_samples(
            dat["track"], dat["fold"], "val", n_frames=dat["n_frames"],
            size=dat["frame_size"], limit=tr["eval_subset"], version=ver, extract=ext,
            grid=_grid)
    # ── ★instseg 重畳の注入（expM00, 2026-09-01）────────────────────
    # `data.overlay` があれば事前生成した重畳画像を **2枚目の画像**として足す。
    # ★**擬似 QA より前**に当てること（擬似 QA は生成元が必ず det-train 動画なので
    #   重畳を付けてはいけない＝ここで触らないのが正しい）。
    # ★arm ゲート: 検出器が注釈を学習に使った動画（`overlay_arm_v001.csv` の CONTROL）
    #   には付けない。付けると「検出器が GT を暗記した重畳」を教師にしてしまう。
    if dat.get("overlay"):
        sys.path.insert(0, str(ROOT / "workspace/expM00_frame_overlay"))
        from overlay_attach import arm_of_video, attach
        ov = dat["overlay"]
        _cache = Path(ov["cache_root"])
        if not _cache.is_absolute():
            _cache = ROOT / _cache
        _arms = arm_of_video(ROOT / ov["arm_csv"]) if ov.get("arm_csv") else arm_of_video()
        _st = attach(train_samples, _cache, ov["variant"], arms=_arms,
                     t1=bool(ov.get("t1", False)))
        log.info("★重畳(train, %s): dual %d / 空 %d / CONTROL arm %d / 全 %d",
                 ov["variant"], _st["dual"], _st["empty"], _st["control_arm"], _st["n"])
        # ★`expect_dual` は**全量学習のときだけ**照合する。smoke は `--train-limit` で
        #   件数を絞るので、そのまま assert すると smoke が必ず落ちる（config の形は
        #   本番と同一に保つ運用なので、ここで分岐するのが正しい）。
        if ov.get("expect_dual") is not None and not tr.get("train_limit"):
            assert _st["dual"] == int(ov["expect_dual"]), (
                f"重畳が付いた件数 {_st['dual']} が期待値 {ov['expect_dual']} と違う")
        elif ov.get("expect_dual") is not None:
            log.info("（train_limit=%s のため expect_dual=%s の照合はスキップ）",
                     tr.get("train_limit"), ov["expect_dual"])
        if ov.get("val", True):
            # ★val（eval_loss 用）にも同じ規則で付ける。qa fold0 に det-train 動画は
            #   1本も無いので arm ゲートは不要だが、空検出ゲートは同じく効かせる。
            _sv = attach(val_samples, _cache, ov["variant"], arms=None,
                         t1=bool(ov.get("t1", False)))
            log.info("★重畳(val, %s): dual %d / 空 %d / 全 %d",
                     ov["variant"], _sv["dual"], _sv["empty"], _sv["n"])

    # ── ★★instseg 重畳の 3 スタイル条件付き学習（expN00, 2026-09-02）──────
    # `data.overlay3` があれば **フレームそのものを重畳版へ差し替える**（in-place。
    # expM00 の overlay=2枚渡しとは排他）。スタイル s0/s1/s2 を qID ハッシュで
    # 1/3 ずつ割り当て、CONTROL arm 動画と検出0クリップは s0 のまま。
    if dat.get("overlay3"):
        sys.path.insert(0, str(ROOT / "workspace/expN00_seg_overlay3"))
        from overlay3 import arm_map, attach3
        o3 = dat["overlay3"]
        _c3 = Path(o3["cache_root"])
        if not _c3.is_absolute():
            _c3 = ROOT / _c3
        _arms3 = arm_map(ROOT / o3["arm_csv"]) if o3.get("arm_csv") else arm_map()
        _s3 = attach3(train_samples, _c3, seed=exp["seed"], arms=_arms3)
        log.info("★重畳3(train): s1 %d / s2 %d / s0 %d / 空 %d / CONTROL %d / 全 %d",
                 _s3["s1"], _s3["s2"], _s3["s0"], _s3["empty"], _s3["control"], _s3["n"])
        if o3.get("expect_s1s2") is not None and not tr.get("train_limit"):
            assert _s3["s1"] + _s3["s2"] == int(o3["expect_s1s2"]), (
                f"重畳が付いた件数 {_s3['s1'] + _s3['s2']} が期待値 {o3['expect_s1s2']} と違う")
        if o3.get("val", True):
            _sv3 = attach3(val_samples, _c3, seed=exp["seed"], arms=None)
            log.info("★重畳3(val): s1 %d / s2 %d / s0 %d / 空 %d / 全 %d",
                     _sv3["s1"], _sv3["s2"], _sv3["s0"], _sv3["empty"], _sv3["n"])

    # ── ★擬似 VQA の注入（expK01, 2026-08-25）──────────────────────
    # `data.pseudo` があれば expK00 生成の擬似 QA を **train にだけ** 追加して再シャッフル。
    # val は実データのみ（擬似は qa fold0 動画を含まない設計だが、評価は常に実データで行う）。
    # --train-limit（smoke）は擬似側の上限としても効かせる（実データだけ絞れて擬似が全量、を防ぐ）。
    if dat.get("pseudo"):
        from pseudo_data import build_pseudo_samples
        _ps = build_pseudo_samples(dat["pseudo"], seed=exp["seed"],
                                   limit_cap=tr.get("train_limit"))
        _n_real = len(train_samples)
        train_samples = list(train_samples) + _ps
        import random as _rnd
        _rnd.Random(exp["seed"]).shuffle(train_samples)
        log.info(f"★擬似 VQA 注入: real {_n_real} + pseudo {len(_ps)} = {len(train_samples)}")
    # ── ★expE23: CoT 教師への差し替え（STaR）──────────────────────
    # ここは**擬似 QA / 重畳の注入がすべて終わった後**。入力の作り方は一切変えず、
    # ①system の OUTPUT RULE を落とす ②質問の末尾に自由記述の指示を足す
    # ③answer を「観察 + ANSWER: GT」に差し替える の3点だけを行う。
    # ★①②は**教師トレース生成時・推論時と 1 バイト単位で同じ**でなければならない
    #   （`run_freeform_full.py --instr-in-question` と同じ並び）。ずれると学習した
    #   出力形式が推論時に出てこない。
    if args.uid_list:
        assert not args.cot_targets, "--uid-list と --cot-targets は同時指定しない"
        _uids = set(json.loads(Path(args.uid_list).read_text()))
        _n0 = len(train_samples)
        train_samples = [s for s in train_samples if s.uid in _uids]
        assert train_samples, f"{args.uid_list} と train サンプルの uid が1つも交差しない"
        log.info(f"★量揃え対照: --uid-list で {_n0} → {len(train_samples)} 問に絞り込み"
                 f"（answer は差し替えない）")

    if args.cot_targets:
        sys.path.insert(0, str(ROOT / "workspace/expE22_freeform_extract"))
        from run_freeform_full import FREEFORM_HEAD, freeform_system

        _cot = json.loads(Path(args.cot_targets).read_text())
        _n0 = len(train_samples)
        train_samples = [s for s in train_samples if s.uid in _cot]
        for s in train_samples:
            s.system_prompt = freeform_system(s.question)
            s.question = s.question + "\n\n" + FREEFORM_HEAD
            s.answer = _cot[s.uid]["target"]
        # val は「入力側だけ」同じ改変を当てる（answer は GT のまま = eval loss の意味が変わる
        # ので、eval は**参考値**として扱う。採点は eval_seg.py で別途行う）
        for s in val_samples:
            s.system_prompt = freeform_system(s.question)
            s.question = s.question + "\n\n" + FREEFORM_HEAD
        assert train_samples, f"{args.cot_targets} と train サンプルの uid が1つも交差しない"
        _lens = sorted(len(s.answer) for s in train_samples)
        log.info(f"★CoT 教師に差し替え: {_n0} → {len(train_samples)} 問 "
                 f"（教師文字数 中央値 {_lens[len(_lens)//2]} / 最大 {_lens[-1]}）")
        log.info(f"  教師の例:\n{train_samples[0].answer[:300]}")

    from collections import Counter
    log.info(f"train={len(train_samples)} {dict(Counter(s.track for s in train_samples))}  "
             f"val(subset)={len(val_samples)} {dict(Counter(s.track for s in val_samples))}")

    # ── ★`enable_input_require_grads` の dtype ガード ────────────────
    # transformers 5.x の `enable_input_require_grads` は **全サブ PreTrainedModel の
    # `get_input_embeddings()` にフックを張る**（modeling_utils.py:2351）。
    # gradient_checkpointing 有効時に Trainer / peft が呼ぶ。
    # ★2026-08-17 の実害: LFM2.5-VL では vision tower(SigLIP2) の `patch_embedding` にも
    #   張られ、その出力で `output.requires_grad_(True)` が
    #   `RuntimeError: only Tensors of floating point dtype can require gradients` で落ちた。
    #   学習が1ステップも回らない。
    # ★浮動小数の出力に対しては**挙動が完全に同一**（既存の Qwen 経路は bit 不変）。
    #   非浮動小数のときだけ黙って飛ばす。
    import transformers.modeling_utils as _mu
    if not getattr(_mu.PreTrainedModel, "_orena_grad_guard", False):
        def _enable_input_require_grads(self):
            def _hook(module, inp, out):
                if torch.is_tensor(out) and out.is_floating_point():
                    out.requires_grad_(True)
            self._require_grads_hook = self.get_input_embeddings().register_forward_hook(_hook)
            for m in self.modules():
                if isinstance(m, _mu.PreTrainedModel) and m is not self:
                    try:
                        emb = m.get_input_embeddings()
                    except (NotImplementedError, AttributeError):
                        continue
                    if emb is not None:
                        emb.register_forward_hook(_hook)
        _mu.PreTrainedModel.enable_input_require_grads = _enable_input_require_grads
        _mu.PreTrainedModel._orena_grad_guard = True
        log.info("enable_input_require_grads に dtype ガードを適用（非浮動小数の出力を飛ばす）")

    # ── model (4bit QLoRA) ──
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    if cc[0] < 8 and mc["compute_dtype"] == "bfloat16":
        mc["compute_dtype"] = "float16"
        log.info(f"GPU sm_{cc[0]}{cc[1]} は bf16 非対応 → compute_dtype=float16")
    use_bf16 = (mc["compute_dtype"] == "bfloat16")
    bnb = BitsAndBytesConfig(
        load_in_4bit=mc["load_in_4bit"], bnb_4bit_quant_type=mc["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=mc["bnb_4bit_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, mc["compute_dtype"]))
    processor = AutoProcessor.from_pretrained(mc["id"])
    # ★DDP: `device_map="cuda"` は **全ランクが cuda:0 に載る**（torchrun 下で即 OOM）。
    #   LOCAL_RANK があるときは自分のGPUに固定する。単一GPUなら従来どおり。
    _lr = int(os.environ.get("LOCAL_RANK", -1))
    _dmap = {"": _lr} if _lr >= 0 else "cuda"
    if _lr >= 0:
        torch.cuda.set_device(_lr)
    # ★FlashAttention2: 長系列（96枚@448 ≒ 12.7k token）で SDPA は O(L^2) のメモリを食う。
    #   sm_80+ で使えるならメモリも速度も大きく改善する。使えない環境では黙って sdpa に落とす。
    _attn = mc.get("attn_implementation", "auto")
    if _attn == "auto":
        _attn = "flash_attention_2" if cc[0] >= 8 else "sdpa"
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            mc["id"], quantization_config=bnb, device_map=_dmap,
            attn_implementation=_attn,
            dtype=getattr(torch, mc["compute_dtype"]))
        log.info(f"attn_implementation={_attn} device_map={_dmap}")
    except (ImportError, ValueError) as e:
        log.warning(f"attn_implementation={_attn} が使えない（{type(e).__name__}: {str(e)[:120]}）→ sdpa")
        model = AutoModelForImageTextToText.from_pretrained(
            mc["id"], quantization_config=bnb, device_map=_dmap,
            attn_implementation="sdpa",
            dtype=getattr(torch, mc["compute_dtype"]))
    # ★`prepare_model_for_kbit_training` は「4bit 化されなかった全パラメータ」を**一律 fp32 に上げる**
    #   （peft/utils/other.py）。Qwen3.5-9B でも embed/lm_head の fp32 化だけで数GB を要求し、
    #   16GB カードではここで OOM する（2026-08-05 に A4000 で実測: 3.79GiB の確保に失敗）。
    #   LoRA では embed/lm_head は凍結され更新されないので fp32 にする意味が無い。
    #   ⚠️「上げてから下げる」は変換の瞬間に fp32+低精度が同時に乗って OOM する。**最初から上げない**。
    #   ※LayerNorm は fp32 のまま上げる（数値安定性）。
    if tr.get("keep_embed_lowp", False):
        skip = {id(m.weight) for m in (model.get_input_embeddings(), model.get_output_embeddings())
                if m is not None and getattr(m, "weight", None) is not None}
        kept = 0
        for p_ in model.parameters():
            p_.requires_grad = False
            if p_.dtype in (torch.float16, torch.bfloat16) and \
                    p_.__class__.__name__ != "Params4bit":
                if id(p_) in skip:
                    kept += p_.numel()
                else:
                    p_.data = p_.data.to(torch.float32)
        if tr["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        log.info(f"embed/lm_head を fp32 に上げずに保持 "
                 f"({kept/1e9:.2f}B params, 約 {kept*2/2**30:.1f}GB 節約)")
    else:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=tr["gradient_checkpointing"])

    lc = cfg["lora"]
    init_from = lc.get("init_from_adapter")
    if init_from:
        # ── 既存 adapter から継続学習（warm start）──
        # ★`PeftModel.from_pretrained(..., is_trainable=True)` を使うこと。
        #   既定は `is_trainable=False` で **LoRA 重みが凍結されたまま**読み込まれ、
        #   「学習しているのに一切変わらない」状態になる（loss は下がらず、adapter も元のまま）。
        from peft import PeftModel
        # ★相対パスはリポジトリルート基準で解決する。絶対パスを config に書くと
        #   dl1(`/data4/...`) と dl2(`/mnt/data/data4/...`) で**同じ config が使えない**。
        _p = Path(init_from)
        if not _p.is_absolute():
            _p = HERE.resolve().parents[1] / init_from
        if not (_p / "adapter_model.safetensors").exists():
            raise FileNotFoundError(f"init_from_adapter が見つからない: {_p}")
        init_from = str(_p)
        model = PeftModel.from_pretrained(model, init_from, is_trainable=True)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if n_tr == 0:
            raise RuntimeError(f"warm start で学習可能パラメータが0（is_trainable を確認）: {init_from}")
        log.info(f"★warm start: {init_from} から継続学習（trainable {n_tr/1e6:.1f}M）")
    else:
        # ★`rank_pattern` / `alpha_pattern` で **モジュール別に rank を変えられる**（PEFT 機能）。
        #   動機: 一律 r だと LoRA 容量が行列サイズ比例で language に偏る。実測の vision 比率は
        #       9B = 15.6% / 27B = 6.5%   ← 設計ではなく副作用
        #   vision tower は 9B/27B でほぼ同一（27層/hidden1152）で language だけ膨らむため。
        #   expD00 で「vision LoRA が FRAME の +0.34 の主因」と分かっているので、
        #   **総容量を据え置いたまま vision に寄せる**のは根拠のある賭け。
        #   ⚠️キーは**正規表現**（例 `'visual\..*': 128`）。alpha も同じ比で上げないと
        #     実効 lr（alpha/r）が変わって「rank の効果」と混ざる。
        _lora_kw = {}
        if lc.get("rank_pattern"):
            _lora_kw["rank_pattern"] = lc["rank_pattern"]
        if lc.get("alpha_pattern"):
            _lora_kw["alpha_pattern"] = lc["alpha_pattern"]
        if _lora_kw:
            log.info(f"★rank/alpha pattern: {_lora_kw}")
        model = get_peft_model(model, LoraConfig(
            r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
            bias="none", task_type="CAUSAL_LM", target_modules=lc["target_modules"],
            **_lora_kw))
    model.print_trainable_parameters()
    model.config.use_cache = False
    # ★モデルを乗り換えたら target_modules のマッチ数を tower 別に実測すること
    #   （Qwen 用リストを Gemma に当てると vision 0 マッチで最大の梃子が黙って無効化される）
    n_vis = sum(1 for n, p in model.named_parameters()
                if p.requires_grad and ("visual" in n or "vision" in n))
    n_lang = sum(1 for n, p in model.named_parameters()
                 if p.requires_grad and not ("visual" in n or "vision" in n))
    log.info(f"LoRA マッチ数: vision={n_vis} / language={n_lang}")
    if n_vis == 0:
        raise RuntimeError("vision 側の LoRA が 0 マッチ。本タスク最大の梃子が無効化されている")

    collator = VideoCollator(processor, mc["max_seq_len"])
    check_seq_lengths(train_samples, collator)

    # ── ★最長サンプルで forward+backward を先に試す（worst-case プローブ）──
    # 2026-08-06 の実害: PROCEDURE 24f の smoke(24件) は通ったのに、本番が **39分後に OOM**。
    # smoke がランダムな24件しか見ておらず、**最長系列を引かなかった**ため。
    # OOM は `cross_entropy` の logits (seq_len × vocab 258k × 4byte) で起きるので、
    # **系列長が最大のサンプルが通れば残りは通る**。ここで2分で落とせば39分を無駄にしない。
    k = int(tr.get("probe_longest", 3))
    if k > 0:
        lens = []
        for s in train_samples:
            n = sum(1 for _ in s.frame_paths)
            lens.append((n, s))
        lens.sort(key=lambda x: -x[0])
        probes = [s for _, s in lens[:k]]
        log.info(f"worst-case プローブ: 最長 {k} サンプル（{[len(s.frame_paths) for s in probes]} フレーム）"
                 f"で forward+backward を試す")
        model.train()
        # ★autocast で包む。Trainer は bf16=True / fp16=True で自動的に autocast するが、
        #   ここは Trainer の外なので素の forward になり、**本番と実行条件が違ってしまう**。
        #   実害(2026-08-07): 4bit 化で LayerNorm 等が fp32 に上げられるため、bf16 の重みに
        #   fp32 の活性が入り linear-attention の conv1d が
        #   `expected scalar type BFloat16 but found Float` で落ちた（本番は通るのにプローブだけ落ちる）。
        _amp = torch.autocast("cuda", dtype=getattr(torch, mc["compute_dtype"]))
        for i, s in enumerate(probes, 1):
            b = collator([s])
            b = {kk: (v.to(model.device) if hasattr(v, "to") else v) for kk, v in b.items()}
            with _amp:
                out = model(**b)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
            log.info(f"  probe {i}/{k}: seq={b['input_ids'].shape[1]} OK "
                     f"(peak {torch.cuda.max_memory_allocated()/2**30:.1f}GB)")
        del out, b
        log.info("worst-case プローブ通過")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=tr["epochs"],
        per_device_train_batch_size=tr["per_device_batch_size"],
        per_device_eval_batch_size=tr["per_device_batch_size"],
        gradient_accumulation_steps=tr["grad_accum"],
        learning_rate=float(tr["lr"]),
        warmup_ratio=tr["warmup_ratio"],
        weight_decay=tr["weight_decay"],
        lr_scheduler_type=tr["lr_scheduler"],
        bf16=use_bf16, fp16=not use_bf16,
        gradient_checkpointing=tr["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=tr["logging_steps"],
        save_steps=tr["save_steps"], save_total_limit=tr.get("save_total_limit", 2),
        save_strategy="steps",
        eval_strategy="steps", eval_steps=tr["eval_steps"],
        report_to=[], remove_unused_columns=False,
        dataloader_num_workers=tr.get("dataloader_num_workers", 4),
        seed=exp["seed"],
        load_best_model_at_end=tr.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss", greater_is_better=False,
        # ★LoRA + gradient checkpointing では「使われないパラメータ」が必ず出るので
        #   False にすると DDP が例外を投げる。True は遅いが正しい。
        ddp_find_unused_parameters=tr.get("ddp_find_unused_parameters", True),
        # ★`group_by_length` は **transformers 5.14 に存在しない**（5.x で削除）。
        #   bs=1 では無効な引数なので指定しない。bs>1 にするときは自前で長さソートすること。
    )
    if targs.load_best_model_at_end and tr["save_steps"] % tr["eval_steps"] != 0:
        raise ValueError(f"load_best_model_at_end には save_steps({tr['save_steps']}) が "
                         f"eval_steps({tr['eval_steps']}) の整数倍である必要がある")

    trainer = Trainer(model=model, args=targs,
                      train_dataset=SeqDataset(train_samples),
                      eval_dataset=SeqDataset(val_samples),
                      data_collator=collator)

    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    resume = str(ckpts[-1]) if ckpts else None
    log.info(f"training… resume={resume}")
    trainer.train(resume_from_checkpoint=resume)

    st = trainer.state
    if getattr(st, "best_model_checkpoint", None):
        log.info(f"★best checkpoint = {st.best_model_checkpoint} "
                 f"(eval_loss={st.best_metric:.4f} / 最終step={st.global_step})")
    else:
        log.info(f"最終 step の重みを保存（global_step={st.global_step}）")

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    log.info(f"saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
