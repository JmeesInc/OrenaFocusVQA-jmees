"""expM03 — 外部 Surgical VQA を混ぜた Round 1 学習（expE01 train_lora_seg.py のフォーク）.

★フォーク元: workspace/expE01_segproc_baseline/train_lora_seg.py（2026-09-01 01:38 時点, 35,210B）
  dataset_seg / pseudo_data / prompts_seg / overlay_attach は **expE01 の生きたコードを import**
  する（フォークしない）。expM00_frame_overlay の改良は自動で取り込まれる。
  このファイル固有の差分は3つだけ（後で expE01 へ逆マージしやすいよう最小に保つ）:
    1. `data.external` レーン: 外部 VQA parquet（SurgMLLMBench / MultiBypass）を
       train にのみ注入（external_data.py）。術式文ドロップアウトは loader 側で適用済み
    2. サンプル別 loss 重み: `VideoVQASample.meta["loss_weight"]` を collator が持ち回り、
       WeightedTrainer が loss に掛ける（外部=0.5 で「事前学習的」に効かせる）
    3. Stage A（混合 1ep）→ Stage B（FOCUS のみ 1ep, `lora.init_from_adapter`）の
       2段アニーリングは **既存の warm start 機構をそのまま使う**（コード変更なし）

実行: .venv/bin/python workspace/expM03_round1_external/train_lora_ext.py --config <yaml>
"""
from __future__ import annotations

import argparse
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
# ★expE01 は同じ workspace/ 階層の兄弟ディレクトリとして解決する（ROOT 計算に依存しない）。
#   フォーク元にあった `ROOT = HERE.resolve().parents[2]` は HERE が**ディレクトリ**なので
#   1階層上（dl1 では MICCAI2026、vast では /workspace）を指すバグだった（本家は修正済み）。
E01 = HERE.resolve().parent / "expE01_segproc_baseline"
# ★共有コードは expE01 から import（フォークは本ファイルのみ）。expM03 を先に置いて
#   external_data だけ自前を使う
sys.path.insert(0, str(E01))
sys.path.insert(0, str(HERE))


def _preload_c2_env() -> dict:
    """★expP00: `data.c2_env` を **dataset_seg の import より前**に os.environ へ流し込む.

    ⚠️`dataset_seg` の `C2_*` / `FO_*` は **モジュール読み込み時に確定する定数**
    （`dataset_seg.py:917-948, 1067-1079`）。`main()` の中で環境変数を立てても手遅れで、
    combo2 は既定値のまま動いてしまう（＝c9w のつもりが別設定で学習される事故になる）。
    そのため argv を先読みして config の `data.c2_env` をここで適用する。

    既に環境変数が立っている場合は**上書きしない**（シェル側の明示指定を優先）。
    """
    try:
        i = sys.argv.index("--config")
        cfg_path = Path(sys.argv[i + 1])
    except (ValueError, IndexError):
        return {}
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        _cfg = yaml.safe_load(f)
    env = ((_cfg or {}).get("data") or {}).get("c2_env") or {}
    applied = {}
    for k, v in env.items():
        if k in os.environ:
            continue
        os.environ[k] = str(v)
        applied[k] = str(v)
    return applied


_C2_ENV_APPLIED = _preload_c2_env()

# ★ROOT は dataset_seg のもの（ファイル基準 parents[2] = リポジトリルート）を唯一の定義とする
from dataset_seg import ROOT, build_messages, build_multitrack_samples, build_samples  # noqa: E402

# ★学習側も `procedure_type` の有無を切り替えられるようにする（config: data.with_procedure）。
#   ⚠️**学習と推論で必ず揃える**こと。片方だけ入れると入力分布がズレる。
WITH_PROCEDURE = False  # noqa: E402
# ★★expP01: 動画入力にするトラック（例 {"PROCEDURE"}）。空なら従来どおり全部画像列。
#   config の `data.video_tracks` から main() で設定する。
VIDEO_TRACKS: set = set()
# 動画のネイティブ時刻表記は `<1234.5 seconds>`。回答形式 hh:mm:ss との対応表を添えるか。
VIDEO_HHMMSS = True


def _apply_logits_to_keep(inputs: dict) -> dict:
    """★expP00: 損失に要る logits は答え部分だけ。**プローブと compute_loss の両方**がこれを通る.

    Qwen3.5 は vocab=248,320 と巨大で、seq 8,793 の logits を fp32 化すると 8.12 GiB を
    一括確保して 32GB カードで OOM する（cross_entropy 内）。labels は
    `labels[:, :mask_len] = -100` で答え以外がマスク済みなので、末尾 K = seq_len - keep_from + 1
    個だけ logits を作れば**数学的に等価**（4 サンプルで |Δ|=0.00e+00 を実測、2026-09-04）。
    ★logits を削ったら labels も同じだけ削る（HF の ForCausalLMLoss は labels を
      sliced logits の長さに合わせない）。
    ⚠️2026-09-04 の実害: worst-case プローブが `model(**b)` を直接呼び `_keep_from` を
      未知 kwarg として**黙って素通り**させ、本番では効くのにプローブだけ OOM した（偽陰性）。
    """
    keep_from = inputs.pop("_keep_from", None)
    if keep_from is not None:
        seq_len = int(inputs["input_ids"].shape[1])
        K = max(2, seq_len - int(keep_from) + 1)
        if K < seq_len:
            inputs["logits_to_keep"] = K
            if "labels" in inputs:
                inputs["labels"] = inputs["labels"][:, -K:]
    return inputs

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
        # ★expM03: サンプル別 loss 重み（[bs] のベクトルに連結）
        if "loss_weight" in items[0]:
            out["loss_weight"] = _t.cat([x["loss_weight"] for x in items], dim=0)
        # ★expP00: logits を作る範囲。**左パディング後の位置**に換算して最小を採る。
        #   元の _keep_from は各サンプル内の位置なので、パディング分だけ右にずれる。
        if "_keep_from" in items[0]:
            shifted = [int(x["_keep_from"]) + (maxlen - int(x["input_ids"].shape[1]))
                       for x in items]
            out["_keep_from"] = min(shifted)
        # 取りこぼしがあれば気づけるようにする（黙って落とさない）
        known = set(self.SEQ_KEYS) | {"pixel_values", "image_grid_thw",
                                      "pixel_values_videos", "video_grid_thw",
                                      "loss_weight", "_keep_from"}
        missing = [k for k in items[0] if k not in known]
        if missing:
            raise RuntimeError(
                f"VideoCollator._merge が扱えないキー {missing}。"
                "bs>1 で黙って落とすと学習が壊れるので明示的に落とす。")
        return out

    def _one_video(self, s):
        """動画入力版。`_one` と **同じ責務**（full / prompt を作って labels のマスク長を出す）。

        ★prompt 側も同じ video を渡す。渡さないと `<|video_pad|>` の展開数が変わり
          「prompt が full の接頭辞」という前提が崩れて mask_len がズレる。
        """
        import sys as _sys
        _vp = str(ROOT / "workspace/expP01_video_input")
        if _vp not in _sys.path:
            _sys.path.insert(0, _vp)
        from video_input import build_messages_video, build_metadata, load_frames

        msg_prompt = build_messages_video(s, with_procedure=WITH_PROCEDURE,
                                          with_hhmmss=VIDEO_HHMMSS)
        msg_full = msg_prompt + [{"role": "assistant", "content": s.answer}]
        prompt_text = self.processor.apply_chat_template(
            msg_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full_text = self.processor.apply_chat_template(
            msg_full, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        vid = load_frames(s, getattr(s, "target_size", None))
        meta = build_metadata(s, vid)
        kw = dict(videos=[vid], video_metadata=[meta], return_tensors="pt",
                  do_sample_frames=False)
        full = self.processor(text=[full_text], **kw)
        prompt = self.processor(text=[prompt_text], **kw)
        return self._finish(s, full, prompt)

    def _one(self, s):
        # ★★expP01: `VIDEO_TRACKS` に入っているトラックは **動画入力**で渡す。
        #   画像列は 1 フレーム = 1 画像トークン束だが、動画は `temporal_patch_size=2` で
        #   2 枚が 1 時間パッチに畳まれ **トークンが約半分**になる（448px 実測 0.527 倍）。
        #   同じ系列長で 2 倍のフレームが入るので temporal_grounding を狙える。
        #   ⚠️`do_sample_frames=False` を渡さないと processor が fps=24 で勝手に間引く。
        #   ⚠️フレームは (i,i+1) で対にされ時刻は対の中点になるので **昇順必須**（load_frames が assert）。
        if VIDEO_TRACKS and s.track in VIDEO_TRACKS:
            return self._one_video(s)
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
        return self._finish(s, full, prompt)

    def _finish(self, s, full, prompt):
        """`_one` / `_one_video` の共通末尾。**画像列版と動画版で完全に同じ処理**にする
        （片方だけ直すと A/B の差が入力形式以外にも入ってしまう）。"""
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
        # ★expP00(2026-09-04): 損失に要る logits は **答え部分だけ**。
        #   Qwen3.5 は vocab_size=248,320 と巨大で、seq 8,793 の logits を fp32 化すると
        #   **8.13 GiB** を一括確保して 32GB カードで OOM する（cross_entropy 内で発生）。
        #   labels は既に `labels[:, :mask_len] = -100` で答え以外をマスク済みなので、
        #   末尾 (seq_len - mask_len + 1) 個の位置だけ logits を作れば**数学的に等価**。
        #   （+1 は「1つ前の位置が次を予測する」シフトのぶん）
        out["_keep_from"] = int(mask_len)
        # ★expM03: サンプル別 loss 重み（既定 1.0。外部レーンは meta["loss_weight"]=0.5 等）
        out["loss_weight"] = torch.tensor(
            [float(getattr(s, "meta", {}).get("loss_weight", 1.0))])
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
            if "_keep_from" in out:
                out["_keep_from"] = min(int(out["_keep_from"]), self.max_seq_len - 1)
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
    ap.add_argument("--config", default=str(HERE / "config_M03A_frame_stageA.yaml"))
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--track", default=None)
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
    # ★`experiment.out_root`（リポジトリルート相対）があればそちらへ出す。
    #   既定は trainer のあるディレクトリ = expM03。**フォークして別 exp を回すと
    #   成果物が expM03 の results に紛れ込み、Stage B の `init_from_adapter`
    #   （自分の exp フォルダを指す）が存在せず warm start に失敗する**（2026-09-02 expN01）。
    #   `init_from_adapter` はリポジトリルート相対（L532）なので、揃えておく必要がある。
    _root = exp.get("out_root")
    out_dir = ((HERE.resolve().parents[1] / _root) if _root else HERE) / "results" / exp["name"] / tag
    # 同名は _001.. で採番（上書き禁止）。**adapter の有無だけで完了判定する**
    # （checkpoint の有無で判定すると、完走した smoke の checkpoint から resume してしまう）
    if out_dir.exists() and (out_dir / "adapter").exists():
        i = 1
        while (p := out_dir.parent / f"{tag}_{i:03d}").exists():
            i += 1
        out_dir = p
    setup_logging(out_dir)
    shutil.copy(args.config, out_dir / "config.yaml")
    # ★expP00: combo2 の設定は import 前に適用済み。**何が効いたかをログに残す**
    #   （効いていない環境変数を「効いたつもり」で読むのが一番危ない）。
    if _C2_ENV_APPLIED:
        log.info("★c2_env を適用（dataset_seg import 前）: %s",
                 " ".join(f"{k}={v}" for k, v in sorted(_C2_ENV_APPLIED.items())))
    import dataset_seg as _DS
    log.info("★combo2 実効値: C2_INTERVAL=%s C2_BRIDGE=%s C2_NONTIME=%s C2_EDGE=%s "
             "C2_EDGE_WIN=%s C2_EDGE_SHARE=%s C2_EDGE_LADDER=%s C2_AFTER=%s "
             "C2_MULTICLASS=%s C2_RULES=%s",
             _DS.C2_INTERVAL, _DS.C2_BRIDGE, _DS.C2_NONTIME, _DS.C2_EDGE,
             _DS.C2_EDGE_WIN, _DS.C2_EDGE_SHARE, _DS.C2_EDGE_LADDER, _DS.C2_AFTER,
             _DS.C2_MULTICLASS, ",".join(sorted(_DS.C2_RULES)) or "(なし)")
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
    # ★★expP01: 動画入力の対象トラック。**multitrack / 単一トラックの両分岐より前**で読む。
    #   （2026-09-09: multitrack 分岐の中に置いていたため、`data.track` を使う単一トラック
    #     構成では `video_tracks` が**黙って無視される**状態だった。設定が効かない不具合は
    #     ログにも出ないので、分岐の外で 1 回だけ読んで必ずログに出す。）
    global VIDEO_TRACKS, VIDEO_HHMMSS
    VIDEO_TRACKS = set(dat.get("video_tracks", []) or [])
    VIDEO_HHMMSS = bool(dat.get("video_hhmmss", True))
    log.info("★入力形式: 動画にするトラック=%s / hh:mm:ss 対応表=%s",
             sorted(VIDEO_TRACKS) or "なし（全部 画像列）", VIDEO_HHMMSS)
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
        # ★expN04: `part` を config で選べるようにする（既定は従来どおり "train"）。
        #   `all` は train+val を連結する ＝ **fold0 val も学習に入れる**（CV は測れなくなる）。
        # ★expN04: 質問種別で入力構成を振り分ける（推論の router と同じ規則を学習にも適用）。
        #   spec に `router_groups: [...]` があると、**その group の問だけ**をその spec が担当する。
        #   ⚠️**spec ごとに build して、その場で絞る**こと。まとめて build してから
        #   `len(frame_paths)` で spec を逆引きする実装は**壊れる**: 短いクリップは実効枚数が
        #   減る（32枚指定でも 30/31枚、16枚指定でも 11枚）ので照合が外れ、
        #   「対象外」として無条件採用されてしまう（2026-09-03 の smoke で実害）。
        _part = dat.get("train_part", "train")
        _parts = ["train", "val"] if _part == "all" else [_part]
        _rt = None
        if any("router_groups" in sp for sp in _specs):
            sys.path.insert(0, str(ROOT / "submit/v005_segment_hybrid"))
            from router import GroupRouter  # noqa: E402
            _rt = GroupRouter.load(ROOT / "submit/v005_segment_hybrid/resources/group_templates.json")
        train_samples = []
        for sp in _specs:
            has_groups = "router_groups" in sp
            # ★★router_groups を持つ spec は build_multitrack_samples が **SEGMENT 全問**
            #   （振り分け前）を対象に組み立てる。`extract=True` のままだと、後で捨てる
            #   分のフレームまで ffmpeg 抽出しようとし、貸しGPU では生動画が無いので
            #   `FileNotFoundError` で全ランク即死する（2026-09-03 expN04 実害）。
            #   同時に **転送量・抽出時間も 2 倍**になる（振り分け後の集合しか使わないのに
            #   両方の構成で全問ぶん抽出するため）。
            #   → **必ず extract=False**。フレームは「振り分け後に実際に使う分」だけを
            #     事前転送しておく前提（`compose_hybrid.py` 系と同じく router.py に委譲した
            #     フレームリストを xfer で送る）。対象外グループの問はフレームが無くて
            #     自然に 0 枚→ build_samples 内部で落ちるので実害はない。
            got = []
            for pt in _parts:
                got += build_multitrack_samples([sp], dat["fold"], pt, version=ver,
                                                extract=(False if has_groups else ext),
                                                shuffle_seed=exp["seed"])
            if _rt is not None and has_groups:
                want = set(sp["router_groups"])
                n0 = len(got)
                got = [x for x in got if _rt.group_of(x.question) in want]
                # ★短いクリップは grid 制約で n_frames に届かないのが**正常**
                #   （29.0s クリップ + grid 1.0s なら最大 30 枚。32 枚要求でも 30 枚が natural max）。
                #   [[小バケットの単発改善はノイズ]]と同型の早とちりで、一度は
                #   「フレーム不足=転送漏れ」と誤診して RuntimeError にした（qID=268197 で実測: 
                #   extract=False のローカル再現でも同じ 30/32 枚 → 転送は正しい）。
                #   ★実際の取りこぼし検知は xfer 側の枚数 assert（送信前後で完全一致）に委ねる。
                eff = [len(x.frame_paths) for x in got]
                exp_nf = int(sp.get("n_frames", 16))
                log.info("★router 振り分け [%s %d枚@%spx] %d → %d 問 (%s)  "
                         "実効フレーム数 mean=%.1f min=%d max=%d",
                         sp["track"], exp_nf, sp.get("size", 448),
                         n0, len(got), "/".join(sorted(want)),
                         (sum(eff)/len(eff) if eff else 0), (min(eff) if eff else 0), (max(eff) if eff else 0))
            else:
                log.info("★[%s %d枚@%spx] %d 問（振り分けなし）",
                         sp["track"], sp.get("n_frames", 16), sp.get("size", 448), len(got))
            train_samples += got
        import random as _r
        _r.Random(exp["seed"]).shuffle(train_samples)
        log.info("★学習セット計 %d 問（part=%s）", len(train_samples), _part)
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
        val_samples = ([] if not tr.get("eval_steps") else build_multitrack_samples(
            val_specs, dat["fold"], "val", version=ver, extract=ext, shuffle_seed=exp["seed"]))
        if not tr.get("eval_steps"):
            log.info("★eval_steps=0 → 評価を行わない（held-out が無いので best 選択も無意味）")
    else:
        # ★`data.grid`（秒）。省略時はトラック既定（PROCEDURE=5.0 = 本番クリップの
        #   キーフレーム間隔 / 他=1.0）。対照を取るときだけ明示する。
        _grid = dat.get("grid")
        # ★expP00: 推論側のフレーム検索を学習にも効かせる。
        #   `data.frame_select`（uniform / combo2 / segrules / fo / phase ...）と
        #   `data.anchor` を **train と val の両方**へ同じ値で渡す。
        #   combo2 の挙動は C2_* 環境変数で決まるので、`data.c2_env` を config に置き、
        #   上流（main 冒頭）で os.environ へ流し込んである（再現性のため config が単一の情報源）。
        _fsel = str(dat.get("frame_select", "uniform"))
        _anc = bool(dat.get("anchor", False))
        log.info("★フレーム選択: frame_select=%s / anchor=%s / grid=%s", _fsel, _anc, _grid)
        # ★expQ00: 単一トラックでも `data.train_part: all` を効かせる（2026-09-04）。
        #   これが無いと FRAME は multitrack 分岐を通らないので **train_part が黙って無視される**。
        #   `all` = train + val を連結 ＝ **fold0 val も学習に入れる**（CV は測れなくなる）。
        #   ⚠️ 既定（"train"）のときは連結もシャッフルもしないので**既存実験と数値一致**。
        _part = dat.get("train_part", "train")
        _parts = ["train", "val"] if _part == "all" else [_part]
        train_samples = []
        for pt in _parts:
            train_samples += build_samples(
                dat["track"], dat["fold"], pt, n_frames=dat["n_frames"],
                size=dat["frame_size"], limit=tr["train_limit"], version=ver, extract=ext,
                grid=_grid, frame_select=_fsel, anchor=_anc)
        if len(_parts) > 1:
            import random as _r0
            _r0.Random(exp["seed"]).shuffle(train_samples)
            log.info("★学習セット計 %d 問（part=%s: %s を連結）", len(train_samples),
                     _part, "+".join(_parts))
        # ★`eval_steps: 0` なら val は作らない。train_part=all では held-out が無く、
        #   eval_loss は学習データ上の値になるので best 選択が意味を失う
        #   （[[load-best-picks-noise-argmin]]: 平坦なノイズの argmin を拾って学習の 76% を捨てた）。
        val_samples = ([] if not tr.get("eval_steps") else build_samples(
            dat["track"], dat["fold"], "val", n_frames=dat["n_frames"],
            size=dat["frame_size"], limit=tr["eval_subset"], version=ver, extract=ext,
            grid=_grid, frame_select=_fsel, anchor=_anc))
        if not tr.get("eval_steps"):
            log.info("★eval_steps=0 → 評価を行わない（held-out が無いので best 選択も無意味）")
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
        # ★expM03: --train-limit（smoke）では母数が変わるので全量時のみ assert
        if ov.get("expect_dual") is not None and tr.get("train_limit") is None:
            assert _st["dual"] == int(ov["expect_dual"]), (
                f"重畳が付いた件数 {_st['dual']} が期待値 {ov['expect_dual']} と違う")
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
        # ★expP00(2026-09-04): `two_way: true` で **s0/s3 の 1/2 割当**にする。
        #   3スタイル 1/3 ずつだと重畳ありの学習量が薄まるため。既定 false ＝ expN00/N01 と同一。
        _tw = bool(o3.get("two_way", False))
        _s3 = attach3(train_samples, _c3, seed=exp["seed"], arms=_arms3, two_way=_tw)
        log.info("★重畳3(train, two_way=%s): s1 %d / s2 %d / s3 %d / s0 %d / 空 %d / "
                 "CONTROL %d / 全 %d", _tw, _s3["s1"], _s3["s2"], _s3["s3"], _s3["s0"],
                 _s3["empty"], _s3["control"], _s3["n"])
        # ★件数をピン留めして「重畳が黙って付かない」事故を防ぐ（[[container-test-passes-while-failing]]）
        _drawn = _s3["s1"] + _s3["s2"] + _s3["s3"]
        if o3.get("expect_drawn") is not None and not tr.get("train_limit"):
            assert _drawn == int(o3["expect_drawn"]), (
                f"重畳が付いた件数 {_drawn} が期待値 {o3['expect_drawn']} と違う")
        elif o3.get("expect_s1s2") is not None and not tr.get("train_limit"):
            assert _s3["s1"] + _s3["s2"] == int(o3["expect_s1s2"]), (
                f"重畳が付いた件数 {_s3['s1'] + _s3['s2']} が期待値 {o3['expect_s1s2']} と違う")
        if _drawn == 0:
            raise SystemExit("★重畳が1件も付いていない。cache_root / arm_csv / "
                             "レンダ済みキャッシュの整合を確認すること")
        if o3.get("val", True):
            _sv3 = attach3(val_samples, _c3, seed=exp["seed"], arms=None, two_way=_tw)
            log.info("★重畳3(val): s1 %d / s2 %d / s3 %d / s0 %d / 空 %d / 全 %d",
                     _sv3["s1"], _sv3["s2"], _sv3["s3"], _sv3["s0"], _sv3["empty"], _sv3["n"])

    # ★expN01: 上のブロックは `expE01/train_lora_seg.py` から**逐語で移植**した
    #   （[[port-inference-call-verbatim]]: 移植は呼び出しごと写す）。
    #   位置も同じ＝**擬似・外部の注入より前**なので `expect_s1s2` は expN00 の
    #   実測値（実 FOCUS 31,986 サンプルに対する 11,466）がそのまま使える。
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
    # ── ★外部 Surgical VQA の注入（expM03, 2026-09-01）──────────────
    # `data.external` があれば SurgMLLMBench / MultiBypass 等の外部 QA を **train にだけ**追加。
    # 術式文ドロップアウトと loss_weight は external_data.py 側で行毎に付与済み。
    # val は常に FOCUS 実データのみ（外部を val に入れる経路は存在しない）。
    if dat.get("external"):
        from external_data import build_external_samples
        _ex = build_external_samples(dat["external"], seed=exp["seed"],
                                     limit_cap=tr.get("train_limit"))
        _n0 = len(train_samples)
        train_samples = list(train_samples) + _ex
        import random as _rnd2
        _rnd2.Random(exp["seed"] + 1).shuffle(train_samples)
        log.info(f"★外部 VQA 注入: {_n0} + external {len(_ex)} = {len(train_samples)}")
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
    # ★★DDP: ここで **device 0 を決め打ちで触ってはいけない**。
    #   `get_device_capability(0)` は呼んだプロセスに **GPU0 の CUDA コンテキストを作る**ので、
    #   torchrun の全ランクが GPU0 に 386MiB ずつ居座り、**GPU0 だけ ~1.16GiB 使える量が減る**。
    #   2026-09-02 expN01 Stage B が step 1694/2481 で OOM（3.33GiB 要求に対し空き 3.26GiB、
    #   **不足はわずか 70MiB**）。他3ランクの居候分を返せば 16 倍の余裕ができる。
    #   → 自分の local rank のデバイスを先に固定してから問い合わせる。
    _lr = int(os.environ.get("LOCAL_RANK", -1))
    if _lr >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(_lr)
    cc = (torch.cuda.get_device_capability(_lr if _lr >= 0 else 0)
          if torch.cuda.is_available() else (0, 0))
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
    _dmap = {"": _lr} if _lr >= 0 else "cuda"   # _lr / set_device は上で確定済み
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
            b.pop("loss_weight", None)  # expM03: model.forward は未知 kwarg を受けない
            b = _apply_logits_to_keep(b)  # ★本番(compute_loss)と同じ経路にする
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
        # ★`eval_steps: 0` で評価を止められるようにする。
        #   `train_part: all`（held-out 無し）では eval_loss は学習データ上の値で
        #   「best」が意味を持たない（[[load-best-picks-noise-argmin]]）。
        **({"eval_strategy": "no"} if not tr.get("eval_steps")
           else {"eval_strategy": "steps", "eval_steps": tr["eval_steps"]}),
        report_to=[], remove_unused_columns=False,
        dataloader_num_workers=tr.get("dataloader_num_workers", 4),
        # ★expP00(2026-09-04): PROCEDURE は 1 サンプル 64 枚の JPEG を開いて前処理するため
        #   前処理律速になりやすい。worker を増やすだけでは足りず、**先読みと worker 常駐**が要る
        #   （epoch 境界で毎回 worker を作り直すと 64 枚 × 大量サンプルの立ち上がりが重い）。
        dataloader_prefetch_factor=(tr.get("dataloader_prefetch_factor")
                                    if tr.get("dataloader_num_workers", 4) > 0 else None),
        dataloader_persistent_workers=bool(tr.get("dataloader_persistent_workers", False)),
        dataloader_pin_memory=bool(tr.get("dataloader_pin_memory", True)),
        # ★★DDP では **rank0 の GPU だけ他ランクの CUDA コンテキストに ~1.16GiB 食われる**
        #   （2026-09-02 実測: 居候は物理GPU0ではなく「rank0 が使う GPU」に付く。
        #     CUDA_VISIBLE_DEVICES を回しても付いて回るので割当の入れ替えでは避けられない）。
        #   その分 rank0 だけ長い系列で OOM しやすい。定期的に empty_cache して
        #   「確保済みだが未使用」の断片を返させる（OOM 時 309MiB あった＝不足 82MiB の 3.7倍）。
        #   ★メモリ管理のノブなので**学習の数値には影響せず、resume も壊さない**。
        torch_empty_cache_steps=tr.get("torch_empty_cache_steps", None),
        seed=exp["seed"],
        load_best_model_at_end=tr.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss", greater_is_better=False,
        # ★LoRA + gradient checkpointing では「使われないパラメータ」が必ず出るので
        #   False にすると DDP が例外を投げる。True は遅いが正しい。
        ddp_find_unused_parameters=tr.get("ddp_find_unused_parameters", True),
        # ★`group_by_length` 引数は transformers 5.x で `train_sampling_strategy="group_by_length"` に
        #   改名された（削除ではない）。ただし datasets.Dataset 以外では長さを自動推定できないので、
        #   expR00 では WeightedTrainer._get_train_sampler を上書きして自前の長さ proxy を渡す
        #   （config `train.group_by_length: true` で有効。既定 off = 従来どおり RandomSampler）。
    )
    if targs.load_best_model_at_end and not tr.get("eval_steps"):
        raise ValueError("eval_steps=0（評価なし）で load_best_model_at_end は成立しない")
    if targs.load_best_model_at_end and tr["save_steps"] % tr["eval_steps"] != 0:
        raise ValueError(f"load_best_model_at_end には save_steps({tr['save_steps']}) が "
                         f"eval_steps({tr['eval_steps']}) の整数倍である必要がある")

    # ── ★expM03: サンプル別 loss 重み ────────────────────────────────
    # 重みが全て 1.0 のバッチ（FOCUS/擬似のみ＝Stage B もここ）は **親クラスの経路をそのまま通す**
    # ＝ 既存実験と数値一致。重み付きバッチだけ per-sample CE を計算して重み平均する。
    import torch.nn.functional as _F

    # ★expR00(2026-09-04): 系列長 proxy。トークン化せずに「並べ替えのための目安」だけ出す。
    #   実測: 32f@448 で max 6,197 ≒ 180 tok/枚 + 本文。FRAME(1枚) は system prompt 込みで 300〜900。
    def _len_proxy(smp) -> int:
        return 180 * len(getattr(smp, "frame_paths", []) or []) + 400 + len(getattr(smp, "question", "") or "") // 3

    class WeightedTrainer(Trainer):
        # ★expR00(2026-09-04): **長さでグループ化した sampler**。
        #   DDP では accelerate が batch i を rank i%world に配るので、系列長がばらつく
        #   （p50 948 / max 6,197）と長いサンプルを引いた rank に他 rank が all-reduce で待たされる。
        #   4x4090 の実測で GPU util が 20〜30% の時間 0% に落ちていた（ストラグラー）。
        #   連続 index を同程度の長さに揃えれば 4 rank が揃って終わる。
        #   変わるのは**訪問順だけ**（データ・有効バッチ・lr・step 数は同一）。
        #   HF 標準の LengthGroupedSampler（megabatch 内ソート＋ランダム）を長さ proxy 付きで使う。
        def _get_train_sampler(self, train_dataset=None):
            if not tr.get("group_by_length"):
                return super()._get_train_sampler(train_dataset)
            from transformers.trainer_pt_utils import LengthGroupedSampler
            ds = train_dataset if train_dataset is not None else self.train_dataset
            lengths = [_len_proxy(x) for x in ds.samples]
            # 1 optimizer step = bs × accum × world 個の連続 index。この単位で長さを揃える
            bsz = (self.args.train_batch_size * self.args.gradient_accumulation_steps
                   * max(1, int(getattr(self.args, "world_size", 1))))
            log.info("★group_by_length: LengthGroupedSampler(batch=%d) proxy min=%d p50=%d max=%d n=%d",
                     bsz, min(lengths), sorted(lengths)[len(lengths)//2], max(lengths), len(lengths))
            return LengthGroupedSampler(bsz, lengths=lengths)

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            w = inputs.pop("loss_weight", None)
            # ★expP00: 損失に要る logits は答え部分だけ。Qwen3.5 は vocab=248,320 と巨大で、
            #   seq 8,793 の logits を fp32 化すると **8.13 GiB** を一括確保して OOM する
            #   （32GB カードで実測。cross_entropy 内で発生）。
            #   labels は `labels[:, :mask_len] = -100` で答え以外がマスク済みなので、
            #   末尾 K = seq_len - keep_from + 1 個だけ logits を作れば**数学的に等価**。
            #   ★logits を削ったら labels も同じだけ削る（HF の ForCausalLMLoss は
            #     labels を sliced logits の長さに合わせないので、渡す前にこちらで揃える）。
            keep_from = inputs.get("_keep_from", None)
            # ★expR00: 等価性の実測（env R00_KEEP_EQUIV_TEST=1 のときだけ、最初の 2 optimizer step）。
            #   同じ入力で「全 logits」と「末尾 K だけ」の loss を突き合わせてログに出す。
            #   本走では env を立てないので経路は従来どおり。
            #   ★`ref` は **ヘルパを当てる前**の全 logits 入力で取る（順序を間違えると
            #     keep vs keep の比較になり常に diff=0 でテストが無意味化する）。
            equiv = (bool(os.environ.get("R00_KEEP_EQUIV_TEST")) and keep_from is not None
                     and int(self.state.global_step) < 2)
            ref = None
            if equiv:
                full = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
                full.pop("_keep_from", None)   # 全 logits 経路（model は _keep_from を知らない）
                with torch.no_grad():
                    ref = self._wloss(model, full, w, return_outputs=False, **kw)
            # ★プローブと同じヘルパ（唯一の実装）。ここでしか logits_to_keep は付けない
            inputs = _apply_logits_to_keep(inputs)
            K = inputs.get("logits_to_keep")
            res = self._wloss(model, inputs, w, return_outputs=return_outputs, **kw)
            if equiv:
                lv = float(res[0] if return_outputs else res)
                log.info("★keep-equiv step=%d rank=%s: full=%.6f keep=%.6f |diff|=%.2e  K=%s seq=%d w=%s",
                         int(self.state.global_step), os.environ.get("RANK", "?"), float(ref), lv,
                         abs(float(ref) - lv), K, int(inputs["input_ids"].shape[1]),
                         None if w is None else [round(float(x), 2) for x in w])
            return res

        def _wloss(self, model, inputs, w, return_outputs=False, **kw):
            if w is None or bool((w == 1.0).all()):
                return super().compute_loss(model, inputs,
                                            return_outputs=return_outputs, **kw)
            labels = inputs.pop("labels")
            out = model(**inputs)
            losses = []
            for i in range(labels.shape[0]):
                tgt = labels[i, 1:]
                tok = _F.cross_entropy(out.logits[i, :-1].float(), tgt,
                                       ignore_index=-100, reduction="none")
                mask = (tgt != -100)
                n_tok = int(mask.sum())
                if n_tok == 0:
                    continue
                # ★expM04: **EOS（answer 最終トークン）だけは常に重み 1.0** で教える。
                #   Round 1 で外部(重み0.5)混合後に EOS 暴走が 46% 発生（expM03 検死）。
                #   短い外部回答では EOS が損失の大きな割合を占めるため、重み減衰が
                #   EOS 分布を直撃した疑い。内容は弱く・停止は強く、に分離する。
                tw = torch.full_like(tok, float(w[i]))
                last = int(mask.nonzero()[-1])
                tw[last] = 1.0
                losses.append((tok * tw)[mask].sum() / n_tok)
            loss = torch.stack(losses).mean()
            return (loss, out) if return_outputs else loss

    trainer = WeightedTrainer(model=model, args=targs,
                              train_dataset=SeqDataset(train_samples),
                              eval_dataset=(SeqDataset(val_samples) if val_samples else None),
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
