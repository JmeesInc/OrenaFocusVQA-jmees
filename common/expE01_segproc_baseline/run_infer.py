"""SEGMENT / PROCEDURE の複数フレーム推論（zero-shot / LoRA 共通）→ responses.json.

FRAME (`expD00/eval_lora.py`) との違いは入力が **複数フレーム + 各フレームの絶対時刻テキスト** だけ。
出力形式は同じで、公式 `Evaluator` にそのまま食わせられる。

Usage:
  .venv/bin/python workspace/expE01_segproc_baseline/run_infer.py \
     --track SEGMENT --fold 0 --n-frames 16 --size 448 --out-tag zeroshot_16f448
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[2]
sys.path.insert(0, str(HERE))

from dataset_seg import build_messages, build_samples, grid_for, hhmmss  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
for n in ("httpx", "httpcore", "filelock", "fsspec", "urllib3", "PIL", "datasets",
          "huggingface_hub"):
    logging.getLogger(n).setLevel(logging.WARNING)
log = logging.getLogger("expE01.infer")


def pick_dtype() -> torch.dtype:
    """Turing(sm_75) は bf16 テンソルコアが無くエミュレーションで遅い → fp16。"""
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def clamp_time_answer(text: str, start: float, end: float) -> str:
    """time 回答を [start, end] にクランプする.

    ★根拠: 正解が区間内にある割合は SEGMENT 0.966 / PROCEDURE 1.000（expE00 実測）。
      区間外の予測は**確実に外れ**なので、端に寄せるのは常に非負の期待値を持つ後処理。
    """
    import re
    parts = [p.strip() for p in text.split(",") if p.strip()]
    out = []
    for p in parts:
        m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", p)
        if not m:
            return text                      # 形式が壊れているものは触らない
        sec = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
        out.append(hhmmss(min(max(sec, start), end)))
    return ", ".join(out) if out else text


def main() -> None:
    ap = argparse.ArgumentParser()
    # ★FRAME も許可する。joint モデルを FRAME で評価するのに必要。
    #   `dataset_seg.py` 側だけ直して**ここを直し忘れ**、チェーンが argparse エラー(rc=2)で
    #   即死したまま7時間気づかなかった（2026-08-06）。track を足すときは
    #   dataset_seg.py / run_infer.py / eval_seg.py の3箇所すべてを確認する。
    ap.add_argument("--track", default="SEGMENT", choices=["FRAME", "SEGMENT", "PROCEDURE"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--version", default="v004")
    # ★サンプリング格子（秒）。既定はトラック依存（PROCEDURE=5s = 本番クリップの
    #   キーフレーム間隔）。**meta.json に記録される**ので qa_cards が同じ格子を再現できる。
    ap.add_argument("--grid", type=float, default=None)
    # ★問題文中の hh:mm:ss のフレームを必ず含める（最も近い一様点を置換。総枚数は不変）。
    #   該当は SEGMENT 17.1% / PROCEDURE 27.8% / FRAME 3.3% のみで、他は完全に同一入力。
    ap.add_argument("--anchor", action="store_true")
    # ★刻み固定サンプリング。指定すると **問ごとに枚数を変える**
    #   （n = clamp(dur/stride + 1, n_min, --n-frames)）。SEGMENT の clip 長は
    #   29s/119s/299s の3塊なので、枚数固定だと刻みが 10倍ばらつく。
    ap.add_argument("--stride", type=float, default=None)
    ap.add_argument("--n-min", type=int, default=8)
    # ★問ごとに解像度を変える（枚数が少ない問だけ高解像度）。例: --adaptive-size 448 768
    ap.add_argument("--adaptive-size", type=int, nargs="+", default=None)
    # ★各スロットで「匿名でなく最も鮮明な」フレームを選ぶ（候補は1秒グリッド）。
    ap.add_argument("--frame-select", default="uniform",
                    choices=["uniform", "sharp", "visible", "visible_sharp", "phase", "phase_anchor", "fo", "combo", "combo2", "segrules"])
    # ★匿名化区間を本文でモデルに明示する（黙って除外しない）。
    ap.add_argument("--anon-note", action="store_true")
    # ── ★expN01: instseg 重畳3スタイルの**推論側**選択 ────────────────────
    #   学習(expN00/expN01)は s0/s1/s2 を qID ハッシュで 1/3 ずつ混ぜる。評価は
    #   **全問同一スタイル**に固定して測る（`attach3(force_style=...)` が評価用に用意されている）。
    #   ★キャッシュの解像度と `--size` は必ず一致させること。cache448 は 448x252 で焼かれており、
    #     768 で回すと重畳フレームだけ拡大されて対照との比較が壊れる（起動時に assert する）。
    ap.add_argument("--overlay3-cache", default="",
                    help="expN00 の重畳キャッシュ root（例 workspace/expN00_seg_overlay3/cache448）")
    ap.add_argument("--overlay3-style", default="s0",
                    choices=["s0", "s1", "s2", "s3", "mix", "mix2"],
                    help="s0=重畳なし / s1=bbox / s2=mask+bbox / "
                         "**s3=クラス色bbox+conf（CLAUDE.md 新規約, expP00 以降の既定）** / "
                         "mix=s0/s1/s2 の 1/3 割当 / mix2=s0/s3 の 1/2 割当（two_way 学習と同じ）")
    ap.add_argument("--overlay3-arm-csv", default="",
                    help="CONTROL arm ゲート用 CSV。空なら全動画 DUAL 扱い（fold0 val には det-train 動画が無い）")
    ap.add_argument("--overlay3-seed", type=int, default=42,
                    help="mix のときのスタイル割当シード。学習 config の experiment.seed と揃える")
    # ★`Request.procedure_type` を user ターン先頭に入れる（公式が推論時にもくれる情報）。
    ap.add_argument("--with-procedure", action="store_true")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", default="", help="空なら base のみ（zero-shot）")
    ap.add_argument("--compute-dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--no-clamp", action="store_true", help="time の区間クランプを無効化（対照用）")
    ap.add_argument("--qids", default="", help="このファイルの qID だけを推論する（1行1つ）")
    ap.add_argument("--with-conf", action="store_true",
                    help="token ごとの log-prob を responses.json に足す（greedy のまま。生成結果は不変）")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    # ★複数 GPU で1つの評価を分割する（`--shard 0/4` .. `--shard 3/4`）。
    #   1問ずつの推論では GPU が遊ぶうえ、32f@768 は 16.1GB/プロセス必要で
    #   24GB カードに2プロセスは載らない → **プロセスを増やすのではなく問題を割る**。
    #   出力は `<out_tag>.shardK` に分かれるので、`merge_shards.py` で結合してから採点する。
    ap.add_argument("--shard", default=None, help="i/N 形式。i 番目の 1/N だけを処理する")
    args = ap.parse_args()

    dtype = pick_dtype() if args.compute_dtype == "auto" else getattr(torch, args.compute_dtype)
    out_dir = Path(args.out_dir) / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = grid_for(args.track) if args.grid is None else args.grid
    samples = build_samples(args.track, args.fold, args.part, n_frames=args.n_frames,
                            size=args.size, limit=args.limit, version=args.version, grid=grid,
                            anchor=args.anchor, stride=args.stride, n_min=args.n_min,
                            adaptive_size=tuple(args.adaptive_size) if args.adaptive_size else None,
                            frame_select=args.frame_select, anon_note=args.anon_note)
    # ── ★instseg 重畳3（expN00/expN01）の適用。**フレームパスを重畳版へ差し替える**（in-place）──
    #   ★build_samples の直後・--qids や --shard の絞り込みより前に置く（学習側と同じ位置）。
    if args.overlay3_cache and args.overlay3_style != "s0":
        sys.path.insert(0, str(HERE.resolve().parent / "expN00_seg_overlay3"))
        from overlay3 import arm_map, attach3  # noqa: E402
        _c3 = Path(args.overlay3_cache)
        if not _c3.is_absolute():
            _c3 = HERE.resolve().parents[1] / _c3
        # ★解像度の一致を assert する。cache448(448x252) を --size 768 で使うと
        #   重畳フレームだけ拡大され、対照(768 素フレーム)との比較が成立しない。
        from PIL import Image as _Im
        _probe = next(_c3.rglob("*.jpg"), None)
        if _probe is None:
            raise SystemExit(f"重畳キャッシュに jpg が無い: {_c3}")
        _cw = _Im.open(_probe).size[0]
        if _cw != args.size:
            raise SystemExit(
                f"★重畳キャッシュの幅 {_cw}px と --size {args.size} が違う（{_probe.name}）。"
                f"\n  この解像度の重畳キャッシュを焼くか、--size {_cw} で回すこと。"
                f"\n  拡大して使うと重畳フレームだけ解像度が落ち、対照との比較が壊れる。")
        _arms = arm_map(HERE.resolve().parents[1] / args.overlay3_arm_csv) if args.overlay3_arm_csv else None
        # ★mix/mix2 は学習と同じハッシュ割当（force しない）。mix2 は s0/s3 の 1/2（two_way）
        _force = None if args.overlay3_style in ("mix", "mix2") else args.overlay3_style
        _two_way = (args.overlay3_style == "mix2")
        _st = attach3(samples, _c3, seed=args.overlay3_seed, arms=_arms,
                      force_style=_force, two_way=_two_way)
        log.info("★重畳3(infer, style=%s): s1 %d / s2 %d / s0 %d / 空 %d / CONTROL %d / 全 %d",
                 args.overlay3_style, _st["s1"], _st["s2"], _st["s0"], _st["empty"],
                 _st["control"], _st["n"])
        if _st["missing"]:
            raise SystemExit(f"★重畳キャッシュに無いフレームが {_st['missing']} 枚（キャッシュ不足）")

    # ★qID を絞って回す。方策が介入する問だけを走らせ、残りは対照の応答をそのまま
    #   流用する（greedy かつフレーム選択は問ごとに独立なので、流用は**厳密に同一**）。
    #   2000問 2.6h → 660問 0.9h に縮む。
    if args.qids:
        want = {ln.strip() for ln in Path(args.qids).read_text().split() if ln.strip()}
        n_before = len(samples)
        samples = [s for s in samples if str(s.qID) in want]
        log.info(f"--qids {args.qids}: {n_before} -> {len(samples)} 問")
        if not samples:
            raise SystemExit("qID が1問も一致しない（型や表記を確認）")

    if args.shard:
        # ★飛び飛び（i::N）で割る。連続ブロックだと動画が偏り、shard 間で難度が揃わない。
        si, sn = (int(x) for x in args.shard.split("/"))
        if not (0 <= si < sn):
            raise ValueError(f"--shard {args.shard} が不正")
        n_before = len(samples)
        samples = samples[si::sn]
        out_dir = Path(args.out_dir) / f"{args.out_tag}.shard{si}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"shard {si}/{sn}: {n_before} -> {len(samples)} 問  out={out_dir}")
    log.info(f"track={args.track} fold={args.fold}/{args.part} n={len(samples)} "
             f"grid={grid}s anchor={args.anchor} stride={args.stride} dtype={dtype}")

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="cuda", dtype=dtype).eval()
    if args.adapter:
        from peft import PeftModel
        # ★4bit ベースに載せた LoRA を merge_and_unload してはいけない（黙って捨てられる）
        model = PeftModel.from_pretrained(model, args.adapter).eval()
        log.info(f"loaded adapter {args.adapter}")

    def gen(s):
        msg = build_messages(s, with_procedure=args.with_procedure)
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        ii, vi = process_vision_info(msg)
        inp = processor(text=[text], images=ii, videos=vi, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            # ★--with-conf は **greedy のまま** scores を持ち帰るだけ。do_sample=False も
            #   max_new_tokens も変えないので、生成される系列は従来と同一
            #   （[[port-inference-call-verbatim]] 生成呼び出しを勝手に変えない）。
            if args.with_conf:
                o = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                                   return_dict_in_generate=True, output_scores=True)
                g = o.sequences
            else:
                g = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        lat = time.perf_counter() - t0
        txt = processor.decode(g[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        conf = None
        if args.with_conf:
            # token ごとの log p(chosen)。EOS 後の埋めが -inf で来るので**必ず有限だけ拾う**
            ts = model.compute_transition_scores(o.sequences, o.scores, normalize_logits=True)[0]
            ts = ts[torch.isfinite(ts)]
            n = int(ts.numel())
            conf = {"logp_sum": float(ts.sum()) if n else 0.0,
                    "logp_mean": float(ts.mean()) if n else 0.0,   # ★長さ正規化。系列和は長い答えを不当に罰する
                    "logp_min": float(ts.min()) if n else 0.0,     # 一番弱い token（律速）
                    "n_gen_tokens": n}
        return txt, lat, int(inp.input_ids.shape[1]), conf

    gen(samples[0])                                     # warmup（初回はカーネルコンパイルで遅い）

    responses, t_start = [], time.perf_counter()
    for i, s in enumerate(samples, 1):
        try:
            txt, lat, ntok, conf = gen(s)
        except Exception as e:                          # 1問の失敗で全体を落とさない
            log.warning(f"gen failed uid={s.uid}: {e}")
            txt, lat, ntok, conf = "", 0.0, 0, None
        raw = txt
        if s.fmt == "time" and not args.no_clamp:
            txt = clamp_time_answer(txt, s.start_time, s.end_time)
        responses.append({
            "qID": s.qID, "uid": s.uid, "dataset": s.dataset, "videoID": s.videoID,
            "content": txt, "raw": raw, "latency": lat, "n_input_tokens": ntok,
            "fmt": s.fmt, "primary": s.primary, "group": s.group, "answer": s.answer,
            "question": s.question, "n_frames": len(s.frame_paths),
            "start_time": s.start_time, "end_time": s.end_time,
            **({"conf": conf} if conf is not None else {}),   # ★--with-conf のときだけ増える
        })
        if i % 100 == 0 or i == len(samples):
            el = time.perf_counter() - t_start
            log.info(f"{i}/{len(samples)}  {el/i:.2f}s/q  eta {(len(samples)-i)*el/i/60:.1f}min")
            (out_dir / "responses.json").write_text(json.dumps(responses, indent=1))

    (out_dir / "responses.json").write_text(json.dumps(responses, indent=1))
    lats = [r["latency"] for r in responses if r["latency"] > 0]
    toks = [r["n_input_tokens"] for r in responses if r["n_input_tokens"] > 0]
    meta = {"track": args.track, "fold": args.fold, "part": args.part, "n": len(responses),
            "n_frames": args.n_frames, "size": args.size, "grid": grid, "anchor": args.anchor, "stride": args.stride,
            "frame_select": args.frame_select, "anon_note": args.anon_note,
            # ★重畳アームを meta に残す。残さないと s0/s1/s2 の run を後から見分けられない
            "overlay3_cache": args.overlay3_cache, "overlay3_style": args.overlay3_style,
            "overlay3_arm_csv": args.overlay3_arm_csv,
        "with_procedure": args.with_procedure,
            "adaptive_size": args.adaptive_size,
            "n_frames_effective_mean": sum(len(s.frame_paths) for s in samples) / max(len(samples), 1),
            "adapter": args.adapter,
            "base_model": args.base_model, "dtype": str(dtype), "clamp": not args.no_clamp,
            "latency_mean": sum(lats) / max(len(lats), 1),
            "latency_max": max(lats, default=0.0),
            "input_tokens_mean": sum(toks) / max(len(toks), 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info(f"wrote {out_dir}/responses.json  " + json.dumps(
        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items()}))


if __name__ == "__main__":
    main()
