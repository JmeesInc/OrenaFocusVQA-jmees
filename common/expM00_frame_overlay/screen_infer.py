"""重畳バリアントのスクリーニング推論（FRAME）→ responses.json.

`run_infer.py` と**同じ出力契約**（`eval_seg.py` にそのまま食わせられる）。違いは
事前生成した重畳キャッシュを `dataset_seg.build_messages` 経由で 2 枚目に足すことだけ。
★**学習と同じ `build_messages` を使う**のが要点（推論だけ別実装にすると条件がズレる）。

    python screen_infer.py --adapter <K01F adapter> --variant r1 \
        --videos-file ../expG00_vlm_grounding/leakaware_v003_videos.txt \
        --out-tag screen_r1 --shard 0/3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "workspace/expE01_segproc_baseline"))
sys.path.insert(0, str(ROOT / "reference/src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
for n in ("httpx", "httpcore", "filelock", "fsspec", "urllib3", "PIL", "datasets",
          "huggingface_hub"):
    logging.getLogger(n).setLevel(logging.WARNING)
log = logging.getLogger("expM00.screen")

from dataset_seg import build_messages, build_samples, grid_for  # noqa: E402
from overlay_render import VARIANTS  # noqa: E402
from overlay_attach import arm_of_video, attach  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="FRAME")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--part", default="val")
    ap.add_argument("--version", default="v004")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--variant", required=True,
                    # ★choices を overlay_render.VARIANTS から動的に取る（r4 追加時に
                    #   ここを更新し忘れて古い版のまま貸しGPUへ送ってしまった実害あり）
                    choices=["control", *VARIANTS])
    ap.add_argument("--t1", action="store_true",
                    help="説明文の後ろに『個数だけ』のテキストを足す（T1 アーム）")
    ap.add_argument("--cache-root", default=str(HERE / "cache"))
    ap.add_argument("--apply-arms", action="store_true",
                    help="arm 表で CONTROL 動画の重畳を落とす（**train 用**。"
                         "val fold0 には det-train 動画が無いので既定 off）")
    ap.add_argument("--videos-file", default="")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default="", metavar="i/n")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--with-procedure", action="store_true")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA が見えない。dl1 では ../../.venv-dl1/bin/python を使うこと")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    samples = build_samples(a.track, a.fold, a.part, n_frames=1, size=a.size,
                            limit=a.limit, version=a.version, grid=grid_for(a.track))
    if a.videos_file:
        keep = {x.strip().rsplit(".", 1)[0].strip()
                for x in Path(a.videos_file).read_text().splitlines() if x.strip()}
        n0 = len(samples)
        samples = [s for s in samples if s.videoID.rsplit(".", 1)[0].strip() in keep]
        log.info("videos-file: %d → %d 問（%d 動画）", n0, len(samples), len(keep))

    st = {}
    if a.variant != "control":
        st = attach(samples, a.cache_root, a.variant,
                    arms=arm_of_video() if a.apply_arms else None, t1=a.t1)
        assert st["missing"] == 0, "重畳キャッシュに欠けがある"
        assert st["dual"] > 0, "重畳が1問も付いていない（キャッシュ/variant を確認）"

    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        samples = [s for k, s in enumerate(samples) if k % n == i]
    n_dual = sum(1 for s in samples if s.overlay_path)
    log.info("variant=%s t1=%s n=%d（うち重畳あり %d = %.1f%%） shard=%s dtype=%s",
             a.variant, a.t1, len(samples), n_dual,
             100 * n_dual / max(len(samples), 1), a.shard or "-", dtype)

    out_dir = Path(a.out_dir) / (a.out_tag + (f".shard{a.shard.replace('/', 'of')}"
                                              if a.shard else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    processor = AutoProcessor.from_pretrained(a.base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.base_model, quantization_config=bnb, device_map="cuda", dtype=dtype).eval()
    from peft import PeftModel
    # ★4bit ベースに載せた LoRA を merge_and_unload してはいけない（黙って捨てられる）
    model = PeftModel.from_pretrained(model, a.adapter).eval()
    log.info("adapter %s", a.adapter)

    def gen(s):
        msg = build_messages(s, with_procedure=a.with_procedure)
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        ii, vi = process_vision_info(msg)
        inp = processor(text=[text], images=ii, videos=vi, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
        lat = time.perf_counter() - t0
        txt = processor.decode(g[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # ★★**1行目だけを採用する**（2026-09-03 実害）。
        #   外部 VQA を混ぜて学習したモデル（expM03B / expN02 / expN03）は EOS 分布が薄まり、
        #   正解のあとに `\nuser\n<think>\nassistant...` を吐き続ける。
        #   expN02 は **84.8% の問**がこれで、SCORE が 0.0013 になった（答え自体は合っていた）。
        #   提出コンテナ v011 も同じ対策（stop_strings + 1行目切断）を入れている。
        return txt.split("\n")[0].strip(), lat, int(inp.input_ids.shape[1])

    gen(samples[0])                                     # warmup
    responses, t0 = [], time.perf_counter()
    for i, s in enumerate(samples, 1):
        try:
            txt, lat, ntok = gen(s)
        except Exception as e:
            log.warning("gen failed uid=%s: %s", s.uid, e)
            txt, lat, ntok = "", 0.0, 0
        responses.append({
            "qID": s.qID, "uid": s.uid, "dataset": s.dataset, "videoID": s.videoID,
            "content": txt, "raw": txt, "latency": lat, "n_input_tokens": ntok,
            "fmt": s.fmt, "primary": s.primary, "group": s.group, "answer": s.answer,
            "question": s.question, "n_frames": len(s.frame_paths),
            "start_time": s.start_time, "end_time": s.end_time,
            "overlay": bool(s.overlay_path),
        })
        if i % 100 == 0 or i == len(samples):
            el = time.perf_counter() - t0
            log.info("%d/%d  %.2f s/q  eta %.1f min", i, len(samples), el / i,
                     (len(samples) - i) * el / i / 60)
            (out_dir / "responses.json").write_text(json.dumps(responses, indent=1))

    (out_dir / "responses.json").write_text(json.dumps(responses, indent=1))
    lats = [r["latency"] for r in responses if r["latency"] > 0]
    toks = [r["n_input_tokens"] for r in responses if r["n_input_tokens"] > 0]
    meta = {"track": a.track, "fold": a.fold, "part": a.part, "n": len(responses),
            "n_frames": 1, "size": a.size, "variant": a.variant, "t1": a.t1,
            "apply_arms": a.apply_arms, "attach_stats": st,
            "n_overlay": sum(1 for r in responses if r["overlay"]),
            "adapter": a.adapter, "base_model": a.base_model, "dtype": str(dtype),
            "with_procedure": a.with_procedure,
            "latency_mean": sum(lats) / max(len(lats), 1),
            "latency_max": max(lats, default=0.0),
            "input_tokens_mean": sum(toks) / max(len(toks), 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    log.info("wrote %s/responses.json %s", out_dir, json.dumps(
        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta.items()
         if k != "attach_stats"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
