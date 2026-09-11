"""解像度 × フレーム数の実行可能フロンティアを実測する（VRAM / token / latency）.

## 目的
SEGMENT / PROCEDURE で**問題タイプごとに解像度とフレーム数を動的に選ぶ**ための土台。
「この解像度なら何枚まで乗るか」「そのとき何 token / 何秒か」を実測表にする。

## 測り方の前提（重要）
- **フレームは再抽出しない**。キャッシュ済み 448px フレームをメモリ上でリサイズして使う。
  vision token 数・VRAM・latency は**画素寸法だけで決まる**ので容量測定としては忠実
  （※中身が変わるので**精度の測定には使えない**）。
- ffmpeg の `scale={W}:-2` と同じく **幅を指定して高さは偶数に丸める**。
- **dl1 (Quadro RTX 8000 / Turing sm_75) は FlashAttention が使えず SDPA が O(L²) のメモリを食う**。
  本番の L40S(sm_89) / RTX PRO 6000(Blackwell) では FA が効くので、
  **ここで出る VRAM 上限は保守的な下限**。移植可能な量は **input token 数**の方。
- latency も同様に Turing 実測。**本番機の値ではない**（順序関係だけ信用する）。

Usage:
  GPU=2 .venv-dl1/bin/python workspace/expE01_segproc_baseline/probe_capacity.py \
      --sizes 336 448 560 672 768 1024 --frames 8 16 32 48 64 96 128 160 192 256 \
      --adapter results/expE03f_joint_all_full/fold0/adapter
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "reference/src"))
sys.path.insert(0, str(HERE))

log = logging.getLogger("probe")


def pick_dtype() -> torch.dtype:
    """★Turing では `is_bf16_supported()` がエミュレーションで True を返すので使わない。"""
    cc = torch.cuda.get_device_capability()
    return torch.bfloat16 if cc[0] >= 8 else torch.float16


def load_frames(cache_root: Path, size_src: int, n: int) -> list[Image.Image]:
    """キャッシュ済みフレームを n 枚集める（足りなければ循環して使う）。"""
    root = cache_root / str(size_src)
    paths = sorted(p for d in root.rglob("*") if d.is_dir() for p in d.glob("*.jpg"))[:512]
    if not paths:
        raise SystemExit(f"{root} に 448px のキャッシュフレームが無い")
    imgs = [Image.open(paths[i % len(paths)]).convert("RGB") for i in range(n)]
    return imgs


def resize_to_width(img: Image.Image, w: int) -> Image.Image:
    """ffmpeg `scale=w:-2` 相当（高さは偶数に丸める）。"""
    h = max(2, int(round(img.height * w / img.width / 2)) * 2)
    return img.resize((w, h), Image.BICUBIC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[336, 448, 560, 672, 768, 1024])
    ap.add_argument("--frames", type=int, nargs="+",
                    default=[8, 16, 32, 48, 64, 96, 128, 160, 192, 256])
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--src-size", type=int, default=448, help="キャッシュ済みフレームの幅")
    ap.add_argument("--cache-root", type=Path, default=HERE / "frames_cache")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "capacity_frontier")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)

    dtype = pick_dtype()
    cc = torch.cuda.get_device_capability()
    gpu = torch.cuda.get_device_name()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU={gpu} sm_{cc[0]}{cc[1]} {total_gb:.1f}GB dtype={dtype} "
             f"(FlashAttention {'あり' if cc[0] >= 8 else '**なし** → SDPA が O(L^2)'})")

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    from prompts_seg import build_system_prompt

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    processor = AutoProcessor.from_pretrained(a.base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.base_model, quantization_config=bnb, device_map="cuda", dtype=dtype).eval()
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).eval()   # ★merge しない
        log.info(f"adapter: {a.adapter}")
    base_mb = torch.cuda.memory_allocated() / 2**20
    log.info(f"モデルロード後の常駐 VRAM: {base_mb:.0f} MB")

    sysp, _ = build_system_prompt(
        "At what time was a Clip first visible in the video? "
        "Please provide an answer in the format hh:mm:ss.")
    question = ("At what time was a Clip first visible in the video? "
                "Please provide an answer in the format hh:mm:ss.")

    src = load_frames(a.cache_root, a.src_size, max(a.frames))
    rows: list[dict] = []

    for size in a.sizes:
        imgs_all = [resize_to_width(im, size) for im in src]
        log.info(f"--- size={size}px ({imgs_all[0].width}x{imgs_all[0].height}) ---")
        for nf in a.frames:
            content: list[dict] = [{"type": "text", "text": f"Video frames ({nf} frames):"}]
            for i in range(nf):
                content.append({"type": "text", "text": f"[{i:02d}:00:00]"})
                content.append({"type": "image", "image": imgs_all[i]})
            content.append({"type": "text", "text": question})
            msg = [{"role": "system", "content": sysp},
                   {"role": "user", "content": content}]

            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                text = processor.apply_chat_template(msg, tokenize=False,
                                                     add_generation_prompt=True,
                                                     enable_thinking=False)
                ii, vi = process_vision_info(msg)
                inp = processor(text=[text], images=ii, videos=vi,
                                return_tensors="pt").to(model.device)
                ntok = int(inp.input_ids.shape[1])
                t0 = time.perf_counter()
                with torch.no_grad():
                    model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
                lat = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated() / 2**20
                rows.append({"size": size, "frames": nf, "ok": True, "tokens": ntok,
                             "peak_mb": round(peak), "latency_s": round(lat, 2)})
                log.info(f"  {nf:4d}f  OK   {ntok:6d} tok  peak {peak:7.0f} MB  {lat:6.2f}s")
                del inp
            except torch.OutOfMemoryError:
                rows.append({"size": size, "frames": nf, "ok": False, "tokens": None,
                             "peak_mb": None, "latency_s": None})
                log.info(f"  {nf:4d}f  OOM  → この解像度は打ち切り")
                gc.collect(); torch.cuda.empty_cache()
                break
            except Exception as e:                       # noqa: BLE001
                rows.append({"size": size, "frames": nf, "ok": False, "error": str(e)[:200]})
                log.warning(f"  {nf:4d}f  ERR  {type(e).__name__}: {str(e)[:150]}")
                gc.collect(); torch.cuda.empty_cache()
                break
            finally:
                gc.collect(); torch.cuda.empty_cache()

        (a.out / "frontier.json").write_text(json.dumps(
            {"gpu": gpu, "sm": f"{cc[0]}{cc[1]}", "total_gb": round(total_gb, 1),
             "dtype": str(dtype), "flash_attention": cc[0] >= 8,
             "base_resident_mb": round(base_mb), "adapter": a.adapter, "rows": rows},
            indent=1, ensure_ascii=False))

    import pandas as pd
    df = pd.DataFrame([r for r in rows if r.get("ok")])
    if not df.empty:
        print("\n## 実行可能フロンティア（OK だった組み合わせ）\n")
        print(df.pivot(index="frames", columns="size", values="latency_s")
              .to_markdown(floatfmt=".2f") + "\n  ↑ latency(s)")
        print()
        print(df.pivot(index="frames", columns="size", values="tokens")
              .to_markdown(floatfmt=".0f") + "\n  ↑ input tokens")
        print()
        print(df.pivot(index="frames", columns="size", values="peak_mb")
              .to_markdown(floatfmt=".0f") + "\n  ↑ peak VRAM(MB)")
        mx = df.groupby("size").frames.max()
        print("\n### 解像度ごとの最大フレーム数\n")
        print(mx.to_markdown())
    print(f"\n→ {a.out/'frontier.json'}")


if __name__ == "__main__":
    main()
