"""bs>1 のパディング結合が **bs=1 と数値的に等価**かを検証する.

## なぜ必須か
可変枚数の画像＋可変長系列をパディング結合する処理は、**間違っても例外にならず
静かに壊れる**（画像とテキストの対応がずれる / labels が1トークンずれる など）。
学習が回ってしまうので気づけない。**loss の一致で検算する**。

判定:
  - bs=2 の loss ≒ 各サンプルの bs=1 loss を **トークン数で重み付けした平均**
    （HF の CausalLM loss は -100 を除くトークンの平均なので、単純平均ではない）
  - 相対誤差 1e-3 未満なら OK（bf16/fp16 の数値誤差の範囲）

Usage: GPU=0 python verify_batching.py --config config_V01_fs_aug_seg.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.resolve().parents[2] / "reference/src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_V01_fs_aug_seg.yaml")
    ap.add_argument("--n", type=int, default=4, help="検証に使うサンプル数")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(HERE / a.config))
    mc, dat = cfg["model"], cfg["data"]

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    from dataset_seg import SEG_AUG_CONFIGS, build_multitrack_samples
    from train_lora_seg import VideoCollator

    specs = [dict(s) for s in dat["tracks"]]
    for sp in specs:
        if sp["track"] == "SEGMENT" and dat.get("seg_aug"):
            sp["aug_configs"] = SEG_AUG_CONFIGS
        sp["limit"] = a.n
    samples = build_multitrack_samples(specs, dat["fold"], "val",
                                       version=dat["qa_version"], extract=False)[: a.n]
    print(f"検証サンプル {len(samples)} 件 / フレーム数 {[len(s.frame_paths) for s in samples]}")

    dtype = torch.bfloat16
    bnb = BitsAndBytesConfig(load_in_4bit=mc["load_in_4bit"], bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    proc = AutoProcessor.from_pretrained(mc["id"])
    model = AutoModelForImageTextToText.from_pretrained(
        mc["id"], quantization_config=bnb, device_map="cuda",
        attn_implementation="flash_attention_2", dtype=dtype).eval()
    col = VideoCollator(proc, mc["max_seq_len"])

    def loss_of(batch):
        enc = col(batch)
        enc = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        n_tok = int((enc["labels"] != -100).sum())
        return float(out.loss), n_tok

    print("\n=== 個別（bs=1）===")
    singles = []
    for i, s in enumerate(samples):
        l, n = loss_of([s])
        singles.append((l, n))
        print(f"  [{i}] loss={l:.6f}  学習対象トークン={n}")

    print("\n=== まとめて（bs=2 / bs=4）===")
    ok = True
    for bs in (2, len(samples)):
        if bs > len(samples):
            continue
        lb, nb = loss_of(samples[:bs])
        # HF の loss は -100 を除くトークンの平均 → トークン数で重み付けした期待値
        tot = sum(n for _, n in singles[:bs])
        exp = sum(l * n for l, n in singles[:bs]) / tot
        rel = abs(lb - exp) / max(abs(exp), 1e-9)
        mark = "✅" if rel < 1e-3 else "❌"
        ok &= rel < 1e-3
        print(f"  bs={bs}: 実測 {lb:.6f} / 期待 {exp:.6f}  相対誤差 {rel:.2e} {mark}")
    print("\n" + ("✅ バッチ結合は bs=1 と等価" if ok else
                  "❌ 不一致。bs>1 を使ってはいけない"))


if __name__ == "__main__":
    main()
