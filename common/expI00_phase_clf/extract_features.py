r"""stage-1 を freeze して全動画の 1 fps 特徴を書き出す（stage-2 の入力）.

出力: `results/<experiment_name>/fold<N>/features/<videoID>.npz`
  - `feat`  (T, D) float16  … pooled backbone 特徴
  - `logit` (T, C) float16  … phase logit（TCN の初期値／単体評価にも使う）
  - `label` (T,)   int16    … GT phase
  - `sec`   (T,)   int32    … 秒（= フレーム番号 / 25）

TCN は**本番の走査刻み（5s / 10s / 20s …）へサブサンプルした系列**で学習するので、
ここでは最も細かい 1 fps のまま保存し、間引きは `train_tcn.py` 側で行う。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset as D  # noqa: E402
from model import PhaseNet  # noqa: E402

log = logging.getLogger("feats")


def find_best(cfg: dict, fold: int) -> Path:
    base = HERE / "results" / cfg["experiment"]["name"] / f"fold{fold}"
    cands = sorted(base.parent.glob(f"{base.name}*"))
    for c in reversed(cands):
        if (c / "best_model.pt").exists():
            return c / "best_model.pt"
    raise FileNotFoundError(f"best_model.pt が見つからない: {base}*")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / args.config).read_text())
    ds, fold = cfg["data"]["dataset"], int(cfg["cv"]["fold"])
    size = int(cfg["data"]["img_size"])
    weights = Path(args.weights) if args.weights else find_best(cfg, fold)
    outdir = weights.parent / "features"
    outdir.mkdir(parents=True, exist_ok=True)
    log.info("weights=%s -> %s", weights, outdir)

    model = PhaseNet.load_for_inference(weights, device="cuda")

    # ★train/val/test を区別せず**全動画**の特徴を出す（TCN の split はラベル側で切る）
    df = pd.read_parquet(D.LABEL_ROOT / D.PARQUET[ds])
    df["key"] = df["videoID"].map(lambda v: Path(str(v)).stem)

    for i, (vid, g) in enumerate(df.groupby("key", sort=True), 1):
        out = outdir / f"{vid}.npz"
        if out.exists():
            continue
        g = g.sort_values("sec").reset_index(drop=True)
        ld = DataLoader(D.PhaseFrameDataset(g, size, False), batch_size=args.batch,
                        shuffle=False, num_workers=args.workers, pin_memory=True)
        feats, logits = [], []
        t0 = time.time()
        with torch.no_grad():
            for x, _, _, _ in ld:
                with torch.autocast("cuda", dtype=torch.float16):
                    o = model(x.cuda(non_blocking=True))
                feats.append(o["feat"].half().cpu().numpy())
                logits.append(o["phase"].half().cpu().numpy())
        np.savez_compressed(
            out, feat=np.concatenate(feats), logit=np.concatenate(logits),
            label=g["phase"].to_numpy(np.int16), sec=g["sec"].to_numpy(np.int32))
        log.info("[%d] %s: T=%d (%ds)", i, vid, len(g), int(time.time() - t0))
    log.info("done -> %s", outdir)


if __name__ == "__main__":
    main()
