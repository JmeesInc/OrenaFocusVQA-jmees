r"""stage-1（フレーム単体）の工程分類器を学習する.

鉄則（CLAUDE.md）:
- AMP 常時 ON（Turing は bf16 が無いので fp16）
- `last.ckpt` からの再開必須 / シード固定 / ハイパラは全部 config
- ログは logging（コンソール INFO + ファイル DEBUG）
- 全出力を `results/<experiment_name>/fold<N>/` へ集約し、config.yaml を自動コピー
- 同名ディレクトリがあれば `_001`, `_002` と採番して上書きしない

損失:
    L = CE(phase) + lambda_tool * maskedBCE(tool) + lambda_seg * (BCE + Dice)

`lambda_* = 0` にすれば補助タスク無しの対照になる。**この A/B が本実験の検証項目**。

Usage:
    ./run.sh smoke heico          # 経路確認（config の形は本番と同一のまま件数だけ絞る）
    ./run.sh train heico          # 本番（デタッチ）
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

import dataset as D
from model import PhaseNet

HERE = Path(__file__).resolve().parent
log = logging.getLogger("phase_train")


# --------------------------------------------------------------------------- #
def get_logger(logdir: Path) -> logging.Logger:
    logdir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger()
    lg.setLevel(logging.DEBUG)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt); lg.addHandler(ch)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(logdir / f"train_{ts}.log"); fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def unique_dir(base: Path) -> Path:
    """同名の実験ディレクトリがあれば `_001`, `_002` と採番して上書きしない。"""
    if not base.exists():
        return base
    i = 1
    while (cand := base.parent / f"{base.name}_{i:03d}").exists():
        i += 1
    return cand


def dice_loss(logit: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logit)
    num = 2 * (p * target).sum((1, 2)) + eps
    den = p.sum((1, 2)) + target.sum((1, 2)) + eps
    return (1 - num / den).mean()


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, n_own: int) -> dict:
    """own phase（`other` を除く）での accuracy / macro-F1 と、`other` の検出率。"""
    model.eval()
    preds, gts = [], []
    for x, y, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x.to(device, non_blocking=True))
        preds.append(out["phase"].float().argmax(1).cpu().numpy())
        gts.append(y.numpy())
    p, g = np.concatenate(preds), np.concatenate(gts)
    own = g < n_own
    acc = float((p[own] == g[own]).mean()) if own.any() else 0.0
    macro_f1 = float(f1_score(g[own], p[own], average="macro",
                              labels=list(range(n_own)), zero_division=0)) if own.any() else 0.0
    # `other` へ誤って倒れた率（索引が効かなくなる方向の誤り）
    other_leak = float((p[own] == n_own).mean()) if own.any() else 0.0
    per_class = f1_score(g[own], p[own], average=None,
                         labels=list(range(n_own)), zero_division=0).tolist() if own.any() else []
    return {"acc": acc, "macro_f1": macro_f1, "other_leak": other_leak,
            "per_class_f1": [round(v, 4) for v in per_class]}


@torch.no_grad()
def evaluate_seg(model, loader, device) -> dict:
    model.eval()
    inter = union = 0.0
    for x, m, _ in loader:
        x, m = x.to(device), m.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(x, want_seg=True)["seg"].float().squeeze(1)
        pred = (torch.sigmoid(logit) > 0.5).float()
        inter += (pred * m).sum().item()
        union += ((pred + m) > 0).float().sum().item()
    return {"seg_iou": inter / max(union, 1.0)}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="config の形は変えず、件数と epoch だけ絞って経路を通す")
    ap.add_argument("--train-limit", type=int, default=None, help="学習件数の上限（smoke 用）")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg_path = HERE / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    ds = cfg["data"]["dataset"]
    fold = int(cfg["cv"]["fold"])
    n_own = D.N_PHASE[ds]
    seed = int(cfg["experiment"]["seed"])

    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA が見えない。dl1 では `.venv-dl1` を使うこと（共有 .venv は cu130）")

    name = cfg["experiment"]["name"] + ("_smoke" if args.smoke else "")
    base = HERE / "results" / name / f"fold{fold}"
    outdir = base if (args.resume and base.exists()) else unique_dir(base)
    outdir.mkdir(parents=True, exist_ok=True)
    get_logger(outdir)
    shutil.copy(cfg_path, outdir / "config.yaml")     # 再現性のため学習開始時にコピー
    log.info("outdir: %s", outdir)
    log.info("GPU: %s", torch.cuda.get_device_name(0))

    # ---- data ----------------------------------------------------------- #
    tabs = D.load_tables(ds, fold, cfg["cv"]["fold_version"],
                         float(cfg["data"]["other_ratio"]), seed)
    tr, va = tabs["train"], tabs["val"]
    if args.smoke:
        lim = args.train_limit or 512
        tr = tr.sample(n=min(lim, len(tr)), random_state=seed)
        va = va.sample(n=min(lim, len(va)), random_state=seed)
    elif args.train_limit:
        tr = tr.sample(n=min(args.train_limit, len(tr)), random_state=seed)
    log.info("%s fold%d: train %d / val %d frames (%d own phases + other)",
             ds, fold, len(tr), len(va), n_own)
    log.info("train phase 分布: %s", tr["phase"].value_counts().sort_index().to_dict())

    size = int(cfg["data"]["img_size"])
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])
    tr_ld = DataLoader(D.PhaseFrameDataset(tr, size, True), batch_size=bs, shuffle=True,
                       num_workers=nw, pin_memory=True, drop_last=True, persistent_workers=nw > 0)
    va_ld = DataLoader(D.PhaseFrameDataset(va, size, False), batch_size=bs, shuffle=False,
                       num_workers=nw, pin_memory=True)

    lam_tool = float(cfg["loss"]["lambda_tool"])
    lam_seg = float(cfg["loss"]["lambda_seg"])
    seg_tr_ld = seg_va_ld = seg_iter = None
    if lam_seg > 0:
        if ds != "heico":
            raise ValueError("器具 seg は heico にしか無い（cholec は tool presence を使う）")
        seg_tabs = D.load_instseg(fold, cfg["cv"]["fold_version"])
        seg_bs = max(1, bs // int(cfg["loss"]["seg_batch_div"]))
        if args.smoke:
            seg_tabs = {k: v.head(32) for k, v in seg_tabs.items()}
        log.info("instseg: train %d / val %d frames (batch %d/step)",
                 len(seg_tabs["train"]), len(seg_tabs["val"]), seg_bs)
        seg_tr_ld = DataLoader(D.InstSegDataset(seg_tabs["train"], size, True), batch_size=seg_bs,
                               shuffle=True, num_workers=max(2, nw // 2), pin_memory=True,
                               drop_last=True, persistent_workers=nw > 0)
        seg_va_ld = DataLoader(D.InstSegDataset(seg_tabs["val"], size, False), batch_size=seg_bs,
                               shuffle=False, num_workers=2, pin_memory=True)
        seg_iter = D.cycle(seg_tr_ld)

    # ---- model ---------------------------------------------------------- #
    mcfg = {
        "backbone": cfg["model"]["backbone"],
        "n_phase": n_own + 1,                     # 最後のクラスが `other`
        "n_tool": len(D.CHOLEC_TOOLS) if lam_tool > 0 else 0,
        "seg": lam_seg > 0,
        "drop_path": float(cfg["model"]["drop_path"]),
    }
    ckpt_path = cfg["model"].get("pretrained_ckpt") or None
    if ckpt_path:
        ckpt_path = str((HERE.parents[1] / ckpt_path).resolve())
    model = PhaseNet(**mcfg, pretrained=bool(cfg["model"]["pretrained"]),
                     pretrained_ckpt=ckpt_path).to(device)

    # class weight（既定 none。文献比較のため accuracy を歪めない）
    weight = None
    if cfg["loss"].get("class_weight", "none") == "inv_sqrt":
        cnt = tr["phase"].value_counts().reindex(range(n_own + 1), fill_value=0).values
        w = 1.0 / np.sqrt(np.maximum(cnt, 1))
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
        log.info("class_weight(inv_sqrt): %s", np.round(weight.cpu().numpy(), 2).tolist())

    epochs = int(cfg["train"]["epochs"]) if not args.smoke else 1
    lr = float(cfg["train"]["lr"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=float(cfg["train"]["weight_decay"]))
    steps_per_ep = max(1, len(tr_ld))
    warm = float(cfg["train"]["warmup_epochs"]) * steps_per_ep
    total = epochs * steps_per_ep

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / max(1.0, warm)
        prog = (step - warm) / max(1.0, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    clip = float(cfg["train"]["grad_clip"])

    ckpt = outdir / "last.ckpt"
    start_ep, best = 0, -1.0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        start_ep, best = st["epoch"] + 1, st["best"]
        log.info("resumed from epoch %d (best macro-F1 %.4f)", start_ep, best)

    log.info("loss: CE + %.2f*tool + %.2f*seg | epochs=%d bs=%d lr=%g size=%d",
             lam_tool, lam_seg, epochs, bs, lr, size)

    hist = []
    for ep in range(start_ep, epochs):
        model.train(); t0 = time.time()
        agg = {"loss": 0.0, "ce": 0.0, "tool": 0.0, "seg": 0.0}
        for x, y, tool, tmask in tr_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
                ce = F.cross_entropy(out["phase"], y, weight=weight)
                loss = ce
                l_tool = torch.zeros((), device=device)
                if lam_tool > 0:
                    tool, tmask = tool.to(device), tmask.to(device)
                    bce = F.binary_cross_entropy_with_logits(out["tool"], tool, reduction="none")
                    l_tool = (bce.mean(1) * tmask).sum() / tmask.sum().clamp(min=1.0)
                    loss = loss + lam_tool * l_tool
                l_seg = torch.zeros((), device=device)
                if lam_seg > 0:
                    sx, sm, _ = next(seg_iter)
                    sx, sm = sx.to(device, non_blocking=True), sm.to(device, non_blocking=True)
                    slogit = model(sx, want_seg=True)["seg"].squeeze(1)
                    l_seg = F.binary_cross_entropy_with_logits(slogit, sm) + dice_loss(slogit, sm)
                    loss = loss + lam_seg * l_seg
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt); scaler.update(); sched.step()
            agg["loss"] += loss.item(); agg["ce"] += ce.item()
            agg["tool"] += l_tool.detach().item(); agg["seg"] += l_seg.detach().item()

        n = len(tr_ld)
        m = evaluate(model, va_ld, device, n_own)
        if seg_va_ld is not None:
            m.update(evaluate_seg(model, seg_va_ld, device))
        log.info("ep%02d loss %.4f (ce %.4f tool %.4f seg %.4f) | val acc %.4f macro-F1 %.4f "
                 "other_leak %.4f%s | %ds", ep, agg["loss"] / n, agg["ce"] / n,
                 agg["tool"] / n, agg["seg"] / n, m["acc"], m["macro_f1"], m["other_leak"],
                 f" seg_iou {m['seg_iou']:.4f}" if "seg_iou" in m else "",
                 int(time.time() - t0))
        log.debug("per-class F1: %s", m["per_class_f1"])
        hist.append({"epoch": ep, **{k: v / n for k, v in agg.items()}, **m})
        json.dump(hist, (outdir / "training_log.json").open("w"), indent=2)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "best": best}, ckpt)
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save({"model": model.state_dict(), "model_cfg": mcfg, "dataset": ds,
                        "fold": fold, "img_size": size, "metrics": m, "epoch": ep},
                       outdir / "best_model.pt")
            json.dump({"epoch": ep, **m}, (outdir / "best_metrics.json").open("w"), indent=2)
            log.info("   ** new best macro-F1 %.4f", best)

    log.info("done. best macro-F1 %.4f. results in %s", best, outdir)


if __name__ == "__main__":
    main()
