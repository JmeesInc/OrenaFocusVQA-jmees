"""HuggingFace Mask2Former で FO instance segmentation を学習する。

    python train.py --config config.yaml --fold 0
    python train.py --config config.yaml --fold 0 --resume   # last.pt から再開

出力はすべて `results/{experiment.name}/fold{N}/` に集約する:
    config.yaml / train.log / training_log.json / last.pt / best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import FocusInstanceSegDataset, collate_fn
from postprocess import decode_instances, to_coco_detections

logger = logging.getLogger("train")


# ----- セットアップ ---------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unique_dir(base: Path) -> Path:
    """同名の実験ディレクトリがあれば _001, _002 と採番して上書きを防ぐ。"""
    if not base.exists():
        return base
    i = 1
    while (cand := base.parent / f"{base.name}_{i:03d}").exists():
        i += 1
    return cand


def setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(out_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    # httpx / urllib3 / PIL の DEBUG がログを埋めるので黙らせる
    for noisy in ("httpx", "httpcore", "urllib3", "PIL", "filelock", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_folds(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as f:
        return {r["video"]: int(r["fold"]) for r in csv.DictReader(f)}


# ----- 評価 ---------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, dataset, device, cfg, amp_dtype, aux=None) -> dict:
    """val セットで COCO segm AP と「陰性フレームでの誤検出」を測る。

    FO タスクでは「無いものを出さない」ことが AP と同じくらい効くので、
    negative フレームの誤検出率を独立に出す。

    ★しきい値を 2 つに分ける:
      * `eval.ap_conf`（低い値, 既定 0.01）… COCO AP 用。AP は PR 曲線を積分するので
        高いしきい値で足切りすると曲線が切れて **AP が大幅に過小評価される**
        （実際 0.5 で測っていた時 M2F 0.075 / YOLO 0.180 と比較不能な数字が出た）。
      * `eval.score_threshold`（運用値）… 陰性誤検出・検出率の集計用。
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    label2catid = {i: c["id"] for i, c in enumerate(dataset.categories)}
    ap_conf = float(cfg["eval"].get("ap_conf", 0.01))
    op_conf = float(cfg["eval"]["score_threshold"])
    detections: list[dict] = []
    neg_total = neg_clean = 0
    neg_fp = 0
    pos_total = pos_hit = 0
    pres_scores: list[np.ndarray] = []
    pres_truth: list[np.ndarray] = []
    prog_pred: list[np.ndarray] = []
    prog_true: list[np.ndarray] = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(pixel_values=pixel_values)
            if aux is not None:
                aux.eval()
                pl, pg = aux(outputs.encoder_last_hidden_state,
                             batch["t_abs"].to(device), batch["progress"].to(device))
                pres_scores.append(torch.sigmoid(pl.float()).cpu().numpy())
                pres_truth.append(batch["presence"].numpy())
                prog_pred.append(torch.sigmoid(pg.float()).cpu().numpy())
                prog_true.append(batch["progress"].numpy())
        # AP 用に低いしきい値で拾い、運用指標だけ op_conf で数え直す
        results = decode_instances(outputs, batch["orig_size"], threshold=ap_conf)
        for res, image_id, is_neg in zip(results, batch["image_id"], batch["is_negative"]):
            detections.extend(to_coco_detections(res, image_id, label2catid))
            n = int((res["scores"] >= op_conf).sum())
            if is_neg:
                neg_total += 1
                neg_fp += n
                neg_clean += int(n == 0)
            else:
                pos_total += 1
                pos_hit += int(n > 0)

    metrics = {
        "ap_conf": ap_conf, "op_conf": op_conf,
        "n_detections": len(detections),
        "neg_images": neg_total,
        "neg_clean_rate": neg_clean / neg_total if neg_total else float("nan"),
        "neg_fp_per_image": neg_fp / neg_total if neg_total else float("nan"),
        "pos_images": pos_total,
        "pos_detect_rate": pos_hit / pos_total if pos_total else float("nan"),
    }

    # COCOeval は positive 画像（GT のある画像）だけで計算する
    gt = {
        "images": [
            {"id": im["id"], "width": im["width"], "height": im["height"]}
            for im in dataset.images
        ],
        "annotations": [
            {**a, "iscrowd": a.get("iscrowd", 0)}
            for im in dataset.images
            for a in dataset.anns[im["id"]]
        ],
        "categories": dataset.categories,
    }
    if gt["annotations"] and detections:
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, iouType="segm")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        metrics["segm_AP"] = float(ev.stats[0])
        metrics["segm_AP50"] = float(ev.stats[1])
        metrics["segm_AP75"] = float(ev.stats[2])

        # ★クラス別の内訳（segm / bbox の AP・AP50）。
        #   どのクラスが伸びて/壊れているかは総合 AP からは読めないので必ず出す
        #   （eval_yolo.py と同じ粒度に揃えて YOLO と直接比較できるようにする）
        per_class = {}
        n_inst = defaultdict(int)
        for a in gt["annotations"]:
            n_inst[a["category_id"]] += 1
        for iou_type in ("segm", "bbox"):
            e2 = COCOeval(coco_gt, coco_dt, iouType=iou_type)
            for c in dataset.categories:
                e2.params.catIds = [c["id"]]
                e2.evaluate()
                e2.accumulate()
                e2.summarize()
                d = per_class.setdefault(c["name"], {"val_inst": n_inst[c["id"]]})
                d[f"{iou_type}_AP"] = float(e2.stats[0])
                d[f"{iou_type}_AP50"] = float(e2.stats[1])
        metrics["per_class"] = per_class
    else:
        metrics["segm_AP"] = metrics["segm_AP50"] = metrics["segm_AP75"] = 0.0

    if pres_scores:
        # presence は「画面上に写っているか」の画像単位2値。AUC と、運用点 0.5 の F1 を出す。
        S = np.concatenate(pres_scores); Y = np.concatenate(pres_truth)
        P = np.concatenate(prog_pred); T = np.concatenate(prog_true)
        pres = {}
        for i, name in dataset.id2label.items():
            y = Y[:, i]; sc = S[:, i]
            n_pos = int(y.sum())
            d = {"n_pos": n_pos}
            if 0 < n_pos < len(y):
                order = np.argsort(sc)
                rank = np.empty(len(sc), dtype=np.float64); rank[order] = np.arange(1, len(sc) + 1)
                d["auc"] = float((rank[y == 1].sum() - n_pos * (n_pos + 1) / 2)
                                 / (n_pos * (len(y) - n_pos)))
                tp = float(((sc >= 0.5) & (y == 1)).sum()); fp = float(((sc >= 0.5) & (y == 0)).sum())
                fn = float(((sc < 0.5) & (y == 1)).sum())
                d["f1"] = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
            pres[name] = d
        metrics["presence"] = pres
        metrics["presence_auc_macro"] = float(np.mean(
            [v["auc"] for v in pres.values() if "auc" in v]) if any("auc" in v for v in pres.values()) else 0.0)
        ok = T >= 0
        metrics["progress_mae"] = float(np.abs(P[ok] - T[ok]).mean()) if ok.any() else float("nan")
    return metrics


# ----- 学習 ---------------------------------------------------------------------


def build_optimizer(model, cfg):
    """backbone は低い lr（COCO 事前学習を壊さない）、その他は base lr。"""
    base_lr = float(cfg["optim"]["lr"])
    mult = float(cfg["optim"]["backbone_lr_mult"])
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if "pixel_level_module.encoder" in name else head).append(p)
    logger.info("optimizer: backbone %d params @lr=%.2e / head %d params @lr=%.2e",
                len(backbone), base_lr * mult, len(head), base_lr)
    return torch.optim.AdamW(
        [{"params": backbone, "lr": base_lr * mult}, {"params": head, "lr": base_lr}],
        lr=base_lr, weight_decay=float(cfg["optim"]["weight_decay"]),
    )


def lr_lambda_factory(total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fold", type=int, default=None, help="config の data.fold を上書き")
    ap.add_argument("--resume", action="store_true", help="results/.../last.pt から再開")
    ap.add_argument("--out-dir", default=None, help="再開先を明示指定（--resume と併用）")
    ap.add_argument("--max-steps", type=int, default=None, help="スモークテスト用")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="CUDA が無くても CPU で走らせる（既定は即エラー終了）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.fold is not None:
        cfg["data"]["fold"] = args.fold
    fold = int(cfg["data"]["fold"])
    seed = int(cfg["experiment"]["seed"])

    here = Path(__file__).resolve().parent
    base = here / cfg["output"]["results_root"] / cfg["experiment"]["name"] / f"fold{fold}"
    if args.resume:
        out_dir = Path(args.out_dir) if args.out_dir else base
        if not (out_dir / "last.pt").exists():
            print(f"resume 先が見つかりません: {out_dir / 'last.pt'}", file=sys.stderr)
            return 1
    else:
        out_dir = unique_dir(base)
    setup_logging(out_dir)
    logger.info("output dir: %s", out_dir)

    # 再現性のため config を実験ディレクトリへコピー
    shutil.copy(args.config, out_dir / "config.yaml")
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True

    # --- データ ---
    root = Path(cfg["data"]["root"])
    folds_csv = Path(cfg["data"]["folds_csv"])
    if not folds_csv.is_absolute():
        folds_csv = (here / folds_csv).resolve()
    assign = load_folds(folds_csv)
    train_videos = [v for v, f in assign.items() if f != fold]
    val_videos = [v for v, f in assign.items() if f == fold]
    logger.info("fold %d | train %d videos / val %d videos", fold, len(train_videos), len(val_videos))
    logger.info("val videos: %s", sorted(val_videos))

    coco_json = root / "coco" / "instances.json"
    classes = cfg["data"].get("classes") or None      # specialist（None = 全クラス）
    mt = cfg.get("multitask") or {}
    with_time = bool(mt.get("enabled"))
    train_ds = FocusInstanceSegDataset(
        coco_json, videos=train_videos, height=cfg["data"]["height"], width=cfg["data"]["width"],
        train=True, drop_negatives=cfg["data"]["drop_negatives"],
        max_negative_ratio=cfg["data"]["max_negative_ratio"], seed=seed,
        base_grid_repeat=cfg["data"].get("base_grid_repeat", 1),
        classes=classes, with_time=with_time,
        aug=str(cfg["data"].get("aug", "default")),
    )
    val_ds = FocusInstanceSegDataset(
        coco_json, videos=val_videos, height=cfg["data"]["height"], width=cfg["data"]["width"],
        train=False, classes=classes, with_time=with_time,
    )
    nw = int(cfg["data"]["num_workers"])
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["optim"]["batch_size"]), shuffle=True, num_workers=nw,
        collate_fn=collate_fn, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["optim"]["batch_size"]), shuffle=False, num_workers=nw,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=nw > 0,
    )

    # --- モデル ---
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    if not torch.cuda.is_available():
        # dl1 のドライバ (CUDA 12.6) は共有 .venv の torch cu130 と非互換で
        # torch.cuda.is_available() が False になる。黙って CPU に落ちると
        # 何時間も無駄になるのでここで止める（学習は dl2 の 4090 で回す）。
        logger.error(
            "CUDA が使えません。dl2 で回してください:\n"
            "  ssh dl2 'cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena/workspace/"
            "expF00_fo_instseg && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 "
            "../../.venv/bin/python train.py ...'\n"
            "（CPU で回したい場合のみ --allow-cpu）"
        )
        if not args.allow_cpu:
            return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = Mask2FormerConfig.from_pretrained(cfg["model"]["pretrained"])
    model_cfg.id2label = train_ds.id2label
    model_cfg.label2id = {v: k for k, v in train_ds.id2label.items()}
    model_cfg.num_labels = train_ds.num_labels
    model_cfg.num_queries = int(cfg["model"]["num_queries"])
    for k in ("no_object_weight", "class_weight", "mask_weight", "dice_weight"):
        setattr(model_cfg, k, float(cfg["model"][k]))
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg["model"]["pretrained"], config=model_cfg, ignore_mismatched_sizes=True,
    ).to(device)

    aux = None
    if with_time:
        from multitask import AuxHeads, aux_loss
        in_ch = model.config.backbone_config.hidden_size * 8 \
            if hasattr(model.config, "backbone_config") else 1536
        with torch.no_grad():
            probe = model(pixel_values=torch.zeros(1, 3, int(cfg["data"]["height"]),
                                                   int(cfg["data"]["width"])).to(device))
            in_ch = probe.encoder_last_hidden_state.shape[1]
        aux = AuxHeads(in_ch, train_ds.num_labels,
                       time_input=str(mt.get("time_input", "none"))).to(device)
        logger.info("multitask 有効: presence %d クラス / progress 回帰 / time_input=%s "
                    "（encoder feat %dch）", train_ds.num_labels,
                    mt.get("time_input", "none"), in_ch)

    amp_name = str(cfg["optim"]["amp_dtype"]).lower()
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}[amp_name]
    # torch.cuda.is_bf16_supported() は Turing でもエミュレーションで True を返す。
    # ネイティブ bf16 は Ampere (sm_80) 以降なので compute capability で判定する。
    if amp_dtype is torch.bfloat16 and torch.cuda.is_available() \
            and torch.cuda.get_device_capability()[0] < 8:
        logger.warning("%s は bf16 ネイティブ非対応 — fp16 にフォールバックします",
                       torch.cuda.get_device_name())
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    optimizer = build_optimizer(model, cfg)
    if aux is not None:
        optimizer.add_param_group({"params": list(aux.parameters()),
                                   "lr": float(cfg["optim"]["lr"])})
    accum = int(cfg["optim"]["accum_steps"])
    epochs = int(cfg["optim"]["epochs"])
    steps_per_epoch = max(1, len(train_loader) // accum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(steps_per_epoch * epochs, float(cfg["optim"]["warmup_ratio"]))
    )

    start_epoch, best_metric = 0, -float("inf")
    history: list[dict] = []
    if args.resume:
        ckpt = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        # ★monitor を変えて resume すると、**別指標の値**を best_metric として引き継いでしまう。
        #   2026-08-29: expF34 を AP50:95 → AP50 に切り替えて延長したところ、
        #   best_metric に 0.3046（AP50:95 の値）が入ったまま AP50 と比較され、
        #   再開後の最初の eval（AP50 0.4068）で **必ず** best が更新されて
        #   ep13 の重み（AP50 0.4330）が上書きされた。
        #   → monitor が変わったら history から**新しい指標での過去最良**を引き直す。
        prev_monitor = ckpt.get("monitor")
        best_metric = ckpt.get("best_metric", -float("inf"))
        # ★`monitor` を保存する前に書かれた古い last.pt では prev_monitor が None になる。
        #   その場合も**別指標の値を引き継いでいる可能性がある**ので引き直す
        #   （2026-08-29 expF39: segm_AP の 0.1384 を AP50 の best として復元してしまった）。
        if prev_monitor != cfg["eval"]["monitor"]:
            vals = [h[cfg["eval"]["monitor"]] for h in history
                    if isinstance(h, dict) and cfg["eval"]["monitor"] in h]
            best_metric = max(vals) if vals else -float("inf")
            logger.warning("monitor %s → %s: best_metric を history から %.4f に引き直した",
                           prev_monitor, cfg["eval"]["monitor"], best_metric)
        if ckpt.get("mid_epoch") is not None:
            logger.warning(
                "epoch 途中（ep%d step%d）の last.pt から再開する。"
                "その epoch を頭からやり直すので、LR schedule が最大 0.5 epoch 前倒しになる",
                ckpt["mid_epoch"], ckpt.get("mid_step", -1))
        logger.info("resumed from epoch %d (best %s=%.4f)",
                    start_epoch, cfg["eval"]["monitor"], best_metric)

    monitor = cfg["eval"]["monitor"]
    # eval.every_n_epochs が 1 未満なら「エポック内評価」に変換する（0.5 → 半エポックごと）
    steps_per_epoch = max(1, len(train_loader) // accum)
    every_ep = float(cfg["eval"]["every_n_epochs"])
    eval_every_steps = int(round(steps_per_epoch * every_ep)) if every_ep < 1 else 0
    if eval_every_steps:
        logger.info("エポック内評価: %.2f epoch = %d optimizer 更新ごと（1 epoch = %d 更新）",
                    every_ep, eval_every_steps, steps_per_epoch)
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        running, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            mask_labels = [m.to(device, non_blocking=True) for m in batch["mask_labels"]]
            class_labels = [c.to(device, non_blocking=True) for c in batch["class_labels"]]
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(pixel_values=pixel_values, mask_labels=mask_labels,
                            class_labels=class_labels)
                total = out.loss
                if aux is not None:
                    pres, prog = aux(out.encoder_last_hidden_state,
                                     batch["t_abs"].to(device), batch["progress"].to(device))
                    total = total + aux_loss(
                        pres, prog, batch["presence"].to(device), batch["progress"].to(device),
                        float(mt.get("presence_weight", 1.0)),
                        float(mt.get("progress_weight", 0.2)))
                loss = total / accum
            scaler.scale(loss).backward()
            running += float(total.item() if aux is not None else out.loss.item())
            n_batches += 1
            if (i + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(cfg["optim"]["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if global_step % 20 == 0:
                    logger.info("epoch %d step %d | loss %.4f | lr %.2e",
                                epoch, global_step, running / max(1, n_batches),
                                scheduler.get_last_lr()[-1])
                # ★エポック内評価。データを増やすと 1 epoch が長くなり（40k枚で ~5h）、
                #   エポック境界だけだと収束の様子が見えない。`eval.every_n_steps`
                #   （optimizer 更新回数）ごとに val を回して best を更新する。
                if eval_every_steps and global_step % eval_every_steps == 0:
                    metrics = evaluate(model, val_loader, val_ds, device, cfg, amp_dtype, aux)
                    logger.info(
                        "[mid] epoch %d step %d | segm_AP %.4f AP50 %.4f | "
                        "neg_clean %.3f | pos_detect %.3f",
                        epoch, global_step, metrics["segm_AP"], metrics["segm_AP50"],
                        metrics["neg_clean_rate"], metrics["pos_detect_rate"])
                    history.append({"epoch": epoch, "global_step": global_step,
                                    "train_loss": running / max(1, n_batches), **metrics})
                    (out_dir / "training_log.json").write_text(
                        json.dumps(history, indent=2), encoding="utf-8")
                    # ★エポック途中でも last.pt を保存する。
                    #   epoch が 8〜26 時間ある構成では、エポック境界だけの保存だと
                    #   途中で止めた瞬間に**全部消える**（2026-08-31 expF42: dl1 で 2.8h 回したが
                    #   ep0 を終えておらず last.pt が無く、貸し機へ引き継げなかった）。
                    #   `epoch - 1` で保存するので resume は**その epoch を頭からやり直す**。
                    #   重みと optimizer は保つので学習は失われない。
                    #   scheduler は途中位置のまま復元されるため、やり直した分だけ
                    #   LR schedule が最大 0.5 epoch ぶん先に進む（6 epoch なら 8% の前倒し）。
                    torch.save({
                        "epoch": epoch - 1, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "best_metric": best_metric, "monitor": monitor,
                        "history": history, "config": cfg,
                        "mid_epoch": epoch, "mid_step": global_step,
                    }, out_dir / "last.pt")
                    if metrics.get(monitor, -float("inf")) > best_metric:
                        best_metric = metrics[monitor]
                        model.save_pretrained(out_dir / "best_model")
                        if aux is not None:
                            torch.save(aux.state_dict(), out_dir / "best_model" / "aux_heads.pt")
                        (out_dir / "best_model" / "metrics.json").write_text(
                            json.dumps({**metrics, "epoch": epoch,
                                        "global_step": global_step}, indent=2),
                            encoding="utf-8")
                        logger.info("best 更新: %s=%.4f (step %d) → %s", monitor,
                                    best_metric, global_step, out_dir / "best_model")
                    model.train()
                if args.max_steps and global_step >= args.max_steps:
                    logger.info("--max-steps に到達 — 打ち切り")
                    break
        train_loss = running / max(1, n_batches)
        record = {"epoch": epoch, "train_loss": train_loss,
                  "lr": scheduler.get_last_lr()[-1], "sec": round(time.time() - t0, 1)}

        hit_max_steps = bool(args.max_steps and global_step >= args.max_steps)
        # スモークテストでも評価パスを必ず 1 度通す（ここが一番壊れやすい）
        if ((not eval_every_steps
             and (epoch + 1) % max(1, int(every_ep)) == 0)
                or epoch == epochs - 1 or hit_max_steps):
            metrics = evaluate(model, val_loader, val_ds, device, cfg, amp_dtype, aux)
            record.update(metrics)
            logger.info(
                "epoch %d | loss %.4f | segm_AP %.4f AP50 %.4f | "
                "neg_clean %.3f (%.2f fp/img, n=%d) | pos_detect %.3f",
                epoch, train_loss, metrics["segm_AP"], metrics["segm_AP50"],
                metrics["neg_clean_rate"], metrics["neg_fp_per_image"], metrics["neg_images"],
                metrics["pos_detect_rate"],
            )
            if "presence" in metrics:
                logger.info("  presence AUC macro %.3f / progress MAE %.3f | %s",
                            metrics["presence_auc_macro"], metrics["progress_mae"],
                            {k: (round(v.get("auc", float("nan")), 3), v["n_pos"])
                             for k, v in metrics["presence"].items()})
            if metrics.get(monitor, -float("inf")) > best_metric:
                best_metric = metrics[monitor]
                model.save_pretrained(out_dir / "best_model")
                if aux is not None:
                    torch.save(aux.state_dict(), out_dir / "best_model" / "aux_heads.pt")
                (out_dir / "best_model" / "metrics.json").write_text(
                    json.dumps({**metrics, "epoch": epoch}, indent=2), encoding="utf-8")
                logger.info("best 更新: %s=%.4f → %s", monitor, best_metric,
                            out_dir / "best_model")
        else:
            logger.info("epoch %d | loss %.4f | %.1fs", epoch, train_loss, record["sec"])

        history.append(record)
        (out_dir / "training_log.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_metric": best_metric, "monitor": monitor, "history": history, "config": cfg,
        }, out_dir / "last.pt")

        if args.max_steps and global_step >= args.max_steps:
            break

    logger.info("done. best %s = %.4f | %s", monitor, best_metric, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
