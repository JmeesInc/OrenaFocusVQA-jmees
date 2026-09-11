# Source: workspace/expF00_fo_instseg/postprocess.py（逐語コピー。書き換えない）
"""Mask2Former の出力 → インスタンス（mask, label, score）。

`Mask2FormerImageProcessor.post_process_instance_segmentation` は内部で
マスクを一度 384x896... ではなく **決め打ちの 384x384** に補間してから
target_sizes へ戻す実装になっている（transformers 5.14 時点）。
入力を 512x896 で学習しているとここで解像度と縦横比を落とすので、
評価・可視化では同じ計算を target_sizes へ直接補間する形で書き直して使う。
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

# ★decode のクエリ・チャンク幅（メモリ削減用。結果は変わらない）
_DECODE_CHUNK = int(os.environ.get("FOCUS_DECODE_CHUNK", "32"))


@torch.no_grad()
def decode_instances(
    outputs,
    target_sizes: list[tuple[int, int]],
    threshold: float = 0.5,
    mask_threshold: float = 0.0,
) -> list[dict]:
    """バッチの出力を画像ごとの {masks, labels, scores} に変換する。

    masks は uint8 (N, H, W)（H, W = target_sizes[i]）。N=0 になり得る（陰性予測）。
    score は Mask2Former 公式と同じ「クラス確率 × マスク内平均確率」。
    """
    class_logits = outputs.class_queries_logits  # (B, Q, C+1)
    mask_logits = outputs.masks_queries_logits  # (B, Q, h, w)
    num_classes = class_logits.shape[-1] - 1
    num_queries = class_logits.shape[-2]

    results: list[dict] = []
    for i in range(class_logits.shape[0]):
        scores = class_logits[i].softmax(dim=-1)[:, :-1]  # (Q, C)
        labels = (
            torch.arange(num_classes, device=scores.device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )
        scores_flat, topk = scores.flatten(0, 1).topk(num_queries, sorted=False)
        labels_per_query = labels[topk]
        query_idx = torch.div(topk, num_classes, rounding_mode="floor")

        # 元解像度へ直接補間（384x384 を経由しない）
        # ★**クエリ方向にチャンクする**（2026-09-03）。200 クエリ ×
        #   表示解像度 の float テンソルを一度に作ると、1024x576 で 1 本 450MB、
        #   中間（m / sigmoid / binary）が同時に生きて **数 GB** になり、
        #   VLM と同居するコンテナで CUDA OOM した（検出器 2 本 = F40 + F39@768x1344）。
        #   ★計算式は一切変えていない（chunk ごとに同じ演算をして連結するだけ）ので
        #     **結果はビット単位で同じ**。
        keep_masks, keep_labels, keep_scores = [], [], []
        chunk = int(_DECODE_CHUNK)
        for a in range(0, len(query_idx), chunk):
            b = min(a + chunk, len(query_idx))
            m = F.interpolate(
                mask_logits[i][query_idx[a:b]].unsqueeze(0).float(),
                size=target_sizes[i],
                mode="bilinear",
                align_corners=False,
            )[0]
            binary = (m > mask_threshold).float()
            area = binary.flatten(1).sum(1)
            mask_score = (m.sigmoid().flatten(1) * binary.flatten(1)).sum(1) / (area + 1e-6)
            final = scores_flat[a:b] * mask_score
            keep = (final >= threshold) & (area > 0)
            if bool(keep.any()):
                keep_masks.append(binary[keep].to(torch.uint8).cpu())
                keep_labels.append(labels_per_query[a:b][keep].cpu())
                keep_scores.append(final[keep].cpu())
            del m, binary, area, mask_score, final, keep
        results.append({
            "masks": (torch.cat(keep_masks) if keep_masks
                      else torch.zeros((0, *target_sizes[i]), dtype=torch.uint8)),
            "labels": (torch.cat(keep_labels) if keep_labels
                       else torch.zeros((0,), dtype=torch.long)),
            "scores": (torch.cat(keep_scores) if keep_scores
                       else torch.zeros((0,), dtype=torch.float)),
        })
    return results


def to_coco_detections(
    result: dict, image_id: int, label2catid: dict[int, int]
) -> list[dict]:
    """COCOeval(segm) に食わせる detection のリストへ。"""
    from pycocotools import mask as mask_utils

    dets = []
    masks = result["masks"].numpy()
    for m, label, score in zip(masks, result["labels"].tolist(), result["scores"].tolist()):
        rle = mask_utils.encode(m.astype("uint8", order="F"))
        rle["counts"] = rle["counts"].decode("ascii")
        dets.append({
            "image_id": image_id,
            "category_id": label2catid[label],
            "segmentation": rle,
            "score": score,
        })
    return dets
