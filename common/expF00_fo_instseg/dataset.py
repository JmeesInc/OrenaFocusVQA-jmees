"""build_dataset.py が吐いた COCO を HuggingFace Mask2Former 用に読む Dataset。

Mask2Former の forward は `mask_labels: list[Tensor(N, H, W)]` と
`class_labels: list[Tensor(N,)]` を取る。COCO の RLE から直接この 2 つを作る
（`Mask2FormerImageProcessor` のセマンティックマップ経由だとインスタンスの
重なりが潰れるため使わない）。

negative フレーム（annotations が 1 つも無い image）は N=0 の空テンソルを返す。
Mask2Former の loss は num_masks を min=1 で clamp するので空でも落ちない。
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_transforms(height: int, width: int, train: bool, preset: str = "default"):
    """Albumentations の変換を返す。`preset` で強度を切り替える。

    - `default` … 弱め（CLAUDE.md「Augmentation はまず弱めで」）。expF32〜F37 が使用
    - `strong`  … **blur を強く / affine を大きく**（2026-08-29 ユーザ指示、clip 対策）

    ★`strong` の設計意図（clip は val 275 inst で AP 0.07〜0.11 と最弱）:
      * **blur**: 内視鏡映像の実劣化は **motion blur（スコープの振れ）** と
        **defocus（近接時のピンボケ）**。clip は数ピクセルなので、ここに耐性が無いと
        「ボケた瞬間だけ落ちる」。`blur_limit` を 21 まで振る（既定 7 の 3 倍）
      * **affine**: スコープは自由に回るので **rotate ±30 / shear ±10 / scale 0.6-1.6**
        まで許す。小物体の見かけサイズを大きく振ることがスケール不変性に効く
      * **cutmix/mixup は使わない**（ユーザ指示。インスタンス境界が壊れるため instance seg と相性が悪い）
      * blur は image-only なのでマスクは変形しない。affine は mask も同じ変換を受ける
    """
    import albumentations as A

    if not train:
        return A.Compose([A.Resize(height=height, width=width, interpolation=1)])
    if preset == "strong":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.6, 1.6), translate_percent=(-0.15, 0.15), rotate=(-30, 30),
                     shear=(-10, 10), border_mode=0, p=0.9),
            # ★blur は「かなり強く」。4 種のどれかを 70% で当てる
            A.OneOf([
                A.MotionBlur(blur_limit=(3, 21)),
                A.Defocus(radius=(3, 10)),
                A.GaussianBlur(blur_limit=(3, 21)),
                A.ZoomBlur(max_factor=(1.02, 1.15)),
            ], p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.4),
            # 実際の配信映像は圧縮劣化を含む
            A.ImageCompression(quality_range=(40, 90), p=0.3),
            A.GaussNoise(std_range=(0.02, 0.10), p=0.2),
            A.Resize(height=height, width=width, interpolation=1),
        ])
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.2), translate_percent=(-0.05, 0.05), rotate=(-10, 10),
                 border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.Resize(height=height, width=width, interpolation=1),
    ])


class FocusInstanceSegDataset(Dataset):
    """COCO instances(RLE) → Mask2Former の (pixel_values, mask_labels, class_labels)。

    Args:
        coco_json: build_dataset.py が出した instances.json
        videos: この Dataset に含める動画 basename（None なら全部）。fold 分割用。
        height/width: リサイズ後のサイズ（size_divisor=32 の倍数にすること）
        train: augmentation の有無
        drop_negatives: True なら negative フレームを除外する（陰性教師を切る比較用）
        max_negative_ratio: negative / positive の上限比。None なら制限なし
    """

    def __init__(
        self,
        coco_json: str | Path,
        videos: list[str] | None = None,
        height: int = 512,
        width: int = 896,
        train: bool = False,
        drop_negatives: bool = False,
        max_negative_ratio: float | None = None,
        seed: int = 42,
        base_grid_repeat: int = 1,
        classes: list[str] | None = None,
        with_time: bool = False,
        aug: str = "default",
    ):
        coco_json = Path(coco_json)
        doc = json.loads(coco_json.read_text(encoding="utf-8"))
        self.root = coco_json.parent.parent  # <out>/coco/instances.json → <out>
        self.categories = sorted(doc["categories"], key=lambda c: c["id"])
        # ★クラス限定（specialist）。指定クラス以外の instance は**教師から消える**ので、
        #   画面に写っている他の FO が「背景」として教えられる点に注意
        #   （expF17-20 で YOLO の specialist が all-class に全クラスで負けた既知の機序）。
        if classes:
            want = set(classes)
            unknown = want - {c["name"] for c in self.categories}
            assert not unknown, f"未知のクラス: {unknown}"
            self.categories = [c for c in self.categories if c["name"] in want]
        # COCO の category_id (1始まり, 欠番あり得る) → 0始まりの連番クラス
        self.catid2label = {c["id"]: i for i, c in enumerate(self.categories)}
        self.id2label = {i: c["name"] for i, c in enumerate(self.categories)}
        self.num_labels = len(self.categories)

        keep_catids = {c["id"] for c in self.categories}
        anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for a in doc["annotations"]:
            if a["category_id"] in keep_catids:
                anns_by_image[a["image_id"]].append(a)

        images = doc["images"]
        if videos is not None:
            keep = set(videos)
            images = [im for im in images if im["video"] in keep or Path(im["video"]).stem in keep]

        pos = [im for im in images if anns_by_image.get(im["id"])]
        neg = [im for im in images if not anns_by_image.get(im["id"])]
        if drop_negatives:
            neg = []
        elif max_negative_ratio is not None and pos:
            cap = int(round(len(pos) * max_negative_ratio))
            if len(neg) > cap:
                rng = np.random.default_rng(seed)
                neg = [neg[i] for i in sorted(rng.choice(len(neg), cap, replace=False))]

        self.images = sorted(pos + neg, key=lambda im: im["id"])
        # ★SAM3 pseudolabel（sampling='pseudo'）や密サンプリング（'dense'）を混ぜると
        # 枚数で 30 秒グリッドの GT を圧倒してしまう。export_yolo.py の
        # --base-grid-repeat と同じ意味で、30 秒グリッド側を N 回出して重みを付ける
        # （画像ごとの loss 係数を持たないので複製で代替。train のみ）。
        n_rep = 0
        if train and base_grid_repeat > 1:
            # ★複製するのは「30秒グリッドの陽性/陰性」だけ。densify で足した
            #   'dense' / 'pseudo' / 'neg_dense' は複製しない（export_yolo.py と同じ意味）。
            #   neg_dense を複製すると陰性比が 2.00 のつもりで 6.5 になり、
            #   1 epoch の長さも 2.4 倍になる（2026-08-19 に実害）
            extra = [im for im in self.images
                     if im.get("sampling") not in ("dense", "pseudo", "neg_dense")
                     for _ in range(base_grid_repeat - 1)]
            n_rep = len(extra)
            self.images = self.images + extra
        self.anns = {im["id"]: anns_by_image.get(im["id"], []) for im in self.images}
        self.transforms = build_transforms(height, width, train, aug)
        # ★時刻メタ（presence/progress マルチタスク用）。videos.csv の fps / n_frames から
        #   絶対秒 t と正規化位置 t/duration を作る。
        #   ⚠️ **正規化位置は推論時には取れない**（FRAME の request は start_time=end_time の
        #   絶対秒だけで、動画の総尺が渡らない）。学習の補助タスクとしてのみ使う。
        self.with_time = with_time
        self.time_meta: dict[str, tuple[float, float]] = {}
        if with_time:
            import csv as _csv
            with (self.root / "videos.csv").open(encoding="utf-8") as f:
                for r in _csv.DictReader(f):
                    fps = float(r["fps"]) or 30.0
                    self.time_meta[r["video"]] = (fps, float(r["n_frames"]) / fps)
        n_pseudo = sum(1 for im in self.images if im.get("sampling") == "pseudo")
        logger.info(
            "%s: %d images (pos %d / neg %d / pseudo %d / 複製 +%d), %d categories",
            coco_json, len(self.images), len(pos), len(neg), n_pseudo, n_rep, self.num_labels,
        )

    def __len__(self) -> int:
        return len(self.images)

    def _decode_masks(self, image_info: dict) -> tuple[np.ndarray, list[int]]:
        from pycocotools import mask as mask_utils

        anns = self.anns[image_info["id"]]
        if not anns:
            return np.zeros((0, image_info["height"], image_info["width"]), np.uint8), []
        rles = [a["segmentation"] for a in anns]
        # RLE の counts は JSON 往復で str になっているので bytes に戻す
        rles = [{"size": r["size"], "counts": r["counts"].encode("ascii")}
                if isinstance(r["counts"], str) else r for r in rles]
        masks = mask_utils.decode(rles)  # (H, W, N)
        return masks.transpose(2, 0, 1), [a["category_id"] for a in anns]

    def __getitem__(self, idx: int) -> dict:
        import cv2

        info = self.images[idx]
        path = self.root / info["file_name"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks, cat_ids = self._decode_masks(info)
        out = self.transforms(image=image, masks=list(masks))
        image = out["image"]
        masks = out["masks"]

        # augmentation で画像外に出たインスタンスは落とす
        kept_masks, kept_labels = [], []
        for m, cid in zip(masks, cat_ids):
            if m.any():
                kept_masks.append(m)
                kept_labels.append(self.catid2label[cid])

        pixel_values = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        pixel_values = torch.from_numpy(pixel_values.transpose(2, 0, 1))
        h, w = image.shape[:2]
        if kept_masks:
            mask_labels = torch.from_numpy(np.stack(kept_masks).astype(np.float32))
            class_labels = torch.tensor(kept_labels, dtype=torch.long)
        else:
            mask_labels = torch.zeros((0, h, w), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.long)

        item = {
            "pixel_values": pixel_values,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
            "image_id": info["id"],
            "orig_size": (info["height"], info["width"]),
            # ★specialist（classes 指定）では「明示的 negative」だけを陰性にすると
            #   pos_detect の**分母が全 FO 陽性 1,088 枚**になり、clip を含まない 978 枚まで
            #   分母に入って値が壊れる（2026-08-30: expF39 の 0.086 は実際には最大 0.855 だった。
            #   clip を含む val 画像は **110 枚**しかない）。
            #   → 「明示的 negative」**または**「対象クラスの注釈が 1 つも無い」を陰性とする。
            #   全クラス学習では従来と同一（注釈なし画像はもともと neg 側）。
            "is_negative": bool(info.get("is_negative", False) or not self.anns[info["id"]]),
        }
        if self.with_time:
            fps, dur = self.time_meta.get(info["video"], (30.0, 0.0))
            t_abs = info["frame_number"] / fps
            item["t_abs"] = float(t_abs)
            item["progress"] = float(min(1.0, t_abs / dur)) if dur > 0 else -1.0
            # 画像レベルの presence（そのフレームに各クラスが写っているか）
            pres = np.zeros(self.num_labels, dtype=np.float32)
            for cid in set(cat_ids):
                pres[self.catid2label[cid]] = 1.0
            item["presence"] = torch.from_numpy(pres)
        return item


def collate_fn(batch: list[dict]) -> dict:
    """pixel_values だけ stack。mask/class labels は可変長なので list のまま。"""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask_labels": [b["mask_labels"] for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        **({"t_abs": torch.tensor([b["t_abs"] for b in batch], dtype=torch.float32),
            "progress": torch.tensor([b["progress"] for b in batch], dtype=torch.float32),
            "presence": torch.stack([b["presence"] for b in batch])}
           if "presence" in batch[0] else {}),
        "orig_size": [b["orig_size"] for b in batch],
        "is_negative": [b["is_negative"] for b in batch],
    }
