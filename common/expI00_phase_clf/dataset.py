r"""学習用 Dataset / split の組み立て.

## 3 つの供給源

| | 中身 | 使い道 |
|---|---|---|
| `labels/<own>.parquet` | 1 fps フレーム + phase(+tool) | 主タスク |
| `labels/<other>.parquet` | 相手データセットの 1 fps フレーム | **`other` クラスの負例**（router 兼フォールバック判定） |
| `labels/heico_instseg.parquet` | `raw.png` + `instrument_instances.png` | 補助タスク（dense seg） |

## split（★CV 汚染を作らないこと）

- **heico の 30 本は PROCEDURE CV の val 動画そのもの**。`workspace/fold/splits.py` 経由で
  fold v003 の `cv_fold0` を取り、**val 側 6 本を学習から完全に外す**。
  これを守らないと下流（expH01）の A/B が信用できなくなる。
- cholec80 は公式 split（video01–40 = 開発 / video41–80 = 評価）。開発側の末尾 8 本を val。
  lapchole とは別データセットなので FOCUS 側の fold とは無関係。

## `other` 負例の量

主タスクを薄めないよう **自データセットの `other_ratio`（既定 0.1）まで**に抑える。
動画単位でサンプリングし、フレームは動画内で等間隔に間引く（同じ場面の連番を避ける）。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PHASE_ROOT = Path("/mnt/data/data4/shared/miccai/Orena/phase")
FRAME_ROOT = PHASE_ROOT / "frames"
LABEL_ROOT = PHASE_ROOT / "labels"

log = logging.getLogger(__name__)

# cholec80 = 7 phase + other(=7) / heico = 14 phase + other(=14)
N_PHASE = {"cholec": 7, "heico": 14}
PARQUET = {"cholec": "cholec80.parquet", "heico": "heico.parquet"}
OTHER_OF = {"cholec": "heico", "heico": "cholec"}
CHOLEC_TOOLS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def heico_split(fold: int = 0, fold_version: str = "v003") -> tuple[set[str], set[str]]:
    """FOCUS の fold 定義から heico の train/val 動画 stem を返す（拡張子は落とす）。"""
    sys.path.insert(0, str(REPO / "workspace" / "fold"))
    import splits as fold_splits  # noqa: E402

    sp = fold_splits.cv_folds(fold_version)[fold]
    # ★folds.csv は heico を `.avi`、lapchole を `.mp4` で持つ。stem で照合する
    tr = {Path(v).stem for v in sp.train_videos if "Heico" in v}
    va = {Path(v).stem for v in sp.val_videos if "Heico" in v}
    if not va:
        raise RuntimeError(f"fold{fold} に heico の val 動画が無い")
    return tr, va


def cholec_split(n_val: int = 8) -> tuple[set[str], set[str], set[str]]:
    """公式 split。(train, val, test) を返す。test = video41–80 は一切学習に使わない。"""
    dev = [f"video{i:02d}" for i in range(1, 41)]
    test = {f"video{i:02d}" for i in range(41, 81)}
    return set(dev[:-n_val]), set(dev[-n_val:]), test


# --------------------------------------------------------------------------- #
# 表の組み立て
# --------------------------------------------------------------------------- #
def _key(df: pd.DataFrame) -> pd.Series:
    """videoID の表記ゆれ（.avi / .mp4 / 無し）を吸収した照合キー。"""
    return df["videoID"].map(lambda v: Path(str(v)).stem)


def load_tables(dataset: str, fold: int, fold_version: str,
                other_ratio: float, seed: int = 42) -> dict[str, pd.DataFrame]:
    """train / val / (test) の DataFrame を返す。`other` 負例は train にのみ足す。"""
    own = pd.read_parquet(LABEL_ROOT / PARQUET[dataset])
    own["key"] = _key(own)
    n_phase = N_PHASE[dataset]
    other_id = n_phase                      # `other` は最後のクラス

    if dataset == "heico":
        tr_v, va_v = heico_split(fold, fold_version)
        te_v: set[str] = set()
    else:
        tr_v, va_v, te_v = cholec_split()

    tr = own[own["key"].isin(tr_v)].copy()
    va = own[own["key"].isin(va_v)].copy()
    te = own[own["key"].isin(te_v)].copy()
    for name, sel, want in (("train", tr, tr_v), ("val", va, va_v), ("test", te, te_v)):
        got = set(sel["key"].unique())
        if want - got:
            raise RuntimeError(f"{dataset} {name}: labels に無い動画 {sorted(want - got)[:3]} "
                               "（フレーム抽出が未完了の可能性）")

    # ---- `other` 負例（相手データセット）を train にだけ足す ----
    if other_ratio > 0:
        oth_name = OTHER_OF[dataset]
        oth_path = LABEL_ROOT / PARQUET[oth_name]
        if not oth_path.exists():
            log.warning("%s が無いので other 負例をスキップ", oth_path)
        else:
            oth = pd.read_parquet(oth_path)
            oth["key"] = _key(oth)
            # 相手側も「学習に使ってよい動画」だけに限る（相互に val を汚さない）
            if oth_name == "heico":
                allow, _ = heico_split(fold, fold_version)
            else:
                allow, _, _ = cholec_split()
            oth = oth[oth["key"].isin(allow)]
            n_want = int(len(tr) * other_ratio)
            oth = _subsample_per_video(oth, n_want, seed)
            oth = oth.assign(phase=other_id, has_tool=False)
            for t in CHOLEC_TOOLS:
                oth[f"tool_{t}"] = 0
            keep = ["dataset", "videoID", "sec", "phase", "has_tool", "rel_path"] + \
                   [f"tool_{t}" for t in CHOLEC_TOOLS]
            for c in keep:
                if c not in tr.columns:
                    tr[c] = 0 if c.startswith("tool_") else (False if c == "has_tool" else "")
            tr = pd.concat([tr[keep], oth[keep]], ignore_index=True)
            log.info("other 負例 %d 行（%s）を train に追加 → 計 %d", len(oth), oth_name, len(tr))

    return {"train": tr.reset_index(drop=True), "val": va.reset_index(drop=True),
            "test": te.reset_index(drop=True)}


def _subsample_per_video(df: pd.DataFrame, n_want: int, seed: int) -> pd.DataFrame:
    """動画ごとに等間隔で間引く（同じ場面の連番が固まらないように）。"""
    if n_want <= 0 or len(df) == 0:
        return df.iloc[:0]
    vids = sorted(df["key"].unique())
    per = max(1, n_want // len(vids))
    out = []
    for v in vids:
        g = df[df["key"] == v].sort_values("sec")
        step = max(1, len(g) // per)
        out.append(g.iloc[::step].head(per))
    return pd.concat(out, ignore_index=True).sample(
        n=min(n_want, sum(len(o) for o in out)), random_state=seed)


def load_instseg(fold: int, fold_version: str) -> dict[str, pd.DataFrame]:
    """器具 seg のアノテーションフレーム。heico の fold を厳守して分ける。"""
    path = LABEL_ROOT / "heico_instseg.parquet"
    df = pd.read_parquet(path)
    df["key"] = _key(df)
    tr_v, va_v = heico_split(fold, fold_version)
    return {"train": df[df["key"].isin(tr_v)].reset_index(drop=True),
            "val": df[df["key"].isin(va_v)].reset_index(drop=True)}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


class PhaseFrameDataset(Dataset):
    """1 fps フレーム → (image, phase, tool, tool_mask)."""

    def __init__(self, df: pd.DataFrame, size: int, train: bool):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.train = train
        self.tool_cols = [f"tool_{t}" for t in CHOLEC_TOOLS]
        self.has_tool_cols = all(c in self.df.columns for c in self.tool_cols)
        if train:
            from torchvision import transforms as T
            self.jitter = T.ColorJitter(0.2, 0.2, 0.2)
        else:
            self.jitter = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = Image.open(FRAME_ROOT / r["rel_path"]).convert("RGB")
        if self.train:
            if np.random.rand() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = self.jitter(img)
        x = _to_tensor(img, self.size)
        y = torch.tensor(int(r["phase"]), dtype=torch.long)
        if self.has_tool_cols:
            tool = torch.tensor([float(r[c]) for c in self.tool_cols])
            tmask = torch.tensor(1.0 if bool(r["has_tool"]) else 0.0)
        else:
            tool = torch.zeros(len(CHOLEC_TOOLS))
            tmask = torch.tensor(0.0)
        return x, y, tool, tmask


class InstSegDataset(Dataset):
    """`raw.png` + `instrument_instances.png` → (image, binary mask, phase).

    mask はインスタンス ID マップ（0=背景, 1..7=器具インスタンス）なので `>0` で二値化する。
    画像とマスクへ**同じ**幾何変換を当てる（色変換は画像のみ）。
    """

    def __init__(self, df: pd.DataFrame, size: int, train: bool, mask_div: int = 4):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.mask_size = size // mask_div
        self.train = train
        if train:
            from torchvision import transforms as T
            self.jitter = T.ColorJitter(0.2, 0.2, 0.2)
        else:
            self.jitter = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = Image.open(r["raw_path"]).convert("RGB")
        msk = Image.open(r["mask_path"])
        if self.train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
        if self.train:
            img = self.jitter(img)
        x = _to_tensor(img, self.size)
        m = msk.resize((self.mask_size, self.mask_size), Image.NEAREST)
        m = (np.asarray(m, dtype=np.uint8) > 0).astype(np.float32)
        return x, torch.from_numpy(m), torch.tensor(int(r["phase"]), dtype=torch.long)


def cycle(loader):
    """seg ローダを無限に回す（主ローダのステップ数に合わせる）。"""
    while True:
        for batch in loader:
            yield batch
