"""工程（phase）timeline をコンテナ内で作る。ConvNeXtV2 + MS-TCN。

★何に使うか: `combo` が「質問文の時刻と**同じ工程の区間**だけに絞る」のに使う。
  val 実測で、これを外すと該当 229 問が **0.6725 → 0.5677**（10勝34敗, p=0.0004）、
  SCORE は 0.5522 → 0.5441（−0.0081）。特に OBJECT_RECOGNITION が 0.83→0.70 と大きい。

★コスト: **索引用にデコード済みのフレームを使い回す**ので追加のデコードは無い。
  ConvNeXtV2-tiny@224 は軽く、M2F に比べれば無視できる。
  ⚠️元実装は 1fps 全フレームに CNN を掛けてから 5 秒ごとに間引いており **4/5 が無駄**。
    ここでは最初から TCN の刻み（5 秒）で回す。

★術式のルーティング: `procedure_type` で選ぶ。
  "Laparoscopic Cholecystectomy" → cholec モデル / それ以外 → heico モデル。
  ⚠️テストには未知の術式が入る。工程ラベル自体は当たらなくても、
    ここで要るのは「アンカーと同じ区間か」という**相対的な区切り**なので破綻はしない。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("FOCUS_RESOURCES", HERE / "resources"))
def _stride_s() -> float:
    """フレームの刻み（秒）。★モジュール読み込み時に固定すると env の変更が効かない。"""
    return float(os.environ.get("FOCUS_PHASE_STRIDE_S", "5"))
IMG = 224
BATCH = int(os.environ.get("FOCUS_PHASE_BATCH", "128"))
_CHOLEC_KEYS = ("cholecystectom",)          # procedure_type の判定（小文字で部分一致）


def route(procedure_type: str | None, video_id: str = "") -> str:
    """どちらの工程モデルを使うか。胆摘なら cholec、それ以外は heico。"""
    s = f"{procedure_type or ''} {video_id}".lower()
    return "cholec" if any(k in s for k in _CHOLEC_KEYS) else "heico"


class PhaseModel:
    """1 術式ぶんの (ConvNeXtV2, MS-TCN)。両方 `resources/phase_<name>/` に置く。"""

    def __init__(self, name: str, device: str = "cuda"):
        import sys
        sys.path.insert(0, str(RES / "phase_lib"))
        from model import PhaseNet                       # noqa: PLC0415
        from evaluate import load_tcn                    # noqa: PLC0415
        base = RES / f"phase_{name}"
        self.net = PhaseNet.load_for_inference(base / "best_model.pt", device=device)
        # ★MS-TCN は stride 5/10/20 でしか学習していない。
        #   フレームの刻みを変えるときは**最も近い重み**を選ぶ（比で近いものを取る）。
        avail = sorted(int(f.stem.replace("tcn_stride", ""))
                       for f in (base / "tcn").glob("tcn_stride*.pt"))
        # ★フレームの刻みに関わらず **stride5 を使う**。
        #   stride10/20 の重みは平滑化が弱く、工程が細切れになる
        #   （178 分の動画で 56 区間 → 154 区間。基準との区間判定一致も 0.95→0.91 に悪化）。
        #   expI00 の「索引には stage-2 の時系列平滑化が必須」と整合する。
        self.tcn_stride = 5 if 5 in avail else (min(avail) if avail else 5)
        self.tcn = load_tcn(base / "tcn", self.tcn_stride, device)
        self.device = device
        if self.tcn is None:
            raise FileNotFoundError(f"{base/'tcn'} に TCN が無い（候補 {avail}）")
        log.info("工程 %s: フレーム刻み %.0fs / TCN stride%d", name, _stride_s(), self.tcn_stride)

    @torch.no_grad()
    def __call__(self, frames: list[np.ndarray], secs: np.ndarray, total_s: int) -> np.ndarray:
        """`frames` は TCN 刻みで並んだ 224px テンソル。秒解像度の工程 ID を返す。"""
        feats = []
        for i in range(0, len(frames), BATCH):
            x = torch.from_numpy(np.stack(frames[i:i + BATCH])).to(self.device)
            with torch.autocast(self.device, dtype=torch.float16):
                feats.append(self.net(x)["feat"].float())
        f = torch.cat(feats)                              # (T', 768)  T' = 刻みごとのサンプル数
        p = self.tcn(f.T.unsqueeze(0))[-1].argmax(1).squeeze(0).cpu().numpy()
        # 刻みの予測を 1 秒解像度へ戻す（区間として使うので最近傍複製で十分）
        out = np.repeat(p, int(round(_stride_s())))   # フレームの実刻みで秒解像度へ戻す
        if len(out) < total_s:
            out = np.pad(out, (0, total_s - len(out)), mode="edge")
        return out[:total_s].astype(np.int8)


_CACHE: dict[str, PhaseModel] = {}


def get(name: str, device: str = "cuda") -> PhaseModel | None:
    """モデルは 1 回だけ読む（バッチ内で使い回す）。読めなければ None＝工程なしで動く。"""
    if name not in _CACHE:
        try:
            _CACHE[name] = PhaseModel(name, device)
            log.info("工程モデル読み込み: %s", name)
        except Exception as e:                            # noqa: BLE001
            log.warning("工程モデル %s を読めない（工程なしで続行）: %s", name, e)
            _CACHE[name] = None                           # type: ignore[assignment]
    return _CACHE[name]


def preprocess(img: np.ndarray) -> np.ndarray:
    """224px・ImageNet 正規化（学習時と同じ）。"""
    import cv2
    x = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    m = np.array([0.485, 0.456, 0.406], np.float32)
    s = np.array([0.229, 0.224, 0.225], np.float32)
    return ((x - m) / s).transpose(2, 0, 1)
