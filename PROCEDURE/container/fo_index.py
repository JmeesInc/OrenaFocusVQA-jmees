"""FO 検出索引をコンテナ内で構築する（Mask2Former / TensorRT 優先・PyTorch フォールバック）。

★設計の要点
- **索引は動画ごとに決まる**。`videoID` をキーに日和見キャッシュへ書き、次のバッチで
  ファイルシステムが残っていれば再利用する（残らなければ何も起きない＝損はしない）。
  PROCEDURE は 10,000問 / 200動画 ＝ 1動画あたり約50問なので、効けば索引コストが 1/50。
  ⚠️ただし**1バッチに同一動画は2問入らない**ので、バッチ内の再利用は原理的にゼロ。
- **TRT エンジンは GPU アーキ固有**。実行中の GPU の compute capability で選び、
  合わなければ PyTorch へ静かに落ちる（壊れずに遅くなるだけ）。
  本番は L40S(sm_89) 想定。4090 実測で **PyTorch 39fps → TRT 66fps（1.67x）**、
  索引としての一致は thr0.7 マスクで 100%、thr0.3 で 99.94%。
- **クラス列は索引が自己申告する**（expF40 は Gallstone を落として 7 クラス）。
  ここを固定にすると Specimen を Gallstone の列で読むことになり全ルールが誤作動する。
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("FOCUS_RESOURCES", HERE / "resources"))
CACHE = Path(os.environ.get("FOCUS_INDEX_CACHE", "/tmp/fo_index_cache"))
STRIDE_S = int(os.environ.get("FOCUS_INDEX_STRIDE_S", "15"))
IDX_H, IDX_W = 512, 896          # 学習時の前処理と合わせる
BATCH = int(os.environ.get("FOCUS_INDEX_BATCH", "16"))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _load_trt():
    """`tensorrt_libs` の .so を先に掴んでから import（LD_LIBRARY_PATH に依存させない）。"""
    import importlib
    for pat in ("libnvinfer.so*", "libnvonnxparser.so*"):
        for f in sorted(glob.glob(str(Path(sys.prefix) / "lib/python*/site-packages/tensorrt_libs" / pat))):
            try:
                ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    return importlib.import_module("tensorrt")


class Detector:
    """`(cls, area)` を返す。TRT が使えればそれを、駄目なら PyTorch。"""

    def __init__(self, device: str = "cuda"):
        self.device, self.trt, self.ctx, self.classes = device, None, None, None
        cc = "".join(str(v) for v in torch.cuda.get_device_capability()) if torch.cuda.is_available() else ""
        cands = [RES / f"m2f_sm{cc}.trt", RES / "m2f.trt"]
        if os.environ.get("FOCUS_NO_TRT") != "1":
            for eng in [c for c in cands if c.exists()]:
                try:
                    trt = _load_trt()
                    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
                    e = rt.deserialize_cuda_engine(eng.read_bytes())
                    if e is None:
                        raise RuntimeError("deserialize 失敗（GPU アーキ不一致）")
                    self.trt, self.eng = trt, e
                    self.ctx = e.create_execution_context()
                    self.names = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]
                    log.info("索引: TensorRT %s (sm%s)", eng.name, cc)
                    break
                except Exception as ex:                       # noqa: BLE001
                    log.warning("索引: TRT 不可 → PyTorch (%s)", ex)
                    self.ctx = None
        if self.ctx is None:
            from transformers import Mask2FormerForUniversalSegmentation
            m = Mask2FormerForUniversalSegmentation.from_pretrained(str(RES / "m2f")).to(device).eval().half()
            pe = type(m.model.pixel_level_module.decoder.position_embedding)
            if not getattr(pe, "_fp32_patched", False):       # TRT 版と数値を揃える
                _o = pe.forward

                def _f(self, shape, dev, dtype, mask=None):
                    return _o(self, shape, dev, torch.float32, mask).to(dtype)

                pe.forward, pe._fp32_patched = _f, True
            self.model = m
            log.info("索引: PyTorch（TRT なし）")
        cf = RES / "m2f_classes.json"
        self.classes = json.loads(cf.read_text()) if cf.exists() else None

    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        if self.ctx is None:
            o = self.model(pixel_values=x)
            cls = o.class_queries_logits.softmax(-1)[..., :-1]
            area = (o.masks_queries_logits.sigmoid() > 0.5).to(cls.dtype).flatten(2).mean(2)
            return cls, area
        trt = self.trt
        self.ctx.set_input_shape("pixel_values", tuple(x.shape))
        out = {}
        for n in self.names:
            if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.ctx.set_tensor_address(n, x.contiguous().data_ptr())
            else:
                t = torch.empty(tuple(self.ctx.get_tensor_shape(n)), device=self.device, dtype=torch.float16)
                out[n] = t
                self.ctx.set_tensor_address(n, t.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        return out["cls"], out["area"]


def _cache_key(video_id: str, lo: float, hi: float) -> str:
    """索引を一意に決める要素だけでキーを作る（動画・範囲・刻み・重み）。"""
    src = f"{video_id}|{lo:.0f}|{hi:.0f}|{STRIDE_S}|{os.environ.get('FOCUS_M2F_TAG', 'expF40')}"
    return hashlib.sha1(src.encode()).hexdigest()[:20]


def load_cached(video_id: str, lo: float, hi: float):
    p = CACHE / f"{_cache_key(video_id, lo, hi)}.npz"
    if not p.exists():
        return None
    try:
        z = dict(np.load(p))
        log.info("索引キャッシュ命中: %s", video_id)
        return z
    except Exception:                                          # noqa: BLE001
        return None


def save_cached(video_id: str, lo: float, hi: float, z: dict) -> None:
    """日和見キャッシュ。書けなくても無視する（本番で残る保証は無い）。"""
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        p = CACHE / f"{_cache_key(video_id, lo, hi)}.npz"
        tmp = p.with_suffix(".tmp")
        np.savez_compressed(tmp, **z)
        tmp.rename(p)                                          # 中断しても壊れた物を残さない
    except Exception as e:                                     # noqa: BLE001
        log.debug("索引キャッシュ書き込み失敗（無視）: %s", e)


@torch.no_grad()
def build_index(det: Detector, frames: list[np.ndarray], secs: np.ndarray) -> dict:
    """**前処理済み** (C,H,W) の配列列 → (t, score, count, area)。後処理は全部 GPU 上で畳む。

    ⚠️`frames` は `preprocess()` を通した後のもの。生の HWC を渡すと
      Swin の projection が 3ch を期待して落ちる。

    ★素朴な二重ループはクエリ 1 個ごとに GPU→CPU 同期が入り、後処理だけで 134.7ms
      かかっていた（B=16）。scatter_reduce に畳んで **2.1ms**（出力は完全一致）。
    """
    n_cls = len(det.classes) if det.classes else 7
    S = np.zeros((len(frames), n_cls), np.float32)
    C = np.zeros((len(frames), n_cls), np.int16)
    A = np.zeros((len(frames), n_cls), np.float32)
    i = 0
    for k in range(0, len(frames), BATCH):
        chunk = frames[k:k + BATCH]
        x = torch.from_numpy(np.stack(chunk)).to(det.device).half()
        cls, area = det(x)
        b = cls.shape[0]
        sc, lab = cls.max(-1)
        ok = (sc > 0.05) & (area > 0)
        z = torch.zeros(b, n_cls, device=cls.device, dtype=sc.dtype)
        S[i:i + b] = z.clone().scatter_reduce_(1, lab, torch.where(ok, sc, torch.zeros_like(sc)), "amax").float().cpu().numpy()
        A[i:i + b] = z.clone().scatter_reduce_(1, lab, torch.where(ok, area, torch.zeros_like(area)), "amax").float().cpu().numpy()
        C[i:i + b] = torch.zeros(b, n_cls, device=cls.device, dtype=torch.int32).scatter_add_(1, lab, ok.int()).cpu().numpy()
        i += b
    return {"t": secs.astype(np.int32), "score": S.astype(np.float16),
            "count": C.astype(np.int8), "area": A.astype(np.float16),
            "classes": np.array(det.classes if det.classes else [])}


def is_blank(img: np.ndarray, std_thr: float = 6.0) -> bool:
    """匿名化された充填フレーム（単色）か。**チャンネルごとの面内 std** で判定する。

    ★HeiCo は青 RGB(0,0,254)、lapchole は黒が 32%。**平均色では判別できない**ので
      「面内がのっぺり」を見る。過去の実測では匿名区間の FO 検出率は 0.1〜2.3% に対し
      非匿名は 33.7〜55.7% ＝ **匿名での発火は事実上すべて偽陽性**。
      検出器の前で捨てると計算 −18.7% / 到達率 −0.1pt（expI04）。
    """
    a = img[::8, ::8].astype(np.float32)          # 間引いて十分（判定は粗くてよい）
    return bool(a.reshape(-1, a.shape[-1]).std(axis=0).max() < std_thr)


def preprocess(img: np.ndarray) -> np.ndarray:
    """索引用の前処理（学習時と同じ 512x896 / ImageNet 正規化）。"""
    import cv2
    x = cv2.resize(img, (IDX_W, IDX_H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)
