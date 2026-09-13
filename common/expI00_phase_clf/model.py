r"""工程分類モデル本体（backbone + phase / tool / instrument-seg ヘッド）.

## 設計の要点

- **backbone は timm の標準モデルをそのまま使う**（`features_only=True` は使わない）。
  `features_only` はキー名が `stem_0.*` などへ平坦化され、SurgeNet 重みの
  `checkpoint_filter_fn` 変換（expC00 で 198/198 読込を確認済み）が効かなくなる。
  代わりに timm 1.0 の `forward_intermediates()` で中間特徴を取り出す。
- **seg デコーダは推論時に load しない**ので、索引づくりの速度コストはゼロ。
  重みは `best_model.pt` に入るが、`PhaseNet.load_for_inference()` が捨てる。
- **`other` クラス**: phase ヘッドの最後のクラスを「自分の術式ではない」に割り当てる。
  相手データセットのフレームを負例として混ぜて学習することで、**別モデルを足さずに
  「どちらの工程分類器を使うか」の router と未知術式のフォールバック判定を賄う**。
"""
from __future__ import annotations

import logging
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class LightFPN(nn.Module):
    """C2–C5 を 1/4 解像度へ束ねてバイナリ器具マスクを出す軽量デコーダ.

    アノテーションが 2,235 枚しかないので、容量は意図的に小さくしてある
    （lateral 1x1 → 加算 → 3x3 → 1ch）。目的は主タスク(phase)の特徴を
    器具の位置に引き寄せることであって、seg 自体の精度ではない。
    """

    def __init__(self, in_channels: list[int], mid: int = 96):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, mid, 1) for c in in_channels])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1), nn.GroupNorm(8, mid), nn.GELU())
            for _ in in_channels[:-1]
        ])
        self.out = nn.Conv2d(mid, 1, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # feats は解像度が高い順（C2, C3, C4, C5）
        laterals = [l(f) for l, f in zip(self.lateral, feats)]
        x = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            x = F.interpolate(x, size=laterals[i].shape[-2:], mode="nearest") + laterals[i]
            x = self.smooth[i](x)
        return self.out(x)   # 1/4 解像度の logit


class PhaseNet(nn.Module):
    def __init__(self, backbone: str = "convnextv2_tiny", n_phase: int = 8,
                 n_tool: int = 0, seg: bool = False, drop_path: float = 0.1,
                 pretrained: bool = True, pretrained_ckpt: str | None = None):
        super().__init__()
        self.backbone_name = backbone
        self.n_phase = n_phase
        self.n_tool = n_tool
        self.has_seg = seg

        use_timm_pretrained = pretrained and not pretrained_ckpt
        self.encoder = timm.create_model(
            backbone, pretrained=use_timm_pretrained, num_classes=0, drop_path_rate=drop_path)
        if pretrained_ckpt:
            self._load_ssl(pretrained_ckpt)

        dim = self.encoder.num_features
        self.head_phase = nn.Linear(dim, n_phase)
        self.head_tool = nn.Linear(dim, n_tool) if n_tool > 0 else None

        if seg:
            chs = self._probe_channels()
            self.decoder = LightFPN(chs)
        else:
            self.decoder = None

    # ------------------------------------------------------------------ #
    def _load_ssl(self, ckpt: str) -> None:
        """SurgeNet 等の手術特化 SSL 重みで初期化する（expC00 と同じ経路）。"""
        raw = torch.load(ckpt, map_location="cpu")
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        if "convnext" in self.backbone_name:
            from timm.models.convnext import checkpoint_filter_fn
            raw = checkpoint_filter_fn(raw, self.encoder)
        res = self.encoder.load_state_dict(raw, strict=False)
        n_total = len(self.encoder.state_dict())
        log.info("SSL ckpt %s: loaded %d/%d (missing=%d unexpected=%d)",
                 Path(ckpt).name, n_total - len(res.missing_keys), n_total,
                 len(res.missing_keys), len(res.unexpected_keys))
        if len(res.missing_keys) > 0.5 * n_total:
            raise RuntimeError(
                f"SSL 重みがほとんど読めていない（missing={len(res.missing_keys)}/{n_total}）。"
                "backbone 名と ckpt の対応を確認すること")

    @torch.no_grad()
    def _probe_channels(self) -> list[int]:
        was_training = self.encoder.training
        self.encoder.eval()
        x = torch.zeros(1, 3, 128, 128)
        _, inter = self.encoder.forward_intermediates(x)
        self.encoder.train(was_training)
        return [t.shape[1] for t in inter]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, want_seg: bool = False) -> dict[str, torch.Tensor]:
        if want_seg and self.decoder is not None:
            feat, inter = self.encoder.forward_intermediates(x)
            g = feat.mean((2, 3))
            out = {"phase": self.head_phase(g), "seg": self.decoder(inter)}
        else:
            g = self.encoder(x)          # num_classes=0 なので pooled 特徴が出る
            out = {"phase": self.head_phase(g)}
        out["feat"] = g
        if self.head_tool is not None:
            out["tool"] = self.head_tool(g)
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def load_for_inference(path: str | Path, device: str = "cuda",
                           drop_decoder: bool = True) -> "PhaseNet":
        """`best_model.pt` から索引用のモデルを組む。既定で seg デコーダを捨てる。"""
        st = torch.load(path, map_location="cpu", weights_only=False)
        cfg = st["model_cfg"]
        if drop_decoder:
            cfg = {**cfg, "seg": False}
        m = PhaseNet(pretrained=False, **cfg)
        sd = {k: v for k, v in st["model"].items()
              if not (drop_decoder and k.startswith("decoder."))}
        res = m.load_state_dict(sd, strict=False)
        if res.missing_keys:
            raise RuntimeError(f"重みが足りない: {res.missing_keys[:5]}")
        return m.to(device).eval()
