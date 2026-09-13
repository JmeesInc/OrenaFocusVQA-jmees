r"""stage-2: MS-TCN による時系列平滑化（TeCNO 方式）.

フレーム単体の phase 精度は 75-80% 程度で頭打ちになるが、動画全体を見る 1D TCN を
上に載せると +8〜10pt が定石。**推論コストはほぼゼロ**（1,000 step の dilated conv）なので
索引づくりの速度予算に影響しない。

## 本番の走査刻みに合わせる

索引は 5s / 10s / 20s … の可変 stride で作られる（`retrieval.plan_scan`）。
1 fps で保存した特徴を **stride 秒ごとに間引いた系列**で学習し、**stride ごとに別の重み**を持つ。
位相を stride 通りずらした系列を作って augmentation にする（データが 24〜32 本しか無いため）。

    ./run.sh tcn heico
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset as D  # noqa: E402

log = logging.getLogger("tcn")


class DilatedResidual(nn.Module):
    def __init__(self, dilation: int, ch: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.out = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.conv(x))
        return x + self.drop(self.out(h))


class SSTCN(nn.Module):
    def __init__(self, in_dim: int, ch: int, n_classes: int, n_layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, ch, 1)
        self.layers = nn.ModuleList([DilatedResidual(2 ** i, ch, dropout) for i in range(n_layers)])
        self.out = nn.Conv1d(ch, n_classes, 1)

    def forward(self, x):
        h = self.inp(x)
        for l in self.layers:
            h = l(h)
        return self.out(h)


class MSTCN(nn.Module):
    """stage1 は特徴を、以降は前段の softmax を入力に取る（TeCNO と同じ）。"""

    def __init__(self, in_dim: int, ch: int, n_classes: int, n_stages: int,
                 n_layers: int, dropout: float):
        super().__init__()
        self.stage1 = SSTCN(in_dim, ch, n_classes, n_layers, dropout)
        self.rest = nn.ModuleList([SSTCN(n_classes, ch, n_classes, n_layers, dropout)
                                   for _ in range(n_stages - 1)])

    def forward(self, x) -> list[torch.Tensor]:
        outs = [self.stage1(x)]
        for s in self.rest:
            outs.append(s(F.softmax(outs[-1], dim=1)))
        return outs


def mstcn_loss(outs: list[torch.Tensor], y: torch.Tensor, lam_smooth: float,
               tau: float = 4.0) -> torch.Tensor:
    """CE + truncated MSE の平滑化項（MS-TCN 原論文）。全 stage で取る。"""
    total = torch.zeros((), device=y.device)
    for o in outs:
        total = total + F.cross_entropy(o.transpose(1, 2).reshape(-1, o.shape[1]), y.reshape(-1))
        logp = F.log_softmax(o, dim=1)
        mse = F.mse_loss(logp[:, :, 1:], logp.detach()[:, :, :-1], reduction="none")
        total = total + lam_smooth * torch.clamp(mse, max=tau ** 2).mean()
    return total


# --------------------------------------------------------------------------- #
def load_sequences(feat_dir: Path, videos: set[str], stride: int, phases_only: bool = True
                   ) -> list[tuple[np.ndarray, np.ndarray, str, int]]:
    """(feat[T,D], label[T], videoID, offset) を stride ごと・位相ずらしで作る。"""
    seqs = []
    for v in sorted(videos):
        f = feat_dir / f"{v}.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f} が無い（./run.sh feats を先に回す）")
        z = np.load(f)
        feat, lab = z["feat"].astype(np.float32), z["label"].astype(np.int64)
        for off in range(stride):
            sub_f, sub_l = feat[off::stride], lab[off::stride]
            if len(sub_l) < 16:
                continue
            seqs.append((sub_f, sub_l, v, off))
    return seqs


def evaluate(model, seqs, device, n_own: int) -> dict:
    model.eval()
    P, G = [], []
    with torch.no_grad():
        for feat, lab, _, off in seqs:
            if off != 0:
                continue                      # 評価は位相 0 の系列だけ
            x = torch.from_numpy(feat.T).unsqueeze(0).to(device)
            p = model(x)[-1].argmax(1).squeeze(0).cpu().numpy()
            P.append(p); G.append(lab)
    p, g = np.concatenate(P), np.concatenate(G)
    own = g < n_own
    return {"acc": float((p[own] == g[own]).mean()),
            "macro_f1": float(f1_score(g[own], p[own], average="macro",
                                       labels=list(range(n_own)), zero_division=0))}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_tcn.yaml")
    ap.add_argument("--dataset", required=True, choices=["cholec", "heico"])
    ap.add_argument("--feat-dir", default="", help="未指定なら stage-1 の best から探す")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / args.config).read_text())
    stage1_cfg = yaml.safe_load((HERE / f"config_{args.dataset}.yaml").read_text())
    fold = int(stage1_cfg["cv"]["fold"])
    n_own = D.N_PHASE[args.dataset]
    n_cls = n_own + 1

    if args.feat_dir:
        feat_dir = Path(args.feat_dir)
    else:
        import extract_features as EF
        feat_dir = EF.find_best(stage1_cfg, fold).parent / "features"

    if args.dataset == "heico":
        tr_v, va_v = D.heico_split(fold, stage1_cfg["cv"]["fold_version"])
    else:
        tr_v, va_v, _ = D.cholec_split()

    torch.manual_seed(int(cfg["seed"])); np.random.seed(int(cfg["seed"]))
    device = "cuda"
    results = {}
    for stride in cfg["strides"]:
        tr = load_sequences(feat_dir, tr_v, stride)
        va = load_sequences(feat_dir, va_v, stride)
        in_dim = tr[0][0].shape[1]
        model = MSTCN(in_dim, int(cfg["channels"]), n_cls, int(cfg["n_stages"]),
                      int(cfg["n_layers"]), float(cfg["dropout"])).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
        log.info("stride=%ds | train seq %d / val seq %d | in_dim=%d", stride, len(tr), len(va), in_dim)

        best, best_state = -1.0, None
        for ep in range(int(cfg["epochs"])):
            model.train()
            perm = np.random.permutation(len(tr))
            tot = 0.0
            for i in perm:
                feat, lab, _, _ = tr[i]
                x = torch.from_numpy(feat.T).unsqueeze(0).to(device)
                y = torch.from_numpy(lab).unsqueeze(0).to(device)
                opt.zero_grad()
                loss = mstcn_loss(model(x), y, float(cfg["lambda_smooth"]))
                loss.backward(); opt.step()
                tot += loss.item()
            m = evaluate(model, va, device, n_own)
            log.info("  ep%02d loss %.4f | val acc %.4f macro-F1 %.4f",
                     ep, tot / len(tr), m["acc"], m["macro_f1"])
            if m["macro_f1"] > best:
                best = m["macro_f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                results[str(stride)] = {"epoch": ep, **m}

        outdir = feat_dir.parent / "tcn"
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "cfg": cfg, "stride": stride, "n_classes": n_cls,
                    "in_dim": in_dim, "metrics": results[str(stride)]},
                   outdir / f"tcn_stride{stride}.pt")
        log.info("stride=%ds best macro-F1 %.4f -> %s", stride, best, outdir)

    json.dump(results, (feat_dir.parent / "tcn" / "metrics.json").open("w"), indent=2)
    log.info("done: %s", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
