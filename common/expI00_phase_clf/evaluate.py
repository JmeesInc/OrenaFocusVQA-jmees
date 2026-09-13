r"""stage-1 / stage-2 の評価（書き出し済みの特徴の上で回すので速い）.

指標:
- **frame accuracy**（プールと動画平均の両方）… Cholec80 の文献値と比較できるのは動画平均の方
- **macro-F1** / phase 別 precision・recall・Jaccard
- **セグメント編集距離**（過分割の指標。索引としての使い勝手に直結する）
- **stride を落としたときの劣化**（本番は長尺で 10s / 20s 走査になるため）

    ./run.sh eval heico
    ./run.sh eval cholec --split test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset as D  # noqa: E402

log = logging.getLogger("eval")


def segments(seq: np.ndarray) -> list[tuple[int, int, int]]:
    """[(start, end, label)] へ畳む。"""
    out, s = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[s]:
            out.append((s, i, int(seq[s])))
            s = i
    return out


def edit_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """セグメント列の Levenshtein 距離を正規化したスコア（100 = 完全一致）。"""
    p = [l for _, _, l in segments(pred)]
    g = [l for _, _, l in segments(gt)]
    m, n = len(p), len(g)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1); dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1,
                           dp[i - 1, j - 1] + (p[i - 1] != g[j - 1]))
    return 100.0 * (1.0 - dp[m, n] / max(m, n, 1))


def per_phase_metrics(pred: np.ndarray, gt: np.ndarray, n_own: int) -> dict:
    cm = confusion_matrix(gt, pred, labels=list(range(n_own + 1)))
    out = {}
    for c in range(n_own):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        out[c] = {
            "n": int(cm[c, :].sum()),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "jaccard": float(tp / max(tp + fp + fn, 1)),
        }
    return out


def anon_mask(video: str, sec: np.ndarray) -> np.ndarray:
    """匿名化（ベタ青/黒）区間に入るフレームの真偽を返す.

    ★2026-08-19 実測: **heico の phase 9 は 100.0% が匿名区間**（val 15,963/15,963）で、
      val 全体の 23.3% を占める。つまり「青を見分ける」だけで phase 9 の acc 0.999 が出る。
      これを含めた数字は工程認識の実力ではないので、**必ず clean 側も併記する**。
      区間表は expE00 が作った `anon_intervals.csv`（BLUE/BLACK, 秒）。
    """
    iv = _anon_table()
    sub = iv[iv["video"] == video]
    m = np.zeros(len(sec), dtype=bool)
    for s, e in zip(sub["start"].to_numpy(), sub["end"].to_numpy()):
        m |= (sec >= s) & (sec <= e)
    return m


_ANON_CACHE: dict = {}


def _anon_table():
    if "df" not in _ANON_CACHE:
        import pandas as pd
        path = HERE.parents[1] / "workspace/expE00_segproc_eda/anon_intervals.csv"
        df = pd.read_csv(path) if path.exists() else None
        if df is None:
            log.warning("%s が無い。匿名区間を除いた指標は出せない", path)
        _ANON_CACHE["df"] = df
    return _ANON_CACHE["df"]


def load_tcn(tcn_dir: Path, stride: int, device: str):
    path = tcn_dir / f"tcn_stride{stride}.pt"
    if not path.exists():
        return None
    from train_tcn import MSTCN
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    m = MSTCN(st["in_dim"], int(cfg["channels"]), st["n_classes"], int(cfg["n_stages"]),
              int(cfg["n_layers"]), float(cfg["dropout"]))
    m.load_state_dict(st["model"])
    return m.to(device).eval()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--strides", default="1,5,10,20")
    ap.add_argument("--feat-dir", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / args.config).read_text())
    ds, fold = cfg["data"]["dataset"], int(cfg["cv"]["fold"])
    n_own = D.N_PHASE[ds]

    if args.feat_dir:
        feat_dir = Path(args.feat_dir)
    else:
        import extract_features as EF
        feat_dir = EF.find_best(cfg, fold).parent / "features"
    tcn_dir = feat_dir.parent / "tcn"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if ds == "heico":
        tr_v, va_v = D.heico_split(fold, cfg["cv"]["fold_version"])
        videos = va_v if args.split == "val" else va_v
    else:
        tr_v, va_v, te_v = D.cholec_split()
        videos = va_v if args.split == "val" else te_v
    log.info("%s %s: %d videos", ds, args.split, len(videos))

    report: dict = {"dataset": ds, "split": args.split, "n_videos": len(videos), "strides": {}}
    dumps = {}
    for stride in [int(s) for s in args.strides.split(",")]:
        tcn = load_tcn(tcn_dir, stride, device) if stride > 1 else None
        P1, PT, G, AN, accs1, accsT, eds1, edsT = [], [], [], [], [], [], [], []
        for v in sorted(videos):
            z = np.load(feat_dir / f"{v}.npz")
            lab = z["label"].astype(np.int64)[::stride]
            p1 = z["logit"].astype(np.float32)[::stride].argmax(1)
            AN.append(anon_mask(v, z["sec"].astype(np.int64)[::stride])
                      if _anon_table() is not None else np.zeros(len(lab), bool))
            P1.append(p1); G.append(lab)
            accs1.append(float((p1 == lab).mean())); eds1.append(edit_score(p1, lab))
            if tcn is not None:
                x = torch.from_numpy(z["feat"].astype(np.float32)[::stride].T).unsqueeze(0).to(device)
                with torch.no_grad():
                    pt = tcn(x)[-1].argmax(1).squeeze(0).cpu().numpy()
                PT.append(pt)
                accsT.append(float((pt == lab).mean())); edsT.append(edit_score(pt, lab))
                if stride == 5:
                    dumps[v] = {"gt": lab, "stage1": p1, "tcn": pt}
        p1, g, anon = np.concatenate(P1), np.concatenate(G), np.concatenate(AN)
        own = g < n_own
        clean = own & ~anon
        entry = {
            "stage1": {
                "frame_acc": float((p1[own] == g[own]).mean()),
                "video_acc": float(np.mean(accs1)),
                "macro_f1": float(f1_score(g[own], p1[own], average="macro",
                                           labels=list(range(n_own)), zero_division=0)),
                "edit": float(np.mean(eds1)),
                "other_leak": float((p1[own] == n_own).mean()),
                "per_phase": per_phase_metrics(p1[own], g[own], n_own),
                # ★匿名（ベタ青/黒）区間を除いた実力値。heico はこちらが本当の数字
                "anon_frac": float(anon[own].mean()),
                "frame_acc_clean": float((p1[clean] == g[clean]).mean()) if clean.any() else None,
                "macro_f1_clean": float(f1_score(g[clean], p1[clean], average="macro",
                                                 labels=list(range(n_own)), zero_division=0))
                if clean.any() else None,
            }
        }
        if PT:
            pt = np.concatenate(PT)
            entry["tcn"] = {
                "frame_acc": float((pt[own] == g[own]).mean()),
                "video_acc": float(np.mean(accsT)),
                "macro_f1": float(f1_score(g[own], pt[own], average="macro",
                                           labels=list(range(n_own)), zero_division=0)),
                "edit": float(np.mean(edsT)),
                "other_leak": float((pt[own] == n_own).mean()),
                "per_phase": per_phase_metrics(pt[own], g[own], n_own),
                "anon_frac": float(anon[own].mean()),
                "frame_acc_clean": float((pt[clean] == g[clean]).mean()) if clean.any() else None,
                "macro_f1_clean": float(f1_score(g[clean], pt[clean], average="macro",
                                                 labels=list(range(n_own)), zero_division=0))
                if clean.any() else None,
            }
        report["strides"][str(stride)] = entry
        s1, st = entry["stage1"], entry.get("tcn")
        log.info("stride %2ds | stage1 acc %.4f (clean %.4f) macro-F1 %.4f (clean %.4f) "
                 "edit %.1f | anon %.1f%%%s",
                 stride, s1["frame_acc"], s1["frame_acc_clean"] or float("nan"),
                 s1["macro_f1"], s1["macro_f1_clean"] or float("nan"), s1["edit"],
                 100 * s1["anon_frac"],
                 "" if not st else
                 f"  ||  +TCN acc {st['frame_acc']:.4f} (clean {st['frame_acc_clean'] or float('nan'):.4f}) "
                 f"macro-F1 {st['macro_f1']:.4f} (clean {st['macro_f1_clean'] or float('nan'):.4f}) "
                 f"edit {st['edit']:.1f}")

    outdir = feat_dir.parent / "eval"
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(report, (outdir / f"report_{args.split}.json").open("w"), indent=2)
    if dumps:
        np.savez_compressed(outdir / f"timelines_{args.split}.npz",
                            **{f"{v}__{k}": arr for v, d in dumps.items() for k, arr in d.items()})
    log.info("wrote %s", outdir / f"report_{args.split}.json")


if __name__ == "__main__":
    main()
