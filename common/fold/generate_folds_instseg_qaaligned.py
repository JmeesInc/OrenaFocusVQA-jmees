"""instseg の fold を **VQA の fold（v003）に揃えて**作る。

なぜ独自の LPT 分割（instseg_v001..v003）をやめるか
--------------------------------------------------
instseg モデルの出力は VQA 側で
  ① FRAME の instseg 重畳入力（提出 v007/v008）
  ② PROCEDURE の FO 索引（expI03/I04）
に使われる。この用途では「**VQA の val 動画を instseg の学習に使っていない**」ことが
CV の前提になる。ところが instseg_v003 は VQA fold とは独立に切られていたため、
`instseg_v003 train ∩ qa fold v003 fold0(val)` に 6 動画（0828 dump では 8 動画）が
残っており、重畳入力・索引系の CV が過大評価になっていた
（expG00 / expK00 はこの重なりを除外する leak-aware 評価で凌いでいた）。

擬似 VQA 生成（expK00）は最初からこの方針で、
**qa fold v003 の fold0 動画は擬似 QA の生成元から除外**している。
instseg の学習セットも同じ境界に揃えるのがこのバージョン。

割り当て
--------
- **fold 0（= instseg の val）** ... qa fold v003 で fold0 の動画（＝ VQA の val 動画）
- **fold 1（= instseg の train）** ... それ以外すべて
  （qa fold 1〜4 の動画 + VQA が存在しない未ラベル動画）

train.py / export_yolo.py は `--fold 0` で「fold==0 を val、それ以外を train」と扱うので、
2 値の割り当てでそのまま動く。**K-fold ではない**（VQA 側の val が固定なので、
instseg 側だけ回しても意味が無い）。

    python generate_folds_instseg_qaaligned.py \
        --dataset /mnt/data/.../prepared/s30dall_20260828 \
        --version instseg_v004
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger("folds_qaaligned")

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_qa_folds(path: Path) -> dict[str, int]:
    """VQA の動画単位 fold。拡張子はデータセット間で食い違う（heico は .avi / instseg は .mp4）ので
    **stem で突合する**（拡張子を落とさないと heico が黙って 0 件になる）。"""
    with path.open(encoding="utf-8") as f:
        return {Path(r["videoID"]).stem: int(r["fold"]) for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="build_dataset.py の出力ディレクトリ")
    ap.add_argument("--version", default="instseg_v004")
    ap.add_argument("--qa-folds", default=str(REPO_ROOT / "workspace/fold/v003/folds.csv"))
    ap.add_argument("--val-qa-fold", type=int, default=0, help="instseg の val に回す VQA fold")
    ap.add_argument("--out-root", default=str(REPO_ROOT / "workspace" / "fold"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)

    doc = json.loads((Path(args.dataset) / "coco" / "instances.json").read_text(encoding="utf-8"))
    n_ann_per_image: Counter[int] = Counter(a["image_id"] for a in doc["annotations"])
    stats: dict[str, dict] = {}
    for im in doc["images"]:
        ds = "heico" if "heico" in im["video"].lower() else "lapchole"
        s = stats.setdefault(im["video"], {"dataset": ds, "n_pos": 0, "n_neg": 0, "n_inst": 0})
        n = n_ann_per_image.get(im["id"], 0)
        if n:
            s["n_pos"] += 1
            s["n_inst"] += n
        else:
            s["n_neg"] += 1

    qa = load_qa_folds(Path(args.qa_folds))
    rows = []
    for video, s in sorted(stats.items()):
        qf = qa.get(Path(video).stem)
        fold = 0 if qf == args.val_qa_fold else 1
        rows.append({"video": video, "fold": fold, "dataset": s["dataset"],
                     "qa_fold": "" if qf is None else qf,
                     "n_pos": s["n_pos"], "n_neg": s["n_neg"], "n_inst": s["n_inst"]})

    out_dir = Path(args.out_root) / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "folds.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def agg(sel):
        sub = [r for r in rows if sel(r)]
        return {"videos": len(sub), "pos": sum(r["n_pos"] for r in sub),
                "neg": sum(r["n_neg"] for r in sub), "inst": sum(r["n_inst"] for r in sub),
                "heico": sum(r["dataset"] == "heico" for r in sub),
                "no_vqa": sum(r["qa_fold"] == "" for r in sub)}

    meta = {
        "source_dataset": args.dataset,
        "qa_folds_csv": args.qa_folds,
        "split": f"VQA fold v003 の fold{args.val_qa_fold} を instseg の val に固定（K-fold ではない）",
        "val": agg(lambda r: r["fold"] == 0),
        "train": agg(lambda r: r["fold"] == 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    logger.info("→ %s", out_dir / "folds.csv")
    logger.info("val   : %s", meta["val"])
    logger.info("train : %s", meta["train"])
    logger.info("val videos: %s", sorted(r["video"] for r in rows if r["fold"] == 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
