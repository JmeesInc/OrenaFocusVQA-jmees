"""ローカル回帰テスト用の /input バッチを実データから作る（公式仕様に一致させる）.

公式コンテナが受け取るものを再現する:
  test/input/interface_1/request.json        focus.Request のリスト
  test/input/interface_1/frames/<qID>.png    **ネイティブ解像度の可逆PNG**（本番と同じ）
  test/input/interface_1/FO_definitions.json FO定義テキストを JSON エンコードしたもの
  test/input/interface_1/batch.json          qID 索引

★**fold v003（`qa_v004`）の fold0 val から取る**。v001〜v004 の同名スクリプトは
  fold **v001** の split を使っていたが、v006 のメンバー（expV05/V06/E10）は
  **fold v003 で学習**しているので、v001 の val を使うと学習済み動画が混ざる。

★バッチの組み方も本番に揃える: "no two questions in a batch come from the same source video"

Usage (repo ルートから):
  PYTHONPATH=reference/src .venv/bin/python submit/v006_frame_vote/make_test_batch.py --n 20
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from focus import FO_DEFINITIONS_FILE, FocusConfig, FocusDataset, save_items, set_config  # noqa: E402
from focus.enums import DatasetSplit, Track  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
for n in ("httpx", "httpcore", "filelock", "fsspec", "urllib3"):
    logging.getLogger(n).setLevel(logging.WARNING)
log = logging.getLogger("make_test_batch")

DATA_ROOT = ROOT / "data/focus"
OUT = Path(__file__).parent / "test/input/interface_1"

# ★fold v003 fold0 val に紛れている「train 側動画のバイト完全一致コピー」
DUP_VAL_VIDEOS = {"0027 - Laparoscopic Cholecystectomy.mp4"}


def find_video(dataset: str, video_id: str) -> Path:
    p = DATA_ROOT / dataset / "videos" / video_id
    if p.exists():
        return p
    cands = list((DATA_ROOT / dataset / "videos").glob(f"{Path(video_id).stem}*"))
    if not cands:
        raise FileNotFoundError(video_id)
    return cands[0]


def extract_png(dataset: str, video_id: str, t: float, out: Path) -> bool:
    """本番と同じ **ネイティブ解像度の可逆PNG** を抽出（リサイズしない）。"""
    if out.exists():
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}",
                    "-i", str(find_video(dataset, video_id)), "-frames:v", "1", str(out)],
                   capture_output=True)
    return out.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v004", help="qa fold バージョン（v004 = fold v003）")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-same-video", action="store_true",
                    help="1動画1問の制約を外す（精度をまとまった数で測る用途にだけ使う）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)

    set_config(FocusConfig(root_dir=str(DATA_ROOT)))

    rows = []
    with (ROOT / f"workspace/fold/qa_{args.version}/qa_split.csv").open() as f:
        for r in csv.DictReader(f):
            if r["track"] == "FRAME" and int(r["fold"]) == args.fold \
                    and r["videoID"] not in DUP_VAL_VIDEOS:
                rows.append(r)
    log.info(f"fold{args.fold} val FRAME: {len(rows)} 問")

    idx = {}
    for dsn in ("heico", "lapchole"):
        ds = FocusDataset(dsn, DatasetSplit.ALL, Track.FRAME)
        for req in ds.requests:
            idx[(dsn, req.qID)] = req

    random.Random(args.seed).shuffle(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    picked, seen, n_fail = [], set(), 0
    for r in rows:
        if len(picked) >= args.n:
            break
        if not args.allow_same_video and r["videoID"] in seen:
            continue
        req = idx.get((r["dataset"], r["qID"]))
        if req is None:
            continue
        if extract_png(r["dataset"], req.videoID, req.start_time,
                       out_dir / "frames" / f"{req.qID}.png"):
            picked.append(req)
            seen.add(r["videoID"])
        else:
            n_fail += 1

    save_items(picked, out_dir / "request.json")
    (out_dir / "FO_definitions.json").write_text(json.dumps(FO_DEFINITIONS_FILE.read_text()))
    (out_dir / "batch.json").write_text(json.dumps(
        {"qIDs": [r.qID for r in picked], "n": len(picked), "track": "frame"}, indent=1))
    sizes = [p.stat().st_size for p in (out_dir / "frames").glob("*.png")]
    log.info(f"wrote {len(picked)} questions to {out_dir} (extract fail {n_fail}); "
             f"videos {len(seen)}; PNG avg {sum(sizes)/max(len(sizes),1)/1e6:.1f}MB")


if __name__ == "__main__":
    main()
