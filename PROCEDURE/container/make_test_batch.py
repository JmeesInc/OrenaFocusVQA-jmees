"""ローカル回帰テスト用の /input バッチを実データから作る（**本番クリップ仕様に一致させる**）.

公式コンテナが受け取るものを再現する:
  test/input/interface_1/request.json         focus.Request のリスト
  test/input/interface_1/plain/<qID>.mp4      **窓に切り出し済み**のクリップ
  test/input/interface_1/FO_definitions.json  FO定義テキストを JSON エンコードしたもの
  test/input/interface_1/batch.json           qID 索引

★クリップ仕様（公式提出テンプレート `procedure-algorithm/inference.py` の docstring）:
  H.264 MP4 / **厳密に 5 fps** / 高さ最大 **576px**（幅は元動画のアスペクト比）/
  **キーフレーム 5 秒ごと**（5fps なので `-g 25`）。
  ここを本番と揃えないと「デコードが間に合うか」の検証にならない。

★バッチの組み方も本番に揃える: "no two questions in a batch come from the same source video"
  → **1動画1問**でサンプリングする。

fold v003（`qa_v004`）の fold0 val から取るので、既知の CV と同じ母集団になる。

Usage (repo ルートから):
  PYTHONPATH=reference/src .venv/bin/python submit/v005_segment_hybrid/make_test_batch.py --n 20
"""
from __future__ import annotations

import argparse
import csv
import os
import json
import logging
import random
import subprocess
import sys
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

# ★fold v003 fold0 val に紛れている「train 側動画のバイト完全一致コピー」。
#   eval_seg.py と同じく除外する。
DUP_VAL_VIDEOS = {"0027 - Laparoscopic Cholecystectomy.mp4"}


def find_video(dataset: str, video_id: str) -> Path:
    p = DATA_ROOT / dataset / "videos" / video_id
    if p.exists():
        return p
    cands = list((DATA_ROOT / dataset / "videos").glob(f"{Path(video_id).stem}*"))
    if not cands:
        raise FileNotFoundError(video_id)
    return cands[0]


def make_clip(dataset: str, video_id: str, start: float, end: float, out: Path) -> bool:
    """[start, end] を本番仕様（5fps / 高さ<=576 / キーフレーム5秒ごと）で切り出す。

    ★同じ (動画, 窓) のクリップは**ハードリンクで使い回す**。PROCEDURE は窓が動画全体に
      なることが多く、1本 600MB・エンコードも重いので、`--allow-same-video` で問数を
      増やすときに再エンコードすると現実的な時間で終わらない。
    """
    if out.exists():
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    # ★キーは**安定ハッシュ**にする。組み込み `hash()` は文字列に対し実行ごとに変わる
    #   （PYTHONHASHSEED）ので、次回起動時にキャッシュが当たらず再エンコードになる。
    import hashlib
    h = hashlib.md5(f"{dataset}|{video_id}|{start:.3f}|{end:.3f}".encode()).hexdigest()[:16]
    cache = out.parent.parent / ".clipcache"          # plain/ の外に置く（*.mp4 の集計に混ざらない）
    cache.mkdir(parents=True, exist_ok=True)
    key = cache / f"{h}.mp4"
    if key.exists():
        os.link(key, out)
        return True
    dur = max(end - start, 0.2)
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
         "-i", str(find_video(dataset, video_id)),
         "-vf", "fps=5,scale='trunc(iw*min(1,576/ih)/2)*2':'min(576,ih)'",
         "-c:v", "libx264", "-preset", "veryfast", "-g", "25", "-an",
         "-movflags", "+faststart", str(key)],
        capture_output=True)
    if r.returncode != 0:
        log.warning("ffmpeg failed %s: %s", out.name, r.stderr.decode("utf-8", "replace")[:200])
    if not key.exists():
        return False
    os.link(key, out)
    return True


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v004", help="qa fold バージョン（v004 = fold v003）")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--one-per-video", action="store_true", default=True,
                    help="本番同様、1バッチに同一動画からの問を入れない")
    ap.add_argument("--stratify", action="store_true", default=True,
                    help="5つの capability group を巡回して取る（両方の振り分け経路を通すため）")
    ap.add_argument("--allow-same-video", action="store_true",
                    help="1動画1問の制約を外す。★本番バッチの再現ではなくなるので、"
                         "「精度をまとまった数で測る」用途にだけ使う（fold0 val は25動画しかない）")
    ap.add_argument("--out", default=str(OUT), help="出力先（既定は test/input/interface_1）")
    args = ap.parse_args()
    OUT = Path(args.out)

    set_config(FocusConfig(root_dir=str(DATA_ROOT)))

    # fold0 val の SEGMENT 問を列挙
    rows = []
    with (ROOT / f"workspace/fold/qa_{args.version}/qa_split.csv").open() as f:
        for r in csv.DictReader(f):
            if r["track"] == "PROCEDURE" and int(r["fold"]) == args.fold \
                    and r["videoID"] not in DUP_VAL_VIDEOS:
                rows.append(r)
    log.info(f"fold{args.fold} val SEGMENT: {len(rows)} 問")

    idx = {}
    for dsn in ("heico", "lapchole"):
        ds = FocusDataset(dsn, DatasetSplit.ALL, Track.PROCEDURE)
        for req in ds.requests:
            idx[(dsn, req.qID)] = req

    random.Random(args.seed).shuffle(rows)
    if args.stratify:
        # ★素の無作為抽出だと 20問すべてが object/temporal になる（この2群で 84%）。
        #   それでは **32f@768 側の経路が1度も走らない**＝回帰テストとして穴になるので、
        #   5群を巡回しながら取る。
        from focus import Capability
        byg: dict[str, list] = {}
        for r in rows:
            try:
                g = Capability[r["primary"]].group.name
            except KeyError:
                continue
            byg.setdefault(g, []).append(r)
        log.info("group 別: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(byg.items())))
        order, i = sorted(byg), 0
        rows = []
        while any(byg.values()):
            g = order[i % len(order)]
            if byg[g]:
                rows.append(byg[g].pop())
            i += 1

    OUT.mkdir(parents=True, exist_ok=True)
    picked, seen_videos, n_fail = [], set(), 0
    for r in rows:
        if len(picked) >= args.n:
            break
        if args.one_per_video and not args.allow_same_video and r["videoID"] in seen_videos:
            continue
        req = idx.get((r["dataset"], r["qID"]))
        if req is None:
            continue
        if make_clip(r["dataset"], req.videoID, req.start_time, req.end_time,
                     OUT / "plain" / f"{req.qID}.mp4"):
            picked.append(req)
            seen_videos.add(r["videoID"])
        else:
            n_fail += 1

    save_items(picked, OUT / "request.json")
    (OUT / "FO_definitions.json").write_text(json.dumps(FO_DEFINITIONS_FILE.read_text()))
    (OUT / "batch.json").write_text(json.dumps(
        {"qIDs": [r.qID for r in picked], "n": len(picked), "track": "procedure"}, indent=1))
    sizes = [p.stat().st_size for p in (OUT / "plain").glob("*.mp4")]
    durs = [r.end_time - r.start_time for r in picked]
    log.info(f"wrote {len(picked)} questions to {OUT} (clip fail {n_fail}); "
             f"videos {len(seen_videos)}; clip avg {sum(sizes)/max(len(sizes),1)/1e6:.1f}MB; "
             f"duration min/mean/max {min(durs, default=0):.0f}/"
             f"{sum(durs)/max(len(durs),1):.0f}/{max(durs, default=0):.0f}s")


if __name__ == "__main__":
    main()
