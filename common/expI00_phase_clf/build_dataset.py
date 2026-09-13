r"""工程分類モデル用のフレーム抽出とラベル表の作成.

## 何を作るか

```
<PHASE_ROOT>/
├── frames/cholec80/video01/000000.jpg …   # 1 fps, 高さ 576px
├── frames/heico/0000 - Heico - Prokto - 1/000000.jpg …
├── labels/cholec80.parquet                # sec 単位の phase + tool presence
├── labels/heico.parquet                   # sec 単位の phase
└── labels/heico_instseg.parquet           # 器具 seg アノテーションのある個別フレーム
```

## 抽出仕様（本番クリップを模す）

本番の PROCEDURE クリップは **H.264 / 5 fps / 高さ最大 576px**。学習フレームも
`fps=1, scale=-2:576` で作り、576px より上の解像度を学習側に持ち込まない。
（Cholec80 の video78-80 だけ 1920x1080、残り 77 本は 854x480 なので、
 幅を固定と仮定せず高さだけ揃える。`-2` で幅は偶数に丸める）

## ラベルの引き方

- **Cholec80** `phase_annotations/video01-phase.txt` … TSV・ヘッダ `Frame\tPhase`・**25 fps 全フレーム**
  `tool_annotations/video01-tool.txt` … TSV・ヘッダ 8 列・**25 フレーム刻み (= 1 fps)**
- **HeiCo** `<Procedure>/<n>/<Procedure>_<n>_Phase.csv` … **ヘッダなし** `frame_index,phase_id`・25 fps 全フレーム

いずれも `t` 秒目のフレームのラベル = 行 `t * 25`。行数は動画の nb_frames と一致する
（30/30 本で照合済み）ので、末尾だけ範囲クランプすれば足りる。

## HeiCo の動画 ID 対応（全 30 本を Phase.csv 行数で照合済み）

| フォルダ | videoID |
|---|---|
| `Proctocolectomy/1..10`  | `0000`〜`0009` (`Prokto - 1..10`) |
| `Rectal Resection/1..10` | `0010`〜`0019` (`Rektum - 1..10`) |
| `Sigmoid Resection/1..10`| `0020`〜`0029` (`Sigma - 1..10`) |

## 注意

- **`/tmp` の空きがゼロ**なので `TMPDIR` をスクラッチへ向けてから走らせる（`run.sh` が設定する）
- NFS 越しの読み出しが律速する。`--jobs` で動画単位に並列化する
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

log = logging.getLogger("build_dataset")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

PHASE_ROOT = Path("/mnt/data/data4/shared/miccai/Orena/phase")
CHOLEC_ROOT = Path("/data4/shared/Cholecystostomy/Cholec80")
HEICO_ROOT = Path("/data4/shared/miccai/Orena/focus/heico")

# 抽出仕様。本番クリップ（高さ最大 576px）に合わせる。
EXTRACT_HEIGHT = 576
EXTRACT_FPS = 1.0
JPEG_Q = 3

# Cholec80: README の 7 phase。ID は EndoNet 論文の並び。
CHOLEC_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]
CHOLEC_TOOLS = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]

# HeiCo: 整数 ID 0..13 の 14 クラス。名称表はデータ内に存在しない（索引用途では不要）。
HEICO_N_PHASES = 14

# フォルダ名 → (videoID 連番の起点, videos/ の中のドイツ語略称)
HEICO_PROCEDURES = {
    "Proctocolectomy": (0, "Prokto"),
    "Rectal Resection": (10, "Rektum"),
    "Sigmoid Resection": (20, "Sigma"),
}

# 25 fps 固定（reference/src/focus/config.py の DATASET_BASE_FPS と一致）
BASE_FPS = 25

# ★ラベルは動画単位の shard に書き切る（親プロセスが落ちても失わない）
PART_ROOT = PHASE_ROOT / "labels" / "parts"


def _write_shard(rows: list[dict], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(out)          # 部分書きの shard を残さない
    return out


# --------------------------------------------------------------------------- #
# 共通ユーティリティ
# --------------------------------------------------------------------------- #
def setup_logging(logfile: Path | None = None) -> None:
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        handlers.append(fh)
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


def probe(path: Path) -> dict:
    """duration / width / height / nb_frames を返す。"""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,nb_frames,r_frame_rate",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    return {
        "width": int(st["width"]),
        "height": int(st["height"]),
        "nb_frames": int(st.get("nb_frames") or 0),
        "r_frame_rate": st.get("r_frame_rate", ""),
        "duration": float(j["format"]["duration"]),
    }


def extract_frames(video: Path, out_dir: Path, expect: int | None = None,
                   force: bool = False) -> int:
    """`fps=1` で JPEG を吐く。既に完了していれば何もしない（冪等）。

    出力 i 枚目（0 origin）の時刻は i 秒。`_DONE` に枚数を書いて完了を示す。

    ★完了判定は `_DONE` の有無ではなく **`expect`（アノテーション行数から出した期待枚数）
      との一致**で行う。2026-08-17 の dl1 再起動で、17 本ぶんの `_DONE` が
      **中身だけ空**（NFS がファイルは作ったが内容を flush していない）で残り、
      素朴に `int()` して全体が落ちた。しかも実 JPEG は 17 本とも完全だったので、
      マーカーを信じて再抽出するのも、マーカーだけ見て通すのも、どちらも誤り。
    """
    done = out_dir / "_DONE"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _marker() -> int | None:
        try:
            return int(done.read_text().strip())
        except (OSError, ValueError):
            return None

    if not force:
        n = _marker()
        if n is None and done.exists():
            log.warning("%s: _DONE が壊れている。実 JPEG 枚数で検証し直す", out_dir.name)
        if n is None:
            n = len(list(out_dir.glob("*.jpg"))) or None
        # ★枚数が合っていても**中身が空**のことがある。2026-08-17 の dl1 再起動で
        #   17 本ぶん・計 20,398 枚が 0 バイトで残り（ディレクトリ項目だけ NFS に
        #   flush され、データは失われた）、うち 5 本は全枚数が 0 バイトだった。
        #   枚数一致だけで通す検証は**この壊れ方を素通しする**。
        n_empty = sum(1 for f in out_dir.glob("*.jpg") if f.stat().st_size == 0)
        if n_empty:
            log.warning("%s: 0 バイトの JPEG が %d 枚ある → 再抽出", out_dir.name, n_empty)
            n = None
        if n is not None and (expect is None or abs(n - expect) <= 1):
            done.write_text(str(n))          # 壊れたマーカーを直す
            return n
        if n is not None and expect is not None:
            log.warning("%s: 枚数不一致 (実 %d / 期待 %d) → 再抽出", out_dir.name, n, expect)


    for old in out_dir.glob("*.jpg"):
        old.unlink()

    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
           "-vf", f"fps={EXTRACT_FPS},scale=-2:{EXTRACT_HEIGHT}",
           "-vsync", "0", "-q:v", str(JPEG_Q),
           str(out_dir / "%06d.jpg")]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video.name}: "
                           f"{r.stderr.decode('utf-8', 'replace')[:500]}")

    # ffmpeg の %06d は 1 origin なので 0 origin へ振り直す（秒 = ファイル名にする）
    files = sorted(out_dir.glob("*.jpg"))
    for i, f in enumerate(files):
        target = out_dir / f"{i:06d}.jpg"
        if f != target:
            f.rename(target)
    n = len(files)
    n_empty = sum(1 for f in out_dir.glob("*.jpg") if f.stat().st_size == 0)
    if n_empty:
        raise RuntimeError(f"{video.name}: 抽出後も 0 バイトの JPEG が {n_empty} 枚ある")
    n = len(files)
    if expect is not None and abs(n - expect) > 1:
        raise RuntimeError(f"{video.name}: 抽出枚数 {n} が期待 {expect} と違う"
                           "（デコード途中で切れた可能性）")
    done.write_text(str(n))
    return n


# --------------------------------------------------------------------------- #
# Cholec80
# --------------------------------------------------------------------------- #
def cholec_videos(limit: int | None) -> list[str]:
    vids = [f"video{i:02d}" for i in range(1, 81)]
    return vids[:limit] if limit else vids


def read_cholec_phase(vid: str) -> list[str]:
    """フレーム index → phase 名（25 fps 全フレーム）。"""
    path = CHOLEC_ROOT / "phase_annotations" / f"{vid}-phase.txt"
    labels: list[str] = []
    with path.open() as f:
        header = f.readline()
        if not header.lower().startswith("frame"):
            raise ValueError(f"{path}: 期待したヘッダが無い: {header!r}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            labels.append(parts[1].strip())
    return labels


def read_cholec_tools(vid: str) -> dict[int, list[int]]:
    """フレーム index（25 の倍数）→ 7 器具の presence。"""
    path = CHOLEC_ROOT / "tool_annotations" / f"{vid}-tool.txt"
    out: dict[int, list[int]] = {}
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        cols = [c.strip() for c in header[1:]]
        if cols != CHOLEC_TOOLS:
            raise ValueError(f"{path}: 器具列が想定と違う: {cols}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            out[int(parts[0])] = [int(x) for x in parts[1:8]]
    return out


def build_cholec_one(vid: str, force: bool) -> Path:
    """1 動画ぶんを抽出してラベル shard を書く（**親にリストを返さない**）。

    ★2026-08-17 に 79/80 まで進んだところで dl1 が再起動し、親プロセスが全行を
      抱えていたため parquet が 1 行も残らなかった。動画単位で shard を書き切る。
    """
    out_shard = PART_ROOT / "cholec80" / f"{vid}.parquet"
    video = CHOLEC_ROOT / "videos" / f"{vid}.mp4"
    out_dir = PHASE_ROOT / "frames" / "cholec80" / vid

    # 期待枚数はアノテーション行数（= nb_frames）から出す。抽出の健全性検査に使う
    phases = read_cholec_phase(vid)
    tools = read_cholec_tools(vid)
    # ★shard の有無で早期 return してはいけない。フレーム側の健全性検査を必ず通す
    #   （2026-08-18: 0 バイト JPEG 20,398 枚がこの早期 return で素通しされた）
    n_frames = extract_frames(video, out_dir, expect=round(len(phases) / BASE_FPS), force=force)
    if out_shard.exists() and not force:
        return out_shard
    p2i = {p: i for i, p in enumerate(CHOLEC_PHASES)}

    rows = []
    for sec in range(n_frames):
        idx = min(sec * BASE_FPS, len(phases) - 1)
        name = phases[idx]
        if name not in p2i:
            raise ValueError(f"{vid}: 未知の phase ラベル {name!r}")
        tool = tools.get(sec * BASE_FPS)
        rows.append({
            "dataset": "cholec80",
            "videoID": vid,
            "sec": sec,
            "frame_index": sec * BASE_FPS,
            "phase": p2i[name],
            "phase_name": name,
            "has_tool": tool is not None,
            **{f"tool_{t}": (tool[i] if tool else 0) for i, t in enumerate(CHOLEC_TOOLS)},
            "rel_path": f"cholec80/{vid}/{sec:06d}.jpg",
        })
    log.info("cholec80 %s: %d frames (phase rows=%d, tool rows=%d)",
             vid, n_frames, len(phases), len(tools))
    return _write_shard(rows, out_shard)


# --------------------------------------------------------------------------- #
# HeiCo
# --------------------------------------------------------------------------- #
def heico_cases(limit: int | None) -> list[tuple[str, int, str]]:
    """(procedure_dir, case_no, videoID stem) のリスト。"""
    out = []
    for proc, (base, german) in HEICO_PROCEDURES.items():
        for n in range(1, 11):
            vid = f"{base + n - 1:04d} - Heico - {german} - {n}"
            out.append((proc, n, vid))
    out.sort(key=lambda x: x[2])
    return out[:limit] if limit else out


def read_heico_phase(proc: str, n: int) -> list[int]:
    """フレーム index → phase id（ヘッダなし CSV, 25 fps 全フレーム）。

    ★ファイル名からは**空白が抜ける**（ディレクトリは `Rectal Resection/` だが
      中身は `RectalResection_1_Phase.csv`）。ディレクトリ名をそのまま使うと落ちる。
    """
    path = HEICO_ROOT / proc / str(n) / f"{proc.replace(' ', '')}_{n}_Phase.csv"
    labels: list[int] = []
    with path.open() as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            labels.append(int(row[1]))
    return labels


def build_heico_one(proc: str, n: int, vid: str, force: bool) -> Path:
    out_shard = PART_ROOT / "heico" / f"{vid}.parquet"
    video = HEICO_ROOT / "mp4" / f"{vid}.mp4"
    out_dir = PHASE_ROOT / "frames" / "heico" / vid
    phases = read_heico_phase(proc, n)
    n_frames = extract_frames(video, out_dir, expect=round(len(phases) / BASE_FPS), force=force)
    if out_shard.exists() and not force:
        return out_shard
    rows = []
    for sec in range(n_frames):
        idx = min(sec * BASE_FPS, len(phases) - 1)
        pid = phases[idx]
        if not 0 <= pid < HEICO_N_PHASES:
            raise ValueError(f"{vid}: phase id が範囲外 {pid}")
        rows.append({
            "dataset": "heico",
            "videoID": f"{vid}.avi",   # ★fold の folds.csv は .avi 拡張子で持っている
            "video_stem": vid,
            "procedure": proc,
            "case": n,
            "sec": sec,
            "frame_index": sec * BASE_FPS,
            "phase": pid,
            "rel_path": f"heico/{vid}/{sec:06d}.jpg",
        })
    log.info("heico %s (%s/%d): %d frames (phase rows=%d)", vid, proc, n, n_frames, len(phases))
    return _write_shard(rows, out_shard)


def build_heico_instseg() -> pd.DataFrame:
    """器具セグメンテーションのアノテーションフレームを列挙する.

    `<Procedure>/<n>/Instrument segmentations/<frame_index>/` に
    `raw.png` と `instrument_instances.png` が**両方揃っているものだけ**を採る。
    ⚠️ 0 バイトの Synapse 未ダウンロード残骸が大量にあるのでサイズも見る。
    """
    rows = []
    for proc, _ in HEICO_PROCEDURES.items():
        for n in range(1, 11):
            vid = None
            base, german = HEICO_PROCEDURES[proc]
            vid = f"{base + n - 1:04d} - Heico - {german} - {n}"
            seg_root = HEICO_ROOT / proc / str(n) / "Instrument segmentations"
            if not seg_root.is_dir():
                continue
            phases = read_heico_phase(proc, n)
            for d in sorted(seg_root.iterdir(), key=lambda p: p.name):
                if not d.is_dir() or not d.name.isdigit():
                    continue
                raw = d / "raw.png"
                mask = d / "instrument_instances.png"
                if not (raw.is_file() and mask.is_file()):
                    continue
                if raw.stat().st_size == 0 or mask.stat().st_size == 0:
                    continue
                fi = int(d.name)
                rows.append({
                    "dataset": "heico",
                    "videoID": f"{vid}.avi",
                    "video_stem": vid,
                    "procedure": proc,
                    "case": n,
                    "frame_index": fi,
                    "sec": fi // BASE_FPS,
                    "phase": phases[min(fi, len(phases) - 1)],
                    "raw_path": str(raw),
                    "mask_path": str(mask),
                })
    df = pd.DataFrame(rows)
    log.info("heico instseg: %d frames / %d cases",
             len(df), df["case"].nunique() if len(df) else 0)
    return df


# --------------------------------------------------------------------------- #
# エントリポイント
# --------------------------------------------------------------------------- #
def _cholec_worker(args):
    return build_cholec_one(*args)


def _heico_worker(args):
    return build_heico_one(*args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["cholec", "heico", "heico_instseg"])
    ap.add_argument("--limit", type=int, default=None, help="動画数を絞る（疎通確認用）")
    ap.add_argument("--jobs", type=int, default=8, help="動画単位の並列数")
    ap.add_argument("--force", action="store_true", help="抽出済みでもやり直す")
    ap.add_argument("--concat-only", action="store_true",
                    help="抽出はせず、既存 shard から parquet を作り直すだけ")
    ap.add_argument("--out", type=Path, default=PHASE_ROOT / "labels")
    args = ap.parse_args()

    setup_logging(HERE / "results" / "build" / f"build_{args.dataset}.log")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "heico_instseg":
        df = build_heico_instseg()
        path = args.out / "heico_instseg.parquet"
        df.to_parquet(path, index=False)
        log.info("wrote %s (%d rows)", path, len(df))
        return

    if args.dataset == "cholec":
        tasks = [(v, args.force) for v in cholec_videos(args.limit)]
        worker, name = _cholec_worker, "cholec80"
    else:
        tasks = [(p, n, v, args.force) for p, n, v in heico_cases(args.limit)]
        worker, name = _heico_worker, "heico"

    shards: list[Path] = []
    if not args.concat_only:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(worker, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                shards.append(fut.result())
                log.info("[%d/%d] done", i, len(tasks))
    shards = sorted((PART_ROOT / name).glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"{PART_ROOT / name} に shard が無い")
    log.info("concat %d shards", len(shards))
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df = df.sort_values(["videoID", "sec"]).reset_index(drop=True)
    path = args.out / f"{name}.parquet"
    df.to_parquet(path, index=False)
    log.info("wrote %s (%d rows, %d videos)", path, len(df), df["videoID"].nunique())
    log.info("phase 分布:\n%s", df["phase"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
