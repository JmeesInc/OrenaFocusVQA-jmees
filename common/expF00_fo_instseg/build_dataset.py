"""SurVis-Anno の dump を「30 秒に 1 フレーム」の instance segmentation 学習セットへ整形する。

背景
----
`export_dataset.py`（SurVis-Anno）の dump は **shape が付いたフレームだけ** を出力する。
そのため dump には 2 つの問題がある:

1. アノテーション初期は全フレームに shape を引いていたので、区間によって
   フレーム密度が 1 frame（30 fps 連番）と 900 frame（30 s おき）で不揃い。
2. 「見たが FO が無かった」フレームは dump に一切現れない → 陰性教師が無い。

このスクリプトは dump（manifest.json + coco/instances.json）だけを入力として、

* 動画を `--stride-seconds` 秒のビンに区切り、**1 ビン 1 フレーム**に間引く
  （密な区間も疎な区間も同じ密度になる）
* アノテーションが 1 つも無いビンは「FO 非存在」とみなし、そのビン中央の
  フレームを動画からデコードして **negative（annotations 無しの image）** として追加

を行い、学習にそのまま使える COCO を書き出す。アノテーションを追加して
dump を作り直したら、新しい dump を `--dump` に渡して再実行すればよい
（negative フレームの PNG は `--neg-cache` に貯まるので再デコードしない）。

陰性とみなす範囲（`--coverage`）
--------------------------------
dump には「アノテータがどこまで見たか」の記録が無いので、方針を選ぶ:

* ``from_start`` (default) : フレーム 0 〜 最終アノテーションフレーム。
  アノテータは動画を先頭から送りながら FO が出た所で描き始めるため、
  最初のアノテーションより前は「見たが FO 無し」= 真の陰性。
  （`workspace/fo_annotation_seeds` の FO 初出現時刻と実測で一致することを
  `--seeds-dir` で検算できる）
* ``span``   : 最初〜最後のアノテーションフレームの間だけ（最も保守的）
* ``full``   : 動画全体（末尾までアノテータが見終わっている確証がある時のみ）
* ``runs``   : アノテーション済みビンの連続塊（ビン間隔 <= --run-gap-bins）のみ

出力
----
    <out>/
      coco/instances.json      # 間引いた positive + negative（annotations 無し）
      images/<video_stem>/NNNNNN.png   # positive は dump への symlink、negative は
                                       # neg-cache への symlink
      videos.csv               # 動画ごとの統計（fold 設計用）
      build_manifest.json      # 入力・パラメータ・統計（再現性）
      audit_seeds.json         # --seeds-dir 指定時のみ

    <neg-cache>/<video_stem>/NNNNNN.png   # デコードした negative フレーム（永続）

使い方
------
    python build_dataset.py \
        --dump /data4/input/focus-lapchole/focus-lapchole_20260807_001 \
        --out  /data4/input/focus-lapchole/prepared/s30_20260807_001 \
        --seeds-dir ../fo_annotation_seeds/json/lapchole

Note: フレーム抽出（PyAV / SAR 補正 / CFR 前提）は SurVis-Anno の
`server/scripts/export_dataset.py` の `extract_frames` と同一ロジックを移植した。
dump の positive フレームと negative フレームがピクセル空間で一致することが必要。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

logger = logging.getLogger("build_dataset")

_TS_RE = re.compile(r"^(\d+):(\d+):(\d+)(?::(\d+))?$")

# seeds JSON の中で「その時刻に FO が実際に見えている」ことを示すコメント
_SEED_VISIBLE_MARKERS = (
    "FRAME-track annotation - visible foreign objects",
    "becomes visible for the first time",
    "is visible for the last time",
)


# ----- 設定 / データ構造 ---------------------------------------------------------


@dataclass
class VideoPlan:
    """1 動画分の間引き計画。"""

    video: str  # manifest の "video"（拡張子付き basename）
    stem: str  # 出力ディレクトリ名（video_path の stem）
    video_path: Path
    fps: float
    n_frames: int
    width: int  # SAR 補正後（= COCO の width）
    height: int
    bin_size: int  # 1 ビンのフレーム数
    keep_pos: dict[int, int] = field(default_factory=dict)  # bin -> 採用した frame_number
    neg_bins: list[int] = field(default_factory=list)
    neg_frames: dict[int, int] = field(default_factory=dict)  # bin -> frame_number
    dropped_bins: list[int] = field(default_factory=list)  # 画像が見つからず捨てたビン
    coverage: tuple[int, int] = (0, -1)  # 陰性を張るビン範囲 [lo, hi]
    neg_phase: int | None = None  # negative フレームを置くビン内オフセット（None = 中央）
    dump: Path | None = None  # 複数 dump を混ぜるのでフレーム PNG の在り処を持つ
    source_index: int = 0
    keep_dense: dict[int, int] = field(default_factory=dict)  # 希少クラス用の密ビン -> frame
    dense_bin_size: int = 0


# ----- 動画メタ情報 --------------------------------------------------------------


def probe_video(path: Path) -> tuple[float, int, int, int]:
    """(fps, n_frames, width, height) を返す。width/height は SAR 補正後。

    n_frames は duration * fps から算出する（nb_frames はコンテナ依存で欠けることがある）。
    """
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or getattr(stream, "guessed_rate", None)
        if rate is None:
            raise RuntimeError(f"cannot determine fps: {path}")
        fps = float(Fraction(rate))
        width, height = int(stream.width or 0), int(stream.height or 0)
        sar = stream.sample_aspect_ratio
        if sar and sar.numerator > 0 and sar.denominator > 0 and sar.numerator != sar.denominator:
            width = max(1, round(width * sar.numerator / sar.denominator))
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / 1_000_000.0
        if duration is None:
            raise RuntimeError(f"cannot determine duration: {path}")
        n_frames = int(round(duration * fps))
    return fps, n_frames, width, height


def extract_frames(video_path: Path, frame_ids: list[int], on_frame) -> tuple[float, list[int]]:
    """frame_ids（昇順）の各フレームを RGB ndarray で on_frame(fid, rgb) に渡す。

    SurVis-Anno server/scripts/export_dataset.py の同名関数の移植。dump の
    positive フレームと同じデコード規則（CFR 前提 + SAR 補正）を保つために
    ロジックを変えないこと。戻り値: (fps, 見つからなかった frame_ids)。
    """
    import av

    missing: list[int] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        out_w, out_h = int(stream.width or 0), int(stream.height or 0)
        sar = stream.sample_aspect_ratio
        if sar and sar.numerator > 0 and sar.denominator > 0 and sar.numerator != sar.denominator:
            out_w = max(1, round(out_w * sar.numerator / sar.denominator))
        guessed = getattr(stream, "guessed_rate", None)
        rate = stream.average_rate or guessed
        if rate is None:
            raise RuntimeError(f"cannot determine fps: {video_path}")
        fps = Fraction(rate)
        time_base = stream.time_base
        half_frame = 1.0 / float(fps) / 2.0

        def pts_to_seconds(frame) -> float | None:
            if frame.pts is None:
                return None
            return float(frame.pts * time_base)

        it = None
        current_pos = -1e9
        for fid in frame_ids:
            target_t = fid / float(fps)
            if it is None or target_t < current_pos - half_frame or target_t > current_pos + 5.0:
                container.seek(int(target_t / time_base), stream=stream, backward=True)
                it = container.decode(stream)
            found = False
            for frame in it:
                t = pts_to_seconds(frame)
                if t is None:
                    continue
                current_pos = t
                if t >= target_t - half_frame:
                    if t > target_t + half_frame:
                        break
                    on_frame(
                        fid,
                        frame.reformat(width=out_w, height=out_h, format="rgb24").to_ndarray(),
                    )
                    found = True
                    break
            if not found:
                missing.append(fid)
                it = None
    return float(fps), missing


# ----- 間引き計画 ----------------------------------------------------------------


def pick_representative(
    candidates: list[dict], bin_lo: int, bin_size: int, policy: str, n_inst: dict[int, int]
) -> dict:
    """1 ビン内の positive フレーム群から代表を 1 枚選ぶ（決定的）。"""
    if policy == "first":
        return min(candidates, key=lambda im: im["frame_number"])
    if policy == "most_instances":
        # 同数なら中央に近い方 → 完全に決定的
        center = bin_lo + bin_size // 2
        return max(
            candidates,
            key=lambda im: (n_inst.get(im["id"], 0), -abs(im["frame_number"] - center)),
        )
    center = bin_lo + bin_size // 2
    return min(candidates, key=lambda im: (abs(im["frame_number"] - center), im["frame_number"]))


def infer_grid_phase(frames: list[int], bin_size: int) -> int | None:
    """アノテータが辿った「30 秒グリッド」のビン内オフセットを推定する。

    疎アノテーション区間では ``f, f+bin_size, f+2*bin_size, ...`` と等間隔に
    フレームが並ぶ。そこに属するフレームの ``f % bin_size`` の最頻値がグリッドの
    位相。negative フレームは **この位相に置く** — アノテータが実際に開いて
    「FO 無し」と判断したのはその位相のフレームであり、ビン中央の別フレームは
    誰も見ていないため（誤陰性の温床になる）。

    等間隔のペアが 1 つも無い（＝全フレーム密アノテーションのみ）動画では
    None を返し、呼び出し側はビン中央にフォールバックする。
    """
    if len(frames) < 2:
        return None
    fs = sorted(set(frames))
    fset = set(fs)
    # 密アノテーション区間のフレームは f±bin_size も必ず埋まっているので
    # 「等間隔」条件を素通りしてしまう。隣接フレームが無い＝疎区間のフレーム
    # だけを候補にして、密区間に位相を潰されないようにする。
    isolated = [f for f in fs if (f - 1) not in fset and (f + 1) not in fset]
    on_grid = [f for f in isolated if (f + bin_size) in fset or (f - bin_size) in fset]
    if not on_grid:
        return None
    counts: dict[int, int] = {}
    for f in on_grid:
        counts[f % bin_size] = counts.get(f % bin_size, 0) + 1
    phase, n = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    # 位相が割れている（複数のグリッドが混在）なら信用しない
    return phase if n >= max(3, 0.5 * len(on_grid)) else None


def compute_coverage(
    annotated_bins: list[int], last_bin_of_video: int, mode: str, run_gap_bins: int
) -> list[tuple[int, int]]:
    """陰性を張るビン範囲（閉区間）のリストを返す。"""
    if not annotated_bins:
        return []
    lo, hi = min(annotated_bins), max(annotated_bins)
    if mode == "full":
        return [(0, last_bin_of_video)]
    if mode == "from_start":
        return [(0, hi)]
    if mode == "span":
        return [(lo, hi)]
    if mode == "runs":
        runs: list[tuple[int, int]] = []
        start = prev = annotated_bins[0]
        for b in annotated_bins[1:]:
            if b - prev > run_gap_bins:
                runs.append((start, prev))
                start = b
            prev = b
        runs.append((start, prev))
        return runs
    raise ValueError(f"unknown coverage mode: {mode}")


def plan_video(
    entry: dict,
    images: list[dict],
    n_inst: dict[int, int],
    args: argparse.Namespace,
    dump: Path,
    cats_by_image: dict[int, set[str]] | None = None,
    dense_classes: set[str] | None = None,
) -> VideoPlan:
    video_path = Path(entry["video_path"])
    if not video_path.exists() and args.video_root:
        video_path = Path(args.video_root) / video_path.name
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {entry['video_path']} (--video-root で指定可)")

    fps, n_frames, width, height = probe_video(video_path)
    bin_size = int(round(args.stride_seconds * fps))
    if bin_size < 1:
        raise ValueError("--stride-seconds が小さすぎます")

    if images:
        coco_wh = {(im["width"], im["height"]) for im in images}
        if len(coco_wh) > 1 or next(iter(coco_wh)) != (width, height):
            logger.warning(
                "%s: COCO の解像度 %s と動画の SAR 補正後 %s が不一致 — negative の解像度がずれます",
                entry["video"],
                sorted(coco_wh),
                (width, height),
            )

    plan = VideoPlan(
        video=entry["video"],
        stem=video_path.stem,
        video_path=video_path,
        fps=fps,
        n_frames=n_frames,
        width=width,
        height=height,
        bin_size=bin_size,
        dump=dump,
    )

    by_bin: dict[int, list[dict]] = {}
    for im in images:
        by_bin.setdefault(im["frame_number"] // bin_size, []).append(im)

    dump_root = dump
    for b, cands in sorted(by_bin.items()):
        ordered = sorted(cands, key=lambda im: im["frame_number"])
        # 代表 → dump に PNG が実在するものだけ（missing_frames 対策）
        chosen = None
        pool = list(ordered)
        while pool:
            cand = pick_representative(pool, b * bin_size, bin_size, args.pick, n_inst)
            if (dump_root / cand["file_name"]).exists():
                chosen = cand
                break
            logger.warning("%s: frame %d の PNG が dump に無い — 同ビンの別候補を試します",
                           plan.stem, cand["frame_number"])
            pool = [im for im in pool if im["id"] != cand["id"]]
        if chosen is None:
            plan.dropped_bins.append(b)
            continue
        plan.keep_pos[b] = chosen["frame_number"]

    # --- 希少クラスの密サンプリング ---
    # 30 秒に 1 枚だと Silicon_Loop が 8 枚 / Gallstone が 20 枚しか取れず学習にならない。
    # 指定クラスが写っているフレームだけ、細かいビン（既定 1 秒）でも 1 枚拾う。
    # ★拾えるのは **dump に実在するフレームだけ** — 疎アノテーション区間は元々
    #   30 秒に 1 枚しか描かれていないので密にしようがない（heico は全区間これ）。
    if dense_classes and cats_by_image:
        dense_bin = max(1, int(round(args.dense_stride_seconds * fps)))
        plan.dense_bin_size = dense_bin
        dense_by_bin: dict[int, list[dict]] = {}
        for im in images:
            if cats_by_image.get(im["id"], set()) & dense_classes:
                dense_by_bin.setdefault(im["frame_number"] // dense_bin, []).append(im)
        base_frames = set(plan.keep_pos.values())
        for db, cands in sorted(dense_by_bin.items()):
            pool = sorted(cands, key=lambda im: im["frame_number"])
            if any(im["frame_number"] in base_frames for im in pool):
                continue  # 既に 30 秒グリッド側で採用済み
            chosen = next((im for im in pool if (dump_root / im["file_name"]).exists()), None)
            if chosen is not None:
                plan.keep_dense[db] = chosen["frame_number"]
        # 1 動画あたりの上限。時間方向に均等に間引く（先頭だけ残すと偏るため）
        cap = args.dense_max_per_video
        if cap is not None and len(plan.keep_dense) > cap:
            bins = sorted(plan.keep_dense)
            idx = [round(i * (len(bins) - 1) / (cap - 1)) for i in range(cap)] if cap > 1 else [0]
            keep = {bins[i] for i in idx}
            logger.info("  %s: 密フレーム %d → %d に間引く（上限 %d）",
                        plan.stem, len(plan.keep_dense), len(keep), cap)
            plan.keep_dense = {b: f for b, f in plan.keep_dense.items() if b in keep}

    last_bin = (n_frames - 1) // bin_size
    ranges = compute_coverage(sorted(by_bin), last_bin, args.coverage, args.run_gap_bins)
    if ranges:
        plan.coverage = (ranges[0][0], ranges[-1][1])
    covered: set[int] = set()
    for lo, hi in ranges:
        covered.update(range(lo, min(hi, last_bin) + 1))

    plan.neg_bins = sorted(covered - set(by_bin))
    if args.neg_phase == "grid":
        plan.neg_phase = infer_grid_phase([im["frame_number"] for im in images], bin_size)
    phase = plan.neg_phase if plan.neg_phase is not None else bin_size // 2
    for b in plan.neg_bins:
        plan.neg_frames[b] = min(b * bin_size + phase, n_frames - 1)
    return plan


# ----- negative フレームのデコード -------------------------------------------------


def materialize_negatives(plan: VideoPlan, cache_root: Path) -> dict[int, Path]:
    """negative フレームを neg-cache に用意し、bin -> PNG パスを返す。既存はスキップ。"""
    from PIL import Image

    out_dir = cache_root / plan.stem
    paths: dict[int, Path] = {}
    todo: list[int] = []
    for b, fid in plan.neg_frames.items():
        p = out_dir / f"{fid:06d}.png"
        paths[b] = p
        if not p.exists():
            todo.append(fid)
    if not todo:
        logger.info("  %s: negative %d 枚すべて cache 済み", plan.stem, len(paths))
        return paths

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  %s: negative %d 枚をデコード（cache 済み %d 枚）",
                plan.stem, len(todo), len(paths) - len(todo))

    def on_frame(fid: int, rgb) -> None:
        Image.fromarray(rgb).save(out_dir / f"{fid:06d}.png", format="PNG")

    _fps, missing = extract_frames(plan.video_path, sorted(todo), on_frame)
    if missing:
        logger.warning("  %s: %d 枚がデコードできず除外（先頭: %s）",
                       plan.stem, len(missing), missing[:5])
        miss = set(missing)
        for b, fid in list(plan.neg_frames.items()):
            if fid in miss:
                del plan.neg_frames[b]
                paths.pop(b, None)
        plan.neg_bins = sorted(plan.neg_frames)
    return paths


def is_degenerate(path: Path, std_threshold: float) -> bool:
    """no-signal のベタ塗り（青一色など）フレームか。輝度の標準偏差で判定。"""
    import numpy as np
    from PIL import Image

    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return float(arr.std()) < std_threshold


# ----- seeds による陰性の検算 -------------------------------------------------------


def parse_timestamp(ts: str) -> float | None:
    m = _TS_RE.match(ts.strip())
    if not m:
        return None
    h, mnt, s, ms = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + int(s) + (int(ms) / 1000.0 if ms else 0.0)


def load_seed_visible_times(seed_path: Path) -> list[float]:
    """「その時刻に FO が見えている」と読める seed コメントの時刻（秒）を返す。"""
    events = json.loads(seed_path.read_text(encoding="utf-8"))
    times: list[float] = []
    for ev in events:
        comment = ev.get("comment", "")
        if not any(marker in comment for marker in _SEED_VISIBLE_MARKERS):
            continue
        t = parse_timestamp(ev.get("timestamp", ""))
        if t is not None:
            times.append(t)
    return sorted(times)


def audit_with_seeds(plan: VideoPlan, seeds_dir: Path) -> dict | None:
    """negative ビンを seeds（QA 逆算の FO 可視イベント）と突き合わせる。

    - conflicts        : negative にしたビンの中に「FO が見えている」証拠がある = 誤陰性の疑い
    - confirmed_before : 最初の FO 可視時刻より前 = 独立ソースが陰性を裏付けたビン
    """
    # seeds は json/<dataset>/<video>.json に置かれている。dataset ごとのディレクトリを
    # 渡しても、その親（json/）を渡しても引けるように再帰でも探す。
    candidates = [seeds_dir / f"{plan.stem}.json", seeds_dir / f"{plan.video}.json"]
    candidates += sorted(seeds_dir.glob(f"*/{plan.stem}.json"))
    seed_path = next((p for p in candidates if p.exists()), None)
    if seed_path is None:
        return None

    times = load_seed_visible_times(seed_path)
    if not times:
        return None
    first_visible = times[0]
    first_ann_frame = min(plan.keep_pos.values()) if plan.keep_pos else None

    conflicts, confirmed = [], []
    for b in plan.neg_bins:
        lo_t = b * plan.bin_size / plan.fps
        hi_t = (b + 1) * plan.bin_size / plan.fps
        if any(lo_t <= t < hi_t for t in times):
            conflicts.append({"bin": b, "frame": plan.neg_frames.get(b), "t_sec": round(lo_t, 1)})
        elif hi_t <= first_visible:
            confirmed.append(b)

    return {
        "video": plan.video,
        "seed_file": str(seed_path),
        "seed_first_visible_sec": round(first_visible, 1),
        "seed_first_visible_frame": int(round(first_visible * plan.fps)),
        "first_annotated_frame": first_ann_frame,
        "first_annotated_minus_seed_sec": (
            round(first_ann_frame / plan.fps - first_visible, 1)
            if first_ann_frame is not None
            else None
        ),
        "n_neg_bins": len(plan.neg_bins),
        "n_confirmed_negative_bins": len(confirmed),
        "n_conflict_bins": len(conflicts),
        "conflicts": conflicts[:200],
    }


# ----- 出力 -----------------------------------------------------------------------


def portable(path: Path, prefix_map: list[tuple[str, str]]) -> Path:
    """symlink 先を「どのマシンからでも解決できる」パスへ書き換える。

    dl1 のローカルパス `/data4/...` は dl2 には存在せず（dl2 では
    `/mnt/data/data4/...` に autofs マウントされる）、`/data4` を指す symlink は
    dl2 から**壊れたリンク**になる。実際 YOLO 学習で positive 画像 78 枚が
    「No such file or directory」で黙って除外された。
    dl1 からは両方の綴りが同じファイルシステムを指すので、常に長い方へ寄せる。
    """
    s = str(path)
    for old, new in prefix_map:
        if s.startswith(old):
            return Path(new + s[len(old):])
    return path


def link(src: Path, dst: Path, prefix_map: list[tuple[str, str]]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(portable(src.resolve(), prefix_map))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dump", required=True, nargs="+",
                    help="export_dataset.py の出力ディレクトリ（複数指定で heico と lapchole を"
                         "1 つの学習セットに統合できる。ラベル体系が同じことが前提）")
    ap.add_argument("--out", required=True, help="出力先")
    ap.add_argument("--neg-cache", default=None,
                    help="negative フレーム PNG の永続キャッシュ（既定: <out>/../neg_cache）")
    ap.add_argument("--video-root", default=None,
                    help="manifest の video_path が解決できない時に basename を探すディレクトリ")
    ap.add_argument("--stride-seconds", type=float, default=30.0, help="ビン幅（秒）")
    ap.add_argument("--dense-classes", default="auto",
                    help="密に拾うクラス。カンマ区切りのクラス名 / "
                         "'auto'（--dense-auto-threshold 未満のビン数のクラス）/ "
                         "'all'（全クラス＝全フレームアノテーション区間を活かす）/ "
                         "'none'（無効）。既定 auto")
    ap.add_argument("--dense-max-per-video", type=int, default=None,
                    help="1 動画あたりの密フレーム数の上限（時間方向に均等間引き）。"
                         "★密アノテーションは一部の動画に極端に偏っており"
                         "（20260813 では 2 動画で全体の 51%）、無制限だと学習が"
                         "その数症例に支配される。'all' 指定時はほぼ必須")
    ap.add_argument("--dense-stride-seconds", type=float, default=1.0,
                    help="希少クラスのビン幅（秒）。既定 1 秒")
    ap.add_argument("--dense-auto-threshold", type=int, default=150,
                    help="--dense-classes auto で希少とみなす基準（30秒グリッドでのビン数）。"
                         "既定 150 は 20260813 の実データで clip 391 / sponge 485 と "
                         "それ以外（<=112）がきれいに分かれる値")
    ap.add_argument("--coverage", default="from_start",
                    choices=["from_start", "span", "full", "runs"],
                    help="陰性を張るビン範囲の決め方")
    ap.add_argument("--run-gap-bins", type=int, default=1,
                    help="--coverage runs で連続とみなすビン間隔の上限")
    ap.add_argument("--pick", default="center", choices=["center", "most_instances", "first"],
                    help="1 ビン内に複数 positive がある時の代表の選び方")
    ap.add_argument("--neg-phase", default="grid", choices=["grid", "center"],
                    help="negative フレームのビン内位置。grid = アノテータが実際に開いた"
                         "30 秒グリッドの位相に合わせる（既定・推奨）／center = ビン中央")
    ap.add_argument("--seeds-dir", default=None,
                    help="workspace/fo_annotation_seeds/json/<dataset> — 陰性の検算に使う")
    ap.add_argument("--seeds-exclude-conflicts", action="store_true",
                    help="seeds が「FO 可視」と言うビンを negative から外す（誤陰性の除去）")
    ap.add_argument("--degenerate-std", type=float, default=8.0,
                    help="輝度標準偏差がこれ未満のフレームを no-signal とみなす")
    ap.add_argument("--keep-degenerate", action="store_true",
                    help="no-signal フレームを除外せず degenerate フラグだけ付ける")
    ap.add_argument("--videos", nargs="*", default=None, help="対象動画の basename（既定: 全部）")
    ap.add_argument("--clean", action="store_true",
                    help="書き出し前に <out>/images の既存 symlink を消す。アノテーション追加で"
                         "代表フレームが変わると前回の symlink が孤児として残るため、"
                         "作り直しでは指定推奨（--neg-cache は消さない）")
    ap.add_argument("--link-prefix-map", default="/data4/=/mnt/data/data4/",
                    help="symlink 先のパス前置きを OLD=NEW で置換（カンマ区切りで複数可）。"
                         "既定は dl1 ローカルの /data4 を dl2 からも見える /mnt/data/data4 へ寄せる")
    ap.add_argument("--dry-run", action="store_true", help="計画だけ表示して書き込まない")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    out = Path(args.out)
    neg_cache = Path(args.neg_cache) if args.neg_cache else out.parent / "neg_cache"
    prefix_map = [
        (old, new)
        for entry in args.link_prefix_map.split(",") if entry.strip()
        for old, _, new in [entry.strip().partition("=")]
    ]

    # --- 複数 dump をまとめて読む（heico + lapchole）---
    # image_id / annotation_id は dump ごとに 1 始まりなので、(dump index, id) で
    # 一意化してから扱う。混ぜる前にラベル体系が同じことを確認する。
    dumps = [Path(d) for d in args.dump]
    sources: list[dict] = []
    label_maps: list[tuple[str, dict]] = []
    for di, d in enumerate(dumps):
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        cc = json.loads((d / "coco" / "instances.json").read_text(encoding="utf-8"))
        sources.append({"dump": d, "manifest": man, "coco": cc, "index": di})
        label_maps.append((str(d), man.get("explicit_labels", {})))
        logger.info("dump[%d] %s: %d videos, %d images, %d annotations",
                    di, d.name, len(man["videos"]), len(cc["images"]), len(cc["annotations"]))
    base_labels = label_maps[0][1]
    for name, lm in label_maps[1:]:
        if lm != base_labels:
            logger.error("ラベル体系が dump 間で不一致:\n  %s: %s\n  %s: %s",
                         label_maps[0][0], base_labels, name, lm)
            return 1

    catname: dict[int, str] = {}
    n_inst: dict[tuple[int, int], int] = {}
    ann_by_image: dict[tuple[int, int], list[dict]] = {}
    cats_by_image: dict[tuple[int, int], set[str]] = {}
    images_by_video: dict[tuple[int, str], list[dict]] = {}
    for s in sources:
        di, cc = s["index"], s["coco"]
        catname.update({c["id"]: c["name"] for c in cc["categories"]})
        for a in cc["annotations"]:
            key = (di, a["image_id"])
            ann_by_image.setdefault(key, []).append(a)
            n_inst[key] = n_inst.get(key, 0) + 1
            cats_by_image.setdefault(key, set()).add(catname[a["category_id"]])
        for im in cc["images"]:
            images_by_video.setdefault((di, im["video"]), []).append(im)

    # --- 希少クラスの決定 ---
    dense_classes: set[str] = set()
    if args.dense_classes.strip().lower() not in ("none", ""):
        if args.dense_classes.strip().lower() == "all":
            dense_classes = set(catname.values())
            logger.info("全クラスを密サンプリング対象にする（全フレームアノテーション区間を活用）")
        elif args.dense_classes.strip().lower() == "auto":
            # 30 秒グリッドで何ビンに現れるかを数え、閾値未満を希少とする
            bins_per_class: dict[str, set] = {}
            for s in sources:
                di = s["index"]
                fps_guess = 25.0 if "heico" in str(s["dump"]).lower() else 30.0
                bs = max(1, int(round(args.stride_seconds * fps_guess)))
                for im in s["coco"]["images"]:
                    for c in cats_by_image.get((di, im["id"]), set()):
                        bins_per_class.setdefault(c, set()).add(
                            (di, im["video"], im["frame_number"] // bs))
            dense_classes = {c for c, b in bins_per_class.items()
                             if len(b) < args.dense_auto_threshold}
            logger.info("希少クラス(auto, 30秒ビン数 < %d): %s",
                        args.dense_auto_threshold,
                        {c: len(bins_per_class[c]) for c in sorted(dense_classes)})
            logger.info("  非希少: %s", {c: len(b) for c, b in sorted(bins_per_class.items())
                                        if c not in dense_classes})
        else:
            dense_classes = {s.strip() for s in args.dense_classes.split(",") if s.strip()}
        if dense_classes:
            logger.info("希少クラスは %.1f 秒ビンでも拾う（他は %.1f 秒）",
                        args.dense_stride_seconds, args.stride_seconds)

    plans: list[VideoPlan] = []
    for s in sources:
        di, man = s["index"], s["manifest"]
        entries = man["videos"]
        if args.videos:
            wanted = {v.lower() for v in args.videos}
            entries = [e for e in entries if e["video"].lower() in wanted
                       or Path(e["video"]).stem.lower() in wanted]
        for entry in entries:
            # dump の "video" は拡張子付きの basename。COCO の image["video"] と同じキー。
            imgs = images_by_video.get((di, entry["video"]), [])
            plan = plan_video(
                entry, imgs, {k[1]: v for k, v in n_inst.items() if k[0] == di},
                args, s["dump"],
                {k[1]: v for k, v in cats_by_image.items() if k[0] == di}, dense_classes,
            )
            plan.source_index = di
            plans.append(plan)
        logger.info(
            "%s: %d frames @%.2ffps, bin=%d | dump pos %d → keep %d bins, neg %d bins "
            "(coverage %s bins %d-%d, neg phase %s)",
            plan.stem, plan.n_frames, plan.fps, plan.bin_size, len(imgs),
            len(plan.keep_pos), len(plan.neg_bins), args.coverage, *plan.coverage,
            plan.neg_phase if plan.neg_phase is not None else "center",
        )

    # --- seeds（QA 逆算の FO 可視イベント）で陰性を検算 ---
    # negative を実体化する前に走らせる: --seeds-exclude-conflicts で
    # 「FO が見えている証拠のあるビン」を陰性から外せるようにするため。
    audits: list[dict] = []
    if args.seeds_dir:
        seeds_dir = Path(args.seeds_dir)
        if not seeds_dir.is_absolute():
            seeds_dir = (Path(__file__).parent / seeds_dir).resolve()
        for plan in plans:
            a = audit_with_seeds(plan, seeds_dir)
            if not a:
                logger.warning("%s: seeds が見つからず検算をスキップ", plan.stem)
                continue
            audits.append(a)
            if args.seeds_exclude_conflicts and a["conflicts"]:
                drop = {c["bin"] for c in a["conflicts"]}
                plan.neg_bins = [b for b in plan.neg_bins if b not in drop]
                for b in drop:
                    plan.neg_frames.pop(b, None)
        n_conf = sum(a["n_conflict_bins"] for a in audits)
        n_ok = sum(a["n_confirmed_negative_bins"] for a in audits)
        logger.info(
            "seeds 検算: %d 動画 | 陰性裏付け %d bins / 矛盾 %d bins%s | "
            "初アノテーション − FO 初出現 (秒): %s",
            len(audits), n_ok, n_conf,
            "（除外した）" if args.seeds_exclude_conflicts else "（保持）",
            [a["first_annotated_minus_seed_sec"] for a in audits],
        )
        for a in audits:
            if a["n_conflict_bins"]:
                logger.warning("  %s: negative %d bins に FO 可視の証拠あり（先頭 %s）",
                               a["video"], a["n_conflict_bins"], a["conflicts"][:3])

    tot_pos = sum(len(p.keep_pos) for p in plans)
    tot_dense = sum(len(p.keep_dense) for p in plans)
    tot_neg = sum(len(p.neg_bins) for p in plans)
    logger.info(
        "plan: %d videos, positive %d (+希少クラス密 %d) , negative %d (計 %d) → %s",
        len(plans), tot_pos, tot_dense, tot_neg, tot_pos + tot_dense + tot_neg, out)
    if args.dry_run:
        return 0

    # --- negative の実体化 ---
    neg_paths: dict[str, dict[int, Path]] = {}
    for plan in plans:
        neg_paths[plan.stem] = materialize_negatives(plan, neg_cache)

    if args.clean:
        images_dir = out / "images"
        n_removed = 0
        for p in sorted(images_dir.rglob("*.png")) if images_dir.is_dir() else []:
            # symlink だけを消す。実体ファイルは触らない（neg-cache や dump の保護）
            if p.is_symlink():
                p.unlink()
                n_removed += 1
        if n_removed:
            logger.info("--clean: 既存 symlink %d 本を削除", n_removed)

    # --- COCO 組み立て ---
    out_images: list[dict] = []
    out_anns: list[dict] = []
    n_degenerate = 0
    rows: list[dict] = []

    for plan in plans:
        pos_by_frame = {im["frame_number"]: im
                        for im in images_by_video.get((plan.source_index, plan.video), [])}
        kept_pos = kept_neg = kept_dense = 0
        # 30 秒グリッドの陽性 / 希少クラスの密フレーム / 陰性 を frame 番号で統合する
        # （密フレームは 30 秒グリッド上に無いのでビン番号では並べられない）
        items: list[tuple[int, str, int | None]] = []
        for b, f in plan.keep_pos.items():
            items.append((f, "pos", b))
        for db, f in plan.keep_dense.items():
            if f not in plan.keep_pos.values():
                items.append((f, "dense", db))
        for b, f in plan.neg_frames.items():
            items.append((f, "neg", b))
        for frame_number, kind, b in sorted(items):
            is_pos = kind in ("pos", "dense")
            src = (
                plan.dump / pos_by_frame[frame_number]["file_name"]
                if is_pos
                else neg_paths[plan.stem][b]
            )
            degenerate = is_degenerate(src, args.degenerate_std)
            if degenerate:
                n_degenerate += 1
                if not args.keep_degenerate:
                    continue

            rel = f"images/{plan.stem}/{frame_number:06d}.png"
            link(src, out / rel, prefix_map)
            image_id = len(out_images) + 1
            src_im = pos_by_frame.get(frame_number)
            out_images.append({
                "id": image_id,
                "file_name": rel,
                "width": src_im["width"] if src_im else plan.width,
                "height": src_im["height"] if src_im else plan.height,
                "video": plan.video,
                "frame_number": frame_number,
                "bin": b,
                "is_negative": not is_pos,
                "sampling": kind,          # pos = 30秒グリッド / dense = 希少クラス密 / neg
                "degenerate": degenerate,
                "source": str(src),
                "source_dump": str(plan.dump),
            })
            if is_pos:
                kept_pos += kind == "pos"
                kept_dense += kind == "dense"
                for a in ann_by_image.get((plan.source_index, src_im["id"]), []):
                    out_anns.append({**a, "id": len(out_anns) + 1, "image_id": image_id})
            else:
                kept_neg += 1

        rows.append({
            "video": plan.video,
            "stem": plan.stem,
            "fps": round(plan.fps, 3),
            "n_frames": plan.n_frames,
            "width": plan.width,
            "height": plan.height,
            "bin_size": plan.bin_size,
            "coverage_bin_lo": plan.coverage[0],
            "coverage_bin_hi": plan.coverage[1],
            "neg_phase": plan.neg_phase if plan.neg_phase is not None else plan.bin_size // 2,
            "neg_phase_inferred": plan.neg_phase is not None,
            "n_dump_images": len(pos_by_frame),
            "n_positive": kept_pos,
            "n_dense": kept_dense,
            "n_negative": kept_neg,
            "n_dropped_bins": len(plan.dropped_bins),
            "source_dump": str(plan.dump),
        })

    doc = {
        "info": {
            "description": (
                f"FO instance segmentation — {args.stride_seconds}s grid + negatives"
                + (f" + 希少クラス {args.dense_stride_seconds}s ({','.join(sorted(dense_classes))})"
                   if dense_classes else "")
            ),
            "tool": "build_dataset.py/2",
            "source_dumps": [str(d) for d in dumps],
            "source_xml_git_commits": [s["manifest"].get("xml_git_commit") for s in sources],
        },
        "images": out_images,
        "annotations": out_anns,
        # ラベル体系が同一であることは読み込み時に検証済みなので先頭 dump のものを使う
        "categories": sources[0]["coco"]["categories"],
    }
    (out / "coco").mkdir(parents=True, exist_ok=True)
    (out / "coco" / "instances.json").write_text(
        json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    with (out / "videos.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    if audits:
        (out / "audit_seeds.json").write_text(
            json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    build_manifest = {
        "tool": "build_dataset.py/2",
        "dumps": [{"path": str(s["dump"]),
                   "xml_git_commit": s["manifest"].get("xml_git_commit"),
                   "generated_at": s["manifest"].get("generated_at")} for s in sources],
        "neg_cache": str(neg_cache),
        "params": vars(args),
        "dense_classes": sorted(dense_classes),
        "n_images": len(out_images),
        "n_positive": sum(1 for im in out_images if not im["is_negative"]),
        "n_dense": sum(1 for im in out_images if im.get("sampling") == "dense"),
        "n_negative": sum(1 for im in out_images if im["is_negative"]),
        "n_annotations": len(out_anns),
        "n_degenerate_found": n_degenerate,
        "degenerate_dropped": not args.keep_degenerate,
        "videos": rows,
    }
    (out / "build_manifest.json").write_text(
        json.dumps(build_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        "done: images %d (pos %d / neg %d), annotations %d, no-signal %d 枚を%s → %s",
        len(out_images), build_manifest["n_positive"], build_manifest["n_negative"],
        len(out_anns), n_degenerate,
        "保持" if args.keep_degenerate else "除外", out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
