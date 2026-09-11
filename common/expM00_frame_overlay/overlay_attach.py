"""事前生成した重畳キャッシュを `VideoVQASample` に貼る（**学習と推論で共用**）.

`render_cache.py` が作った `cache/<variant>/index.json` を読み、arm 表に従って
`sample.overlay_path` / `sample.overlay_note` を埋める。

規則（学習・CV・コンテナで同一）:
  1. arm が CONTROL の動画 → **重畳を付けない**（検出器がその動画を学習しているため）
  2. 検出0件（index の `file` が null）→ **重畳を付けない**（空の重畳は expG00 §39 で −0.0429）
  3. `t1=True` のときだけ、説明文の後ろに**個数だけ**のテキストを足す
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("expM00.attach")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_index(cache_root: str | Path, variant: str) -> dict:
    """`index*.json` を**全部マージ**して返す（part 別・shard 別に分かれている）。

    ⚠️ 旧実装は `index.json` があればそれだけを返していた。2026-09-01 に
    val と train の同時レンダが同じ `index.json` を奪い合って片方が消え、しかも
    「1本だけ読む」実装がそれを正常な index として受け入れてしまった。
    → **常に全部マージする**（欠けは呼び出し側の `missing` カウントで検出する）。
    """
    d = Path(cache_root) / variant
    shards = sorted(d.glob("index*.json"))
    if not shards:
        raise SystemExit(f"index が無い: {d}")
    merged: dict = {}
    for s in shards:
        merged.update(json.loads(s.read_text()))
    # ★**書き出さない**。レンダ途中の shard を index.json として固めてしまうと、
    #   以降の run が「欠けたキャッシュ」を正しいものとして読む事故になる。
    log.info("shard %d 本を（メモリ上で）マージ: %d frames", len(shards), len(merged))
    return merged


def arm_of_video(arm_csv: str | Path = HERE / "overlay_arm_v001.csv") -> dict[str, str]:
    import pandas as pd
    df = pd.read_csv(arm_csv)
    return {os.path.splitext(v)[0].strip(): a for v, a in zip(df.videoID, df.arm)}


def frame_key(p) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def attach(samples, cache_root: str | Path, variant: str,
           arms: dict[str, str] | None = None,
           dual_arms: tuple[str, ...] = ("DUAL",),
           t1: bool = False, strict: bool = True) -> dict:
    """samples を破壊的に更新して統計を返す。

    `arms=None` なら arm による除外をしない（**val 用**。qa fold0 には
    det-train 動画が 1 本も無いので全問 DUAL 扱いでよい）。
    """
    idx = load_index(cache_root, variant)
    root = Path(cache_root)
    st = {"n": 0, "dual": 0, "control_arm": 0, "empty": 0, "missing": 0}
    for s in samples:
        st["n"] += 1
        if arms is not None:
            vid = s.videoID.rsplit(".", 1)[0].strip()
            if arms.get(vid, "DUAL") not in dual_arms:
                st["control_arm"] += 1
                continue
        rec = idx.get(frame_key(s.frame_paths[0]))
        if rec is None:
            st["missing"] += 1
            if strict:
                raise SystemExit(f"重畳キャッシュに無いフレーム: {s.frame_paths[0]}")
            continue
        if not rec.get("file"):
            st["empty"] += 1                      # ★検出0件 → 単画像のまま
            continue
        s.overlay_path = str(root / rec["file"])
        note = rec["note"]
        if t1 and rec.get("count_text"):
            note = note + " " + rec["count_text"]
        s.overlay_note = note
        st["dual"] += 1
    log.info("overlay(%s): %d 問中 dual %d / 空 %d / CONTROL arm %d / 欠け %d",
             variant, st["n"], st["dual"], st["empty"], st["control_arm"], st["missing"])
    return st
