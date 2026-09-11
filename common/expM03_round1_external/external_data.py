"""外部 Surgical VQA（expM01 MultiBypass / expM02 SurgMLLMBench）を train へ注入するローダ.

config 例（train_lora_ext.py の `data.external`）:

    external:
      parquets:
        - workspace/expM02_surgmllmbench/out/surgmllmbench_qa_v1.parquet
        - workspace/expM01_multibypass_qa/out/multibypass_qa_v1.parquet
      size: 768                  # target_size（読み込み時に縮小）
      loss_weight: 0.5           # ★「事前学習的」利用: FOCUS(1.0) より弱く効かせる
      procedure_dropout: 0.5     # ★質問先頭の術式文を確率で落とす（OOD: 術式名が無くても
                                 #   視覚だけで答えられ、あれば使う、の両方を教える）
      track_limits:              # track ごとの投入上限（省略時は全量）
        FRAME: null
        SEGMENT: null

設計（pseudo_data.py に準拠、ただし差分あり）:
- **paraphrase_bank は通さない**（外部の質問文は FOCUS 正準テンプレートに一致せず壊れるため）
- system prompt は parquet の answer_format から直接組む（detect_format に頼らない）
- 術式文ドロップアウトは行ごとに決定的（seed 固定）— 再実行で同じデータになる
- フレーム欠落は数えて警告し、**既定では1件でも失敗**（allow_missing: true で緩和。
  smoke を通すために壊れた経路を隠さない）
- val へ入れる経路は無い（train 専用）
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from dataset_seg import ROOT, VideoVQASample
from prompts_seg import FORMAT_RULES, LOOK_CAREFULLY, PERSONA, _fo_class_rule

log = logging.getLogger("expM03.external")

# parquet の frame_paths は dl1 の絶対パス（/data4/src/shunsuke/MICCAI2026/Orena/...）で
# 記録されている。dl2 は /data4 が無く、vast はリポジトリ位置が異なるため、
# **実行時のリポジトリルート（ROOT）基準に読み替える**（dl1 では同一パスになる）。
_DL1_PREFIX = "/data4/src/shunsuke/MICCAI2026/Orena/"


def _portable(p: str) -> Path:
    if p.startswith(_DL1_PREFIX):
        return ROOT / p[len(_DL1_PREFIX):]
    return Path(p)

# 質問先頭の術式文（expM01/expM02 の生成器が付けたもの）にだけマッチさせる
_PROC_SENT = re.compile(
    r"^This (?:image|video) (?:is from|shows) a[^.]*\.\s*", re.IGNORECASE)


def _system_prompt(fmt: str) -> str:
    rule = _fo_class_rule() if fmt == "fo_class" else FORMAT_RULES[fmt]
    return "\n\n".join([PERSONA, LOOK_CAREFULLY, rule])


def build_external_samples(ecfg: dict, seed: int = 42,
                           limit_cap: int | None = None) -> list[VideoVQASample]:
    import pandas as pd
    rng = random.Random(seed + 7)
    size = int(ecfg.get("size", 768))
    weight = float(ecfg.get("loss_weight", 0.5))
    pdrop = float(ecfg.get("procedure_dropout", 0.5))

    frames = []
    for pq in ecfg["parquets"]:
        p = Path(pq)
        if not p.is_absolute():
            p = ROOT / p
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)

    tl = ecfg.get("track_limits") or {}
    if tl:
        parts = []
        for trk, g in df.groupby("track"):
            cap = tl.get(str(trk))
            if cap is not None and len(g) > int(cap):
                g = g.sample(n=int(cap), random_state=seed)
            parts.append(g)
        df = pd.concat(parts, ignore_index=True)
    if limit_cap is not None and len(df) > int(limit_cap):
        # smoke: 生成元(generation)の比率を保って絞る
        parts = [g.head(max(1, int(int(limit_cap) * len(g) / len(df))))
                 for _, g in df.groupby("generation")]
        df = pd.concat(parts, ignore_index=True)

    out: list[VideoVQASample] = []
    n_missing = 0
    n_dropped_proc = 0
    for r in df.itertuples():
        paths = [_portable(x) for x in json.loads(r.frame_paths)]
        if not all(p.exists() for p in paths):
            n_missing += 1
            continue
        times = [float(x) for x in json.loads(r.frame_times)]
        if not times:
            # FRAME 外部（SurgMLLMBench）は時刻なし。ダミーの0秒は入れず空のまま扱えないので
            # 1点の 0.0 を与える（build_messages は frame_times と frame_paths を zip する）
            times = [0.0] * len(paths)
        q = str(r.question)
        if pdrop > 0 and rng.random() < pdrop:
            q2 = _PROC_SENT.sub("", q, count=1)
            if q2 != q:
                n_dropped_proc += 1
                q = q2
        fmt = str(r.answer_format)
        out.append(VideoVQASample(
            uid=f"ext:{r.id}", qID=str(r.id), dataset=str(r.dataset),
            videoID=str(r.video), track=str(r.track),
            start_time=float(r.start_time), end_time=float(r.end_time),
            frame_times=times, frame_paths=paths,
            target_size=size, system_prompt=_system_prompt(fmt), question=q,
            answer=str(r.answer), fmt=fmt,
            primary=str(r.primary_capability), group="EXTERNAL",
            meta={"loss_weight": weight}))

    if n_missing:
        msg = f"★外部フレーム欠落 {n_missing} 件（DL/抽出が未完の可能性）"
        if ecfg.get("allow_missing"):
            log.warning(msg + " — allow_missing により続行")
        else:
            raise FileNotFoundError(msg + "。抽出完了を待つか allow_missing: true を指定")
    from collections import Counter
    log.info(f"external samples {len(out)} 件 @{size}px weight={weight} "
             f"proc_dropout={pdrop}（実際に落とした {n_dropped_proc} 件） "
             f"内訳 {dict(Counter(s.dataset for s in out))} "
             f"track {dict(Counter(s.track for s in out))}")
    return out
