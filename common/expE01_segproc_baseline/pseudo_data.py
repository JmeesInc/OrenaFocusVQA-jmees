"""擬似 VQA（expK00 生成物）を学習セットへ注入するローダ.

config 例（train_lora_seg.py の `data.pseudo`）:

    pseudo:
      parquets:
        - workspace/expK00_pseudo_vqa/out/pseudo_frame_v1_packed.parquet
        - workspace/expK00_pseudo_vqa/out/pseudo_frame_cutpaste_v1_packed.parquet
      frames_root: workspace/expK00_pseudo_vqa/out/packed   # 相対パスは ROOT 基準
      size: 768                # target_size（packed 768px から読み込み時縮小）
      paraphrase_prob: 0.5     # 質問文の言い換え率（正準文からパラメータを回収して再充填）
      limit: null              # smoke では train_lora_seg が --train-limit を上限として適用
      track_limits:            # track ごとの投入上限（比率調整用, expK03）
        FRAME: 3744

設計:
- 生成 parquet は **qa fold v003 fold0（val）動画を含まない**（expK00 側で除外済み）ので
  train にしか使わない。val へは絶対に入れない。
- fmt は `build_system_prompt`（detect_format）に任せ、parquet の answer_format と
  食い違う行は**数えてログに出す**（多発したらパラフレーズ銀行の文面を直す）。
- `group` は "PSEUDO" 固定（学習ログ上で実データと区別するため。評価には使われない）。
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

from dataset_seg import ROOT, VideoVQASample
from prompts_seg import build_system_prompt

sys.path.insert(0, str(ROOT / "workspace/expK00_pseudo_vqa"))
from paraphrase_bank import rephrase  # noqa: E402

log = logging.getLogger("expE01.pseudo")

_PROCEDURE_TYPE = {
    "Prokto": "Proctocolectomy",
    "Rektum": "Rectal Resection",
    "Sigma": "Sigmoid Resection",
}


def _procedure_type(dataset: str, video: str) -> str:
    if dataset == "lapchole":
        return "Laparoscopic Cholecystectomy"
    return next((v for k, v in _PROCEDURE_TYPE.items() if k in video), "")


def build_pseudo_samples(pcfg: dict, seed: int = 42,
                         limit_cap: int | None = None) -> list[VideoVQASample]:
    import pandas as pd
    rng = random.Random(seed)
    root = Path(pcfg["frames_root"])
    if not root.is_absolute():
        root = ROOT / root
    size = int(pcfg.get("size", 768))
    pprob = float(pcfg.get("paraphrase_prob", 0.5))
    limit = pcfg.get("limit")
    if limit_cap is not None:
        limit = limit_cap if limit is None else min(int(limit), limit_cap)

    # ★`parquets` の各要素は **文字列**（重み 1.0）でも
    #   **{path: ..., loss_weight: x}** でもよい（expM00-E, 2026-09-02）。
    #   狙い: 擬似 v2 は v1 より件数が多い（11,433 vs 7,668）ぶん勾配への寄与が過大になる。
    #   件数を削って v1 に合わせる代わりに、**重みで v1 相当の寄与に揃える**と
    #   v2 の網羅性（unlabeled 68本を含む 104 動画）を保ったまま影響度だけ揃えられる。
    #   ⚠️ 重み 1.0 のみの構成は **既存実験と数値一致**（WeightedTrainer が親経路に落ちる）。
    frames = []
    for pq in pcfg["parquets"]:
        w = 1.0
        if isinstance(pq, dict):
            pq, w = pq["path"], float(pq.get("loss_weight", 1.0))
        p = Path(pq)
        if not p.is_absolute():
            p = ROOT / p
        f = pd.read_parquet(p)
        f["_loss_weight"] = w
        if w != 1.0:
            log.info("pseudo %s: %d 件 × loss_weight %.4f", p.name, len(f), w)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    # ★track ごとの上限（expK03）。K01S の敗因は「擬似が FRAME 形式だけで
    #   SEGMENT:FRAME の比が 1:1 → 1:1.76 に崩れた」ことなので、比を保つために
    #   FRAME 擬似の投入数を明示的に絞れるようにする。
    # ★★形式構成比の調整（expK04, 2026-08-28）。K03 の敗因は **件数比 1:1 は揃えたのに
    #   answer_format の構成比が実データとズレていた**こと:
    #     FRAME 擬似は number が 44%（実 FRAME は 31%）/ SEGMENT 擬似は number が 0%（実 7%）
    #   → number が −0.0685** と壊れ AGGREGATION バケットが −0.0678*** に崩壊した。
    #   `format_mix` に track ごとの目標比率を書くと、**超過している形式をダウンサンプル**して
    #   その比率に寄せる（不足は増やせないので、他形式を削って相対比を合わせる）。
    fm = pcfg.get("format_mix") or {}
    if fm:
        parts = []
        for trk, g in df.groupby(df.get("track", "FRAME")):
            target = fm.get(str(trk))
            if not target:
                parts.append(g)
                continue
            # 各形式の「その比率を満たすのに許される総数」の最小値を全体サイズとする
            cap_total = min(int(len(g[g["answer_format"] == f]) / r)
                            for f, r in target.items()
                            if r > 0 and (g["answer_format"] == f).any())
            keep = []
            for f, sub in g.groupby("answer_format"):
                r = target.get(f)
                n = int(round(cap_total * r)) if r is not None else len(sub)
                keep.append(sub.sample(n=min(n, len(sub)), random_state=seed))
            g2 = pd.concat(keep, ignore_index=True)
            log.info(f"format_mix[{trk}]: {len(g)} → {len(g2)} / "
                     + " ".join(f"{k}={v/len(g2)*100:.0f}%" for k, v in
                                g2['answer_format'].value_counts().items()))
            parts.append(g2)
        df = pd.concat(parts, ignore_index=True)

    tl = pcfg.get("track_limits") or {}
    if tl:
        parts = []
        for trk, g in df.groupby(df.get("track", "FRAME")):
            cap = tl.get(str(trk))
            if cap is not None and len(g) > int(cap):
                g = g.sample(n=int(cap), random_state=seed)
            parts.append(g)
        df = pd.concat(parts, ignore_index=True)
        log.info("track_limits 適用: "
                 + " / ".join(f"{k}={v}" for k, v in
                              df.groupby(df.get("track", "FRAME")).size().items()))
    if limit is not None:
        # ソース比率を保ったまま先頭から絞る（smoke 用。シャッフルは呼び出し側で行う）
        # ※groupby.apply は pandas の版差（include_groups）で挙動が割れるので使わない
        n_all = len(df)
        parts = [g.head(max(1, int(int(limit) * len(g) / n_all)))
                 for _, g in df.groupby("generation")]
        df = pd.concat(parts, ignore_index=True)

    out: list[VideoVQASample] = []
    n_fmt_mismatch = 0
    n_missing = 0
    for r in df.itertuples():
        # ★SEGMENT 擬似は 1問が複数フレーム（`frame_paths`/`frame_times` が JSON 配列）。
        #   FRAME 擬似は単一（`frame_path`/`timestamp`）。両方をここで吸収する。
        if getattr(r, "track", "FRAME") == "SEGMENT":
            import json as _json
            paths = [root / x for x in _json.loads(r.frame_paths)]
            times = [float(x) for x in _json.loads(r.frame_times)]
            if not all(fp.exists() for fp in paths):
                n_missing += 1
                continue
            t0, t1, track = float(r.start_time), float(r.end_time), "SEGMENT"
        else:
            fp = root / r.frame_path
            if not fp.exists():
                n_missing += 1
                continue
            paths, times = [fp], [float(r.timestamp)]
            t0 = t1 = float(r.timestamp)
            track = "FRAME"
        q = rephrase(r.family, r.question, rng, pprob)
        sysp, fmt = build_system_prompt(q)
        if fmt != r.answer_format:
            n_fmt_mismatch += 1
        out.append(VideoVQASample(
            uid=f"pseudo:{r.id}", qID=str(r.id), dataset=r.dataset,
            videoID=r.video, track=track,
            start_time=t0, end_time=t1, frame_times=times, frame_paths=paths,
            target_size=size, system_prompt=sysp, question=q,
            answer=str(r.answer), fmt=fmt,
            procedure_type=_procedure_type(r.dataset, r.video),
            primary=str(r.primary_capability), group="PSEUDO",
            meta={"loss_weight": float(getattr(r, "_loss_weight", 1.0))}))
    if n_missing:
        log.warning(f"★擬似フレーム欠落 {n_missing} 件（frames_root={root} を確認）")
    if n_fmt_mismatch:
        log.warning(f"★detect_format と answer_format の不一致 {n_fmt_mismatch}/{len(out)} 件")
    from collections import Counter
    log.info(f"pseudo samples {len(out)} 件 @{size}px paraphrase_prob={pprob}  "
             f"内訳 {dict(Counter(s.qID.split('_')[0][:3] for s in out))}")
    return out
