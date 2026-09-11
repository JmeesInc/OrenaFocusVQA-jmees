"""擬似 SEGMENT の教師品質を2通りで実測する.

## (A) 公式 GT との一致率（gold standard）

公式 SEGMENT 問題のうち、**そのクリップ区間 [start,end] を我々の注釈イベントが覆っている**もので、
同じ family の答えを注釈から導出し、公式正解と突き合わせる。
FRAME 擬似でやったのと同じ検算（あちらは計数系 0.70 / 列挙 0.69 だった）。

## (B) 疎化バイアス（我々の近似がどれだけ効くか）

擬似クリップは「注釈イベント = 渡すフレーム」なので**証拠の上では答えは厳密**だが、
イベント間（実測 median ~40s）に真の出現/消失があると**公式 GT とはズレる**。
そこで **密に注釈された区間**（イベント間隔 ~2s）を真値とみなし、
そこから ~40s 間隔に間引いた場合に答えがどう変わるかを測る。
= 「疎な証拠で作った教師が、密な真実からどれだけ外れるか」の直接測定。
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import generate_frame_pseudo_qa as G
import generate_segment_pseudo_qa as S

HERE = Path(__file__).parent
LAP = ("/home/shunsuke/.cache/huggingface/hub/datasets--orena-dkfz--lapchole-focus-vqa/"
       "snapshots/5b3510cde3ba1135c56c4b4b25b50c4f948e235b/data/segment/train.parquet")
HEI = ("/home/shunsuke/.cache/huggingface/hub/datasets--orena-dkfz--heico-focus-vqa/"
       "snapshots/4ee0e4b39ee59006b773beec501bb47e251827eb/data/segment/train.parquet")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("expK00.valseg")


def to_sec(t: str) -> int:
    h, m, s = str(t).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def official_family(q: str) -> str | None:
    if q.startswith("Which foreign object classes appear in this video"):
        return "classes_in_video"
    if q.startswith("What types of foreign objects are seen between"):
        return "classes_between"
    if q.startswith("What surgical foreign object is visible in this video"):
        return "single_in_video"
    if "unique class of foreign object appearing in the video" in q:
        return "nth_unique"
    if q.startswith("At what time was a") and "last visible in the video" in q:
        return "last_visible"
    if q.startswith("At what time was a") and "first visible in the video" in q:
        return "first_visible"
    if "quadrants of the video have not been populated" in q:
        return "quad_not_populated"
    if "quadrants of the video have been populated" in q:
        return "quad_populated"
    if "first appears in this video, in which quadrant" in q:
        return "first_quad"
    if q.startswith("The last time a") and "where was the center" in q:
        return "last_quad"
    return None


def cls_from_q(q: str) -> str | None:
    for c in sorted(G.PLURAL, key=len, reverse=True):
        if c.lower() in q.lower():
            return c
    return None


def derive(fam: str, per_t: list, q: str) -> str | None:
    """公式の質問文 + 我々のイベント列から答えを導く（生成器と同じ規則）。"""
    classes = sorted({c for _, d, _, _ in per_t for c in d})
    if fam in ("classes_in_video", "single_in_video"):
        if not classes:
            return "none"
        if fam == "single_in_video" and len(classes) != 1:
            return None
        return ", ".join(classes)
    if fam == "classes_between":
        import re
        m = re.findall(r"(\d{2}:\d{2}:\d{2})", q)
        if len(m) < 2:
            return None
        t0, t1 = to_sec(m[0]), to_sec(m[1])
        sub = sorted({c for t, d, _, _ in per_t if t0 <= t <= t1 for c in d})
        return ", ".join(sub) if sub else "none"
    if fam == "nth_unique":
        import re
        m = re.search(r"(\d+)(st|nd|rd|th) unique", q)
        if not m:
            return None
        n = int(m.group(1))
        first_t = {c: min(t for t, d, _, _ in per_t if c in d) for c in classes}
        order = sorted(classes, key=lambda c: first_t[c])
        return order[n - 1] if n <= len(order) else "none"
    if fam in ("last_visible", "first_visible", "first_quad", "last_quad"):
        cls = cls_from_q(q)
        if cls is None or cls not in classes:
            return None
        if fam in ("last_visible", "first_visible"):
            # ★生成器と同じ境界検証フィルタを通す（保証できない問は出題しない）
            tol = S.time_tol(per_t[-1][0] - per_t[0][0])
            kind = "last" if fam == "last_visible" else "first"
            t = S.safe_boundary_time(per_t, {cls}, kind, tol)
            return G.hhmmss(t) if t is not None else None
        ts = [t for t, d, _, _ in per_t if cls in d]
        t = max(ts) if fam == "last_quad" else min(ts)
        d, w, h = next((d, w, h) for tt, d, w, h in per_t if tt == t and cls in d)
        if len(d[cls]) != 1:
            return None
        return S.quad_of(d[cls][0], w, h)
    if fam in ("quad_populated", "quad_not_populated"):
        quads = sorted({S.quad_of(i, w, h) for _, d, w, h in per_t for v in d.values() for i in v},
                       key=S.QUAD_ORDER.index)
        if fam == "quad_populated":
            return ", ".join(quads) if quads else "none"
        miss = [q2 for q2 in S.QUAD_ORDER if q2 not in quads]
        return ", ".join(miss) if miss else "none"
    return None


def norm(x: str) -> str:
    return ", ".join(sorted(p.strip().lower() for p in str(x).split(",") if p.strip()))


def per_t_of(events: list) -> list:
    out = []
    for e in events:
        d = defaultdict(list)
        for i in e["insts"]:
            d[i["cls"]].append(i)
        out.append((e["t"], d, e["w"], e["h"]))
    return out


def part_a(store: dict) -> None:
    log.info("=== (A) 公式 GT との一致率 ===")
    dfs = []
    for p, ds in ((LAP, "lapchole"), (HEI, "heico")):
        try:
            d = pd.read_parquet(p)
            d["ds"] = ds
            dfs.append(d)
        except Exception as e:
            log.warning(f"{ds}: {e}")
    off = pd.concat(dfs, ignore_index=True)
    idx = {k: per_t_of(v["events"]) for k, v in store.items()}
    agree, tot = Counter(), Counter()
    time_err = defaultdict(list)
    n_cover = 0
    for r in off.itertuples():
        stem = Path(str(r.video)).stem
        pt = idx.get(stem)
        if pt is None:
            continue
        s, e = to_sec(r.timestamp_start), to_sec(r.timestamp_end)
        inside = [x for x in pt if s <= x[0] <= e]
        # クリップを我々のイベントが十分覆っている問だけ評価する
        if len(inside) < 3 or inside[0][0] - s > 60 or e - inside[-1][0] > 60:
            continue
        fam = official_family(str(r.question))
        if fam is None:
            continue
        # ★時刻系は「窓端＝クリップ端」でないと生成器の条件Aが成立しない。
        #   生成時は窓端＝最終イベントなので、検証も端が許容内で揃う問に限る
        if fam in ("last_visible", "first_visible"):
            tol_edge = S.time_tol(e - s)
            if (inside[0][0] - s) > tol_edge or (e - inside[-1][0]) > tol_edge:
                continue
        n_cover += 1
        mine = derive(fam, inside, str(r.question))
        if mine is None:
            continue
        tot[fam] += 1
        if norm(mine) == norm(r.answer):
            agree[fam] += 1
        elif fam in ("last_visible", "first_visible"):
            try:
                time_err[fam].append(to_sec(mine) - to_sec(r.answer))
            except Exception:
                pass
    log.info(f"被覆した公式問: {n_cover}")
    for f in sorted(tot, key=lambda k: -tot[k]):
        log.info(f"  {f:20s} {agree[f]:4d}/{tot[f]:4d} = {agree[f]/tot[f]:.3f}")
    for f, errs in time_err.items():
        a = np.abs(errs)
        log.info(f"  {f} 誤差(不一致のみ n={len(errs)}): median |{np.median(a):.0f}|s "
                 f"p90 {np.percentile(a,90):.0f}s / 符号中央 {np.median(errs):+.0f}s")


def part_b(store: dict, rng: random.Random) -> None:
    log.info("=== (B) 疎化バイアス（密区間を真値として間引く） ===")
    DENSE_MAX_GAP, SPARSE_STEP = 5.0, 40.0
    agree, tot = Counter(), Counter()
    time_err = defaultdict(list)
    n_win = 0
    for stem, v in store.items():
        evs = v["events"]
        i = 0
        while i < len(evs):
            j = i
            while (j + 1 < len(evs) and evs[j + 1]["t"] - evs[j]["t"] <= DENSE_MAX_GAP
                   and evs[j + 1]["t"] - evs[i]["t"] <= S.MAX_CLIP_S):
                j += 1
            span = evs[j]["t"] - evs[i]["t"]
            if j - i + 1 >= 12 and span >= 120:            # 密で十分長い区間だけ
                dense = evs[i:j + 1]
                keep, last = [], -1e9
                for e in dense:                            # ~40s 間隔に間引く
                    if e["t"] - last >= SPARSE_STEP:
                        keep.append(e)
                        last = e["t"]
                if len(keep) >= S.MIN_EVENTS:
                    n_win += 1
                    pt_d, pt_s = per_t_of(dense), per_t_of(keep)
                    for fam in ("classes_in_video", "last_visible", "first_visible",
                                "quad_populated", "quad_not_populated", "nth_unique"):
                        cls = rng.choice(sorted({c for _, d, _, _ in pt_s for c in d}))
                        q = {"last_visible": S.T_LAST_VISIBLE.format(cls=cls),
                             "first_visible": S.T_FIRST_VISIBLE.format(cls=cls),
                             "first_quad": S.T_FIRST_QUAD.format(cls=cls),
                             "last_quad": S.T_LAST_QUAD.format(cls=cls),
                             "nth_unique": S.T_NTH_UNIQUE.format(n=1, suf="st"),
                             }.get(fam, "")
                        a_d, a_s = derive(fam, pt_d, q), derive(fam, pt_s, q)
                        if a_d is None or a_s is None:
                            continue
                        tot[fam] += 1
                        if norm(a_d) == norm(a_s):
                            agree[fam] += 1
                        elif fam in ("last_visible", "first_visible"):
                            time_err[fam].append(to_sec(a_s) - to_sec(a_d))
            i = j + 1
    log.info(f"密窓 {n_win}")
    for f in sorted(tot, key=lambda k: -tot[k]):
        log.info(f"  {f:20s} {agree[f]:4d}/{tot[f]:4d} = {agree[f]/tot[f]:.3f}")
    for f, errs in time_err.items():
        a = np.abs(errs)
        log.info(f"  {f} 疎化誤差(不一致のみ n={len(errs)}): median |{np.median(a):.0f}|s "
                 f"p90 {np.percentile(a,90):.0f}s")


def main() -> None:
    store = S.load_events()
    part_a(store)
    part_b(store, random.Random(0))


if __name__ == "__main__":
    main()
