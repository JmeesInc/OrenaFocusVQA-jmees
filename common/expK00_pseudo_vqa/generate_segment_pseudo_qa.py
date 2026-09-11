"""SEGMENT 擬似 VQA 生成 — FRAME 注釈だけから作る（SAM3 は使わない）.

## ★設計の核: 「クリップの証拠 = 我々が渡すフレーム」にする

SEGMENT の質問は「いつ現れたか／いつ消えたか／区間に何が居たか」を問うので時間軸が要る。
一見 SAM3 の密トラックが必要に見えるが、**実測すると不要**だった:

- SAM3 は GT シード ±15s しか埋めず、**シードされた個体しか追わない**（新規流入が見えない）。
  score ゲートで早期に切れる個体もあり、「消失時刻」の教師としては信用できない。
- 一方 **GT 注釈フレームはその瞬間の FO 集合が完全**（アノテータは全 FO を塗る）。

そこで **擬似クリップの入力フレームを「注釈フレームそのもの」に限定**する。
するとモデルが見る証拠と我々が知っている事実が**完全に一致**し、
「最後に見えた時刻 = 見えている最後のフレーム」が**証拠の上で厳密に正しい**答えになる。
これは推論時にモデルが取るべき方針そのものでもある（見えないものは答えられない）。

⚠️ 残る近似: 注釈イベント間（実測 median ~40s 間隔）に真の出現/消失があっても我々は知らない。
   `validate_segment_sparse_bias.py` が**密注釈区間（連続フレーム）を使って、この疎化バイアスを
   実測**する（疎な部分集合から出した答え vs 密な真値）。

## 生成単位

窓 = 同一動画の注釈イベント列から取った **duration <= 300s（SEGMENT 上限）** の連続部分列。
- イベント = 連続フレームのバースト（<=2s 間隔）を1点に畳んだもの（同じ瞬間を重複させない）
- 窓には **>= MIN_EVENTS 個**のイベントが要る（時間質問が自明にならないため）
- 入力フレームは窓内の全イベント（上限 MAX_FRAMES、超えたら等間隔に間引き）

## リーク

qa fold v003 fold0（VQA val）の動画は使わない（FRAME 擬似と同じ）。
"""
from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import generate_frame_pseudo_qa as G

HERE = Path(__file__).parent
OUT = HERE / "out"

BURST_GAP_S = 2.0        # これ以下の間隔は「同じ瞬間」として畳む
MAX_CLIP_S = 300.0       # SEGMENT の上限（5分）
MIN_CLIP_S = 25.0
MIN_EVENTS = 5
# ★実データの SEGMENT は 16f@448（≈2,900 tokens）。24枚だと ≈4,350 tokens で
#   `max_seq_len: 4096` を超え、fla の backward が 4090(24GB) で OOM した（2026-08-27 実害）。
#   **実データと同じ 16 枚**に揃える。答えは間引き後のフレーム集合から導出し直されるので
#   （`facts()` は窓の採用フレームだけを見る）、証拠と答えの一致は保たれる。
MAX_FRAMES = 16
MAX_Q_PER_WINDOW = 5
WINDOW_STRIDE = 1      # 窓の開始をイベント何個ずつずらすか
TARGET_TOTAL = 9000
SEED = 20260826

QUAD_ORDER = G.QUAD_ORDER
PLURAL = G.PLURAL

# 公式 lapchole SEGMENT train のテンプレ（byte 一致で採取, 2026-08-26 確認）
T_CLASSES_IN_VIDEO = ("Which foreign object classes appear in this video? "
                      "Please provide the class name(s) or answer with none.")
T_SINGLE_IN_VIDEO = ("What surgical foreign object is visible in this video? "
                     "Please provide a class name or answer with none.")
T_CLASSES_BETWEEN = ("What types of foreign objects are seen between {t0} and {t1}? "
                     "Please provide the class name(s) or answer none.")
T_NTH_UNIQUE = ("What is the {n}{suf} unique class of foreign object appearing in the video? "
                "Please provide a class name or answer with none.")
T_LAST_VISIBLE = ("At what time was a {cls} last visible in the video? "
                  "Please provide an answer in the format hh:mm:ss.")
T_FIRST_VISIBLE = ("At what time was a {cls} first visible in the video? "
                   "Please provide an answer in the format hh:mm:ss.")
T_PAIR_LAST = ("At what time were a {a} and a {b} last visible at the same time in this video? "
               "Please provide an answer in the format hh:mm:ss.")
T_PAIR_FIRST = ("At what time were a {a} and a {b} first visible at the same time in this video? "
                "Please provide an answer in the format hh:mm:ss.")
T_QUAD_POP = ("Considering the center of the foreign object relative to the image center, which "
              "quadrants of the video have been populated with any foreign object during the whole "
              "video? Please select one or multiple answers: top/left; top/right; bottom/left; bottom/right")
T_QUAD_NOTPOP = ("Considering the center of the foreign object relative to the image center, which "
                 "quadrants of the video have not been populated with any foreign object during the "
                 "whole video? Please select none, one or multiple answers: top/left; top/right; "
                 "bottom/left; bottom/right")
T_FIRST_QUAD = ("When a {cls} first appears in this video, in which quadrant of the frame is the "
                "center of the {cls} located relative to the image center in that moment? "
                "Please select one answer: top/left; top/right; bottom/left; bottom/right")
T_LAST_QUAD = ("The last time a {cls} is visible, where was the center of the {cls} located "
               "relative to the image center? Please select one answer: top/left; top/right; "
               "bottom/left; bottom/right")
T_ALSO_APPEAR = "Does the {cls} at {t0} also appear at {t1}? Please answer with yes or no."
T_POSITIONS = G.T_POSITIONS
# ★計数系（2026-08-28 追加）。公式 SEGMENT の number は 63 問と少ないが、最頻は
#   「一度に最大何個」= **各フレームの個数の最大値**で、個体同一性が要らない＝注釈だけで厳密に出せる。
#   （「動画中に何個の distinct な X が居たか」は個体同一性が要るので v1 では作らない）
T_MAX_AT_ONCE = ("What is the maximum number of {plural} appearing at once in a single frame? "
                 "Please provide a number.")
T_CLASS_COUNT_VIDEO = ("How many different foreign object classes do you see in this video? "
                       "Please provide a number.")

FAMILY_FORMAT = {
    "classes_in_video": "fo_class", "single_in_video": "fo_class",
    "classes_between": "fo_class", "nth_unique": "fo_class",
    "last_visible": "time", "first_visible": "time",
    "pair_last": "time", "pair_first": "time",
    "quad_populated": "multiple_choice", "quad_not_populated": "multiple_choice",
    "first_quad": "multiple_choice", "last_quad": "multiple_choice",
    "also_appear": "binary", "positions_all": "open_ended",
    "max_at_once": "number", "class_count_video": "number",
}
FAMILY_CAPABILITY = {
    "classes_in_video": "1a", "single_in_video": "1a", "classes_between": "1a",
    "nth_unique": "1a", "last_visible": "2a", "first_visible": "2a",
    "pair_last": "2a", "pair_first": "2a",
    "quad_populated": "1d", "quad_not_populated": "1d",
    "first_quad": "1d", "last_quad": "1d",
    "also_appear": "1b", "positions_all": "1d",
    "max_at_once": "3a", "class_count_video": "3a",
}
# 公式 SEGMENT train の出現数に比例（導出可能な family のみ）
# ★★★時刻系（last_visible / first_visible / pair_*）は **v1 では作らない**（2026-08-26 実測で棄却）:
#   - 公式 GT との一致率が **last_visible 0.000 (0/6) / first_visible 0.217 (5/23)**。
#     誤差の符号中央は **−16s**＝注釈イベント間隔（median ~40s）の間に本当の消失があり、
#     我々の答えは**常に早すぎる**。許容は min(5, 1+dur*4/360) で 300s クリップでも 4.3s しかない。
#   - 論理的に保証できる条件B（隣接イベントで不在が確認でき、その間隔が許容内）は
#     **全 995 窓で first 32 / last 5 問しか成立しない**（イベントが疎すぎる）。
#   - 条件A（クリップ端で見えている＝答えは端）は 1,215 問取れるが、**公式の答えは
#     80〜95% がクリップ中間**（端は last 13% / first 5%）。A ばかり作ると
#     「端を答える」退化を教えることになり、temporal_grounding を壊す危険がある。
#   → **時刻系は密注釈が取れるまで作らない**。テンプレと `safe_boundary_time` は v2 用に残す。
QUOTA_WEIGHTS = {
    "classes_in_video": 367, "classes_between": 313, "quad_not_populated": 190,
    "quad_populated": 179, "single_in_video": 177, "nth_unique": 162,
    "last_quad": 145, "also_appear": 129, "first_quad": 125, "positions_all": 52,
    # ★number（公式 SEGMENT の 7%）。K03 で「SEGMENT 擬似に number が 0 問」だったせいで
    #   FRAME 側の number 偏重（44%）が SEGMENT の計数を壊した（−0.0685**）。実比率に寄せる。
    #   実測: quota 31/6 だと生成 3.2% にしかならない（max_at_once は「そのクラスが
    #   窓に居る」制約で歩留まりが低い）。**生成後の実比率が 7% になるよう quota を厚くする**。
    "max_at_once": 95, "class_count_video": 20,
}

log = logging.getLogger("expK00.seg")



def time_tol(duration: float) -> float:
    """公式の time 許容（`base_dataset.py`: min(5, 1 + dur*4/360)）。"""
    return min(5.0, 1.0 + duration * (4.0 / 360.0))


def safe_boundary_time(per_t: list, cls_set: set, kind: str, tol: float):
    """★出現/消失時刻が **許容内で保証できるときだけ** 返す（できなければ None）.

    2026-08-26 の検算で、素朴に「見えている最後のイベント」を答えにすると
    公式 GT との一致率が **last_visible 0.000 / first_visible 0.217** しか無かった
    （誤差中央 −16s。注釈イベント間隔 ~40s の間に本当の消失があるため、
    我々の答えは常に**早すぎる**）。

    そこで次の2条件のどちらかを満たすときだけ採用する:
      A. クリップ端のイベントで見えている → 答えはクリップ端そのもの（**厳密**）
      B. 隣のイベント（消失側）で**そのクラスが写っておらず**、その間隔が許容 `tol` 以内
         → 誤差は `tol` 以下に抑えられる
    """
    ts = [t for t, d, _, _ in per_t if cls_set <= set(d)]
    if not ts:
        return None
    if kind == "last":
        t = max(ts)
        if t == per_t[-1][0]:                       # A: クリップ端まで見えている
            return t
        nxt = next((tt for tt, d, _, _ in per_t if tt > t), None)   # B
        if nxt is not None and nxt - t <= tol:
            return t
        return None
    t = min(ts)
    if t == per_t[0][0]:                            # A
        return t
    prv = max((tt for tt, d, _, _ in per_t if tt < t), default=None)  # B
    if prv is not None and t - prv <= tol:
        return t
    return None


def ordinal_suffix(n: int) -> str:
    # ★公式は "1st unique class" だが実データは "1st/2nd/3rd" を使う。実測に合わせる
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def load_events() -> dict:
    """動画 → 注釈イベント列（バーストを畳んだ代表フレーム）。qa fold0（VQA val）は除外。"""
    qa_folds, inst_train = G.load_folds()
    out: dict = {}
    n_excl = 0
    for ds, dump in G.DUMPS.items():
        coco = json.load(open(dump / "coco/instances.json"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a)
        by_video = defaultdict(list)
        for im in coco["images"]:
            by_video[im["video"]].append(im)
        for video, ims in sorted(by_video.items()):
            stem = Path(video).stem
            fold = qa_folds.get(stem)
            if fold == 0:
                n_excl += 1
                continue
            evs = []
            last_t = -1e9
            for im in sorted(ims, key=lambda x: x["frame_number"]):
                t = im["frame_number"] / G.FPS[ds]
                if t - last_t < BURST_GAP_S:      # 同じ瞬間のバーストは代表1枚だけ
                    continue
                insts = []
                for a in anns.get(im["id"], []):
                    cx, cy = G.centroid(a)
                    insts.append({"cls": G.CLASS_MAP[cats[a["category_id"]]][0],
                                  "cx": cx, "cy": cy, "area": a["area"]})
                if not insts:
                    continue
                last_t = t
                evs.append({"t": t, "frame": im["frame_number"], "w": im["width"],
                            "h": im["height"], "path": str(dump / im["file_name"]),
                            "insts": insts})
            if len(evs) >= MIN_EVENTS:
                out[stem] = {"dataset": ds, "video": video, "events": evs,
                             "split": "qa_train" if fold is not None else "out_of_fold",
                             "instseg_train": stem in inst_train}
    log.info(f"動画 {len(out)}（qa fold0 除外 {n_excl}）"
             f"／イベント計 {sum(len(v['events']) for v in out.values()):,}")
    return out


def make_windows(store: dict) -> list[dict]:
    """duration<=300s・イベント>=MIN_EVENTS の窓を、重なりを抑えつつ列挙する。"""
    # ★スライディング窓（stride=WINDOW_STRIDE イベント）。窓は重なってよい:
    #   公式も同一動画から重なるクリップを大量に出題しており、証拠集合が違えば別サンプル。
    wins = []
    for stem, v in store.items():
        evs = v["events"]
        for i in range(0, len(evs), WINDOW_STRIDE):
            j = i
            while j + 1 < len(evs) and evs[j + 1]["t"] - evs[i]["t"] <= MAX_CLIP_S:
                j += 1
            if j - i + 1 < MIN_EVENTS or evs[j]["t"] - evs[i]["t"] < MIN_CLIP_S:
                continue
            sel = evs[i:j + 1]
            if len(sel) > MAX_FRAMES:            # 等間隔に間引く（端は必ず残す）
                idx = np.linspace(0, len(sel) - 1, MAX_FRAMES).round().astype(int)
                sel = [sel[k] for k in sorted(set(idx.tolist()))]
            wins.append({"stem": stem, "dataset": v["dataset"], "video": v["video"],
                         "split": v["split"], "instseg_train": v["instseg_train"],
                         "start": sel[0]["t"], "end": sel[-1]["t"], "events": sel})
    return wins


# ── 窓から導ける事実 ────────────────────────────────────────────────
def quad_of(inst, w, h) -> str:
    return G.quadrant(inst["cx"], inst["cy"], w, h)


def facts(win: dict) -> dict:
    evs = win["events"]
    per_t = []           # [(t, {cls: [inst,...]})]
    for e in evs:
        d = defaultdict(list)
        for i in e["insts"]:
            d[i["cls"]].append(i)
        per_t.append((e["t"], d, e["w"], e["h"]))
    classes = sorted({c for _, d, _, _ in per_t for c in d})
    first_t = {c: min(t for t, d, _, _ in per_t if c in d) for c in classes}
    last_t = {c: max(t for t, d, _, _ in per_t if c in d) for c in classes}
    quads = sorted({quad_of(i, w, h) for _, d, w, h in per_t for v in d.values() for i in v},
                   key=QUAD_ORDER.index)
    order = sorted(classes, key=lambda c: first_t[c])
    return {"per_t": per_t, "classes": classes, "first_t": first_t, "last_t": last_t,
            "quads": quads, "order": order}


def gen_qa(fam: str, win: dict, f: dict, rng: random.Random):
    per_t, classes = f["per_t"], f["classes"]
    if fam == "classes_in_video":
        return T_CLASSES_IN_VIDEO, ", ".join(classes)
    if fam == "single_in_video":
        if len(classes) != 1:
            return None
        return T_SINGLE_IN_VIDEO, classes[0]
    if fam == "classes_between":
        # 窓内の部分区間。**両端は必ず注釈イベント時刻**にする（区間内に証拠がある形にする）
        if len(per_t) < 3:
            return None
        i = rng.randrange(0, len(per_t) - 2)
        j = rng.randrange(i + 1, len(per_t))
        t0, t1 = per_t[i][0], per_t[j][0]
        sub = sorted({c for t, d, _, _ in per_t if t0 <= t <= t1 for c in d})
        return (T_CLASSES_BETWEEN.format(t0=G.hhmmss(t0), t1=G.hhmmss(t1)),
                ", ".join(sub) if sub else "none")
    if fam == "nth_unique":
        n = rng.randint(1, min(3, len(f["order"])))
        return (T_NTH_UNIQUE.format(n=n, suf=ordinal_suffix(n)), f["order"][n - 1])
    if fam in ("last_visible", "first_visible"):
        tol = time_tol(win["end"] - win["start"])
        kind = "last" if fam == "last_visible" else "first"
        cands = [c for c in classes
                 if safe_boundary_time(per_t, {c}, kind, tol) is not None]
        if not cands:
            return None
        cls = rng.choice(cands)
        t = safe_boundary_time(per_t, {cls}, kind, tol)
        tmpl = T_LAST_VISIBLE if fam == "last_visible" else T_FIRST_VISIBLE
        return tmpl.format(cls=cls), G.hhmmss(t)
    if fam in ("pair_last", "pair_first"):
        both = [(t, d) for t, d, _, _ in per_t if len(d) >= 2]
        if not both:
            return None
        pairs = {tuple(sorted(p)) for _, d in both for p in
                 [(a, b) for a in d for b in d if a < b]}
        if not pairs:
            return None
        tol = time_tol(win["end"] - win["start"])
        kind = "last" if fam == "pair_last" else "first"
        ok = [(a, b) for a, b in sorted(pairs)
              if safe_boundary_time(per_t, {a, b}, kind, tol) is not None]
        if not ok:
            return None
        a, b = rng.choice(ok)
        t = safe_boundary_time(per_t, {a, b}, kind, tol)
        tmpl = T_PAIR_LAST if fam == "pair_last" else T_PAIR_FIRST
        return tmpl.format(a=a, b=b), G.hhmmss(t)
    if fam == "quad_populated":
        return T_QUAD_POP, ", ".join(f["quads"])
    if fam == "quad_not_populated":
        miss = [q for q in QUAD_ORDER if q not in f["quads"]]
        return T_QUAD_NOTPOP, ", ".join(miss) if miss else "none"
    if fam in ("first_quad", "last_quad"):
        cls = rng.choice(classes)
        t = f["first_t"][cls] if fam == "first_quad" else f["last_t"][cls]
        ent = next((d, w, h) for tt, d, w, h in per_t if tt == t and cls in d)
        d, w, h = ent
        if len(d[cls]) != 1:            # 同時に複数個あると「その物体の象限」が一意でない
            return None
        tmpl = T_FIRST_QUAD if fam == "first_quad" else T_LAST_QUAD
        return tmpl.format(cls=cls), quad_of(d[cls][0], w, h)
    if fam == "also_appear":
        # 片方の時刻に単一個体で存在するクラスを選び、別時刻での**そのクラスの有無**で答える
        cand = [(t, d) for t, d, _, _ in per_t for c in d if len(d[c]) == 1]
        if not cand or len(per_t) < 2:
            return None
        t0, d0 = rng.choice(cand)
        cls = rng.choice([c for c in d0 if len(d0[c]) == 1])
        others = [t for t, _, _, _ in per_t if t != t0]
        yes_ts = [t for t in others if cls in dict((tt, dd) for tt, dd, _, _ in per_t)[t]]
        no_ts = [t for t in others if t not in yes_ts]
        want_yes = rng.random() < 0.5
        pool = yes_ts if (want_yes and yes_ts) else (no_ts if no_ts else yes_ts)
        if not pool:
            return None
        t1 = rng.choice(pool)
        ans = "yes" if t1 in yes_ts else "no"
        return T_ALSO_APPEAR.format(cls=cls, t0=G.hhmmss(t0), t1=G.hhmmss(t1)), ans
    if fam == "max_at_once":
        # 各フレームでのそのクラスの個数の最大値。**個体同一性は不要**なので注釈だけで厳密。
        cls = rng.choice(classes)
        n = max(len(d.get(cls, [])) for _, d, _, _ in per_t)
        return T_MAX_AT_ONCE.format(plural=PLURAL[cls]), str(n)
    if fam == "class_count_video":
        return T_CLASS_COUNT_VIDEO, str(len(classes))
    if fam == "positions_all":
        t, d, w, h = per_t[rng.randrange(len(per_t))]
        insts = [i for v in d.values() for i in v]
        if len(insts) > G.POSITIONS_MAX_INST:
            return None
        ordered = sorted(insts, key=lambda i: (QUAD_ORDER.index(quad_of(i, w, h)), i["cls"]))
        ans = " ".join(f"{k}. {i['cls']}: {quad_of(i, w, h)}" for k, i in enumerate(ordered, 1))
        return T_POSITIONS.format(hms=G.hhmmss(t)), ans
    return None


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(OUT / f"gen_segment_{ts}.log"),
                                  logging.StreamHandler()])
    rng = random.Random(SEED)
    store = load_events()
    wins = make_windows(store)
    rng.shuffle(wins)
    log.info(f"窓 {len(wins):,}（frames/窓 median "
             f"{int(np.median([len(w['events']) for w in wins]))}, "
             f"duration median {np.median([w['end'] - w['start'] for w in wins]):.0f}s）")

    wsum = sum(QUOTA_WEIGHTS.values())
    quota = {k: max(1, round(TARGET_TOTAL * v / wsum)) for k, v in QUOTA_WEIGHTS.items()}
    rows = []
    for win in wins:
        f = facts(win)
        fams = [k for k in QUOTA_WEIGHTS if quota[k] > 0]
        picked = 0
        while fams and picked < MAX_Q_PER_WINDOW:
            fam = rng.choices(fams, weights=[quota[k] for k in fams])[0]
            fams.remove(fam)
            qa = gen_qa(fam, win, f, rng)
            if qa is None:
                continue
            q, a = qa
            quota[fam] -= 1
            picked += 1
            rows.append({
                "id": f"ps{G.VERSION[1:]}_{win['dataset']}_{win['stem'].split(' ')[0]}_"
                      f"{int(win['start'])}_{fam}",
                "dataset": win["dataset"], "video": win["video"],
                "start_time": win["start"], "end_time": win["end"],
                "frame_times": json.dumps([e["t"] for e in win["events"]]),
                "frame_paths": json.dumps([e["path"] for e in win["events"]]),
                "n_frames": len(win["events"]),
                "question": q, "answer": a,
                "answer_format": FAMILY_FORMAT[fam],
                "primary_capability": FAMILY_CAPABILITY[fam],
                "family": fam, "track": "SEGMENT", "generation": f"pseudo_segment_{G.VERSION}",
                "split": win["split"], "instseg_train": win["instseg_train"],
            })
        if all(v <= 0 for v in quota.values()):
            break

    df = pd.DataFrame(rows)
    # 同一窓×同一 family は id が衝突しうる（窓が重なるため）→ 連番で一意化
    if not df["id"].is_unique:
        df["id"] = df["id"] + "_" + df.groupby("id").cumcount().astype(str)
    assert df["id"].is_unique, "id 重複"
    # ★同じ (動画, 質問文, 答え) の重複は落とす（重なり窓が同じ問題を量産するため）
    n0 = len(df)
    df = df.drop_duplicates(subset=["video", "question", "answer"]).reset_index(drop=True)
    log.info(f"重複除去 {n0:,} → {len(df):,}")
    # 時刻回答は必ずクリップ内（公式も区間内にあることを前提に採点される）
    bad = 0
    for r in df[df.answer_format == "time"].itertuples():
        sec = sum(int(x) * m for x, m in zip(r.answer.split(":"), (3600, 60, 1)))
        if not (r.start_time - 1 <= sec <= r.end_time + 1):
            bad += 1
    assert bad == 0, f"クリップ外の時刻回答 {bad} 件"
    out = OUT / f"pseudo_segment_{G.VERSION}.parquet"
    df.to_parquet(out)
    log.info(f"書き出し {len(df):,} 問 / 窓 {df.groupby(['video','start_time']).ngroups:,} → {out}")
    log.info("family:\n" + df["family"].value_counts().to_string())
    log.info(f"split: {df['split'].value_counts().to_dict()}")
    log.info(f"形式: {df['answer_format'].value_counts().to_dict()}")


if __name__ == "__main__":
    import sys
    G.parse_args()      # dump / version / instseg fold は FRAME 側と同じ CLI で切り替える
    # --target-total は FRAME 側の定数しか書き換えないので、明示指定時だけ SEGMENT 側へ反映する
    if any(a.startswith("--target-total") for a in sys.argv[1:]):
        TARGET_TOTAL = G.TARGET_TOTAL
    main()
