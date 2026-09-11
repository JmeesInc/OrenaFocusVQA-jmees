"""FRAME 向け擬似 VQA 生成 — survis-anno instseg アノテーション → 公式テンプレ互換 QA.

Source: /mnt/data/data4/input/survis-anno/{focus-heico,focus-lapchole}/*_20260825_001
出力: out/pseudo_frame_v1.parquet + out/stats.md

設計（SESSION_NOTES.md に詳細）:
- 質問文は公式 lapchole FRAME train parquet のテンプレ文字列と **byte 一致**（パラメータ置換のみ）
- 回答語彙も公式に一致（Clip / Specimen bag / Silicone loop / ... アルファベット順連結）
- ★expD14 の教訓: FRAME に GT=0 の計数問題は存在しない → **陽性フレームのみ・count>=1 のみ生成**
- ★リーク対策: qa fold v003 の fold0（VQA val）動画は **生成から除外**。
  各行に instseg_v003 train 由来かのフラグを付け、overlay 入力系の実験で分岐できるようにする
"""
from __future__ import annotations

import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE / "out"

# ── 設定 ──────────────────────────────────────────────────────────────
# ★2026-08-28 dump: lapchole 58→99 動画 / 37,847→76,735 枚 / 71,965→172,495 inst。
#   新規 41 動画（0009/0027/0071/0116/... 大半が未ラベル群）がここで初めて注釈された。
#   heico は 0825 が最新（更新なし）。
DUMPS = {
    "heico": Path("/mnt/data/data4/input/survis-anno/focus-heico/focus-heico_20260825_001"),
    "lapchole": Path("/mnt/data/data4/input/survis-anno/focus-lapchole/focus-lapchole_20260828_001"),
}
VERSION = "v2"           # 出力名・id 接頭辞・generation 列に入る（--version で上書き可）
FPS = {"heico": 25.0, "lapchole": 30.0}
QA_FOLD_CSV = ROOT / "workspace/fold/v003/folds.csv"
# ★instseg_v004 = **VQA fold v003 に揃えた** instseg split（fold0 = VQA val = instseg val）。
#   v003 までは VQA fold と独立に切られており、instseg train に VQA val 動画が
#   8 本混ざっていた（重畳入力・FO 索引の CV が過大評価になる）。
INSTSEG_FOLD_CSV = ROOT / "workspace/fold/instseg_v004/folds.csv"
MIN_GAP_S = 3.0          # 同一動画内のフレーム最小間隔（1秒バースト注釈の冗長性除去）
MAX_Q_PER_FRAME = 3
TARGET_TOTAL = 12000
CLOSEST_MARGIN_FRAC = 0.03   # closest_center: 1位と2位の距離差が対角のこの割合未満なら曖昧としてスキップ
POSITIONS_MAX_INST = 8
SEED = 20260825

# dump ラベル → 公式語彙（単数, 複数）
CLASS_MAP = {
    "sponge": ("Sponge", "Sponges"),
    "clip": ("Clip", "Clips"),
    "Specimen_Bag": ("Specimen bag", "Specimen bags"),
    "Silicon_Loop": ("Silicone loop", "Silicone loops"),
    "External_Drain": ("External drain", "External drains"),
    "Needle": ("Needle", "Needles"),
    "Gallstone": ("Gallstone", "Gallstones"),
    "Specimen": ("Specimen", "Specimens"),
}
PLURAL = {s: p for s, p in CLASS_MAP.values()}

# 公式 lapchole FRAME train 5,748問の family 構成比（'other'=属性系490問は生成不能につき除外）
QUOTA_WEIGHTS = {
    "inst_count": 916, "list_all": 849, "class_count": 813, "single_object": 548,
    "combination": 387, "quad_to_class": 381, "class_diversity": 327, "same_class": 234,
    "class_to_quad": 232, "co_occur": 228, "positions_all": 194, "closest_center": 149,
}
FAMILY_CAPABILITY = {
    "inst_count": "3a", "class_count": "3a", "class_diversity": "3a", "same_class": "3a",
    "co_occur": "3a", "list_all": "1a", "single_object": "1a", "combination": "1a",
    "quad_to_class": "1a", "class_to_quad": "1d", "positions_all": "1d", "closest_center": "1d",
}
FAMILY_FORMAT = {
    "inst_count": "number", "class_count": "number", "class_diversity": "number",
    "same_class": "binary", "co_occur": "binary", "list_all": "fo_class",
    "single_object": "fo_class", "combination": "fo_class", "quad_to_class": "fo_class",
    "class_to_quad": "multiple_choice", "positions_all": "open_ended",
    "closest_center": "fo_class",
}

# 公式テンプレ（lapchole FRAME train parquet から byte 一致で採取, 2026-08-25 確認）
T_INST_COUNT = "How many different foreign object instances appear in this frame? Please provide a number."
T_CLASS_COUNT = "How many {plural} appear in this frame? Please provide a number."
T_CLASS_DIVERSITY = "How many different foreign object classes appear in this frame? Please provide a number."
T_LIST_ALL = "List all foreign objects that are visible in this video frame. Please provide the class names or answer with none."
T_SINGLE = ("There is one surgical foreign object visible in the frame. "
            "What surgical foreign object is visible in this video frame? Please provide a class name.")
T_COMBINATION = ("Which combination of foreign object classes is visible in this frame? "
                 "Please provide the class names or answer with none.")
T_SAME_CLASS = "Are all visible foreign objects in this frame of the same class? Please answer with yes or no."
T_CO_OCCUR = "Do {pa} and {pb} co-occur in this frame? Please answer with yes or no."
T_QUAD_TO_CLASS = ("What class is the foreign object located in the {quad} relative to the image center? "
                   "Please provide a class name.")
T_CLASS_TO_QUAD = ("Where is the center of the {cls} located relative to the image center in this frame? "
                   "Please select one answer: top/left; top/right; bottom/left; bottom/right")
T_CLOSEST = ("Which of the visible foreign objects has its centre closest to the centre of the image? "
             "Please provide a class name.")
T_POSITIONS = ("At timepoint {hms} please provide all relative central positions of foreign objects present "
               "in the frame. Please provide the answer in the following format: “number. object type: "
               "quadrant”, where number represents an enumeration starting with 1, object type is the type "
               "of the foreign object and quadrant is one of the following options: top/left, top/right, "
               "bottom/left, bottom/right. Respond “none” in case there are no foreign objects present "
               "at timepoint {hms}. For example: 1. Sponge: top/left 2. Sponge: top/right 3. Needle: bottom/left")

log = logging.getLogger("expK00")


def setup_logging() -> None:
    OUT.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(OUT / f"generate_{ts}.log")
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logging.basicConfig(level=logging.DEBUG, handlers=[fh, sh],
                        format="%(asctime)s | %(levelname)s | %(message)s")


def hhmmss(sec: float) -> str:
    s = int(round(max(sec, 0)))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def quadrant(cx: float, cy: float, w: int, h: int) -> str:
    vert = "top" if cy < h / 2 else "bottom"
    horiz = "left" if cx < w / 2 else "right"
    return f"{vert}/{horiz}"


QUAD_ORDER = ["top/left", "top/right", "bottom/left", "bottom/right"]


def centroid(ann: dict) -> tuple[float, float]:
    """RLE マスク重心。失敗時は bbox 中心（細長い loop/drain で bbox 中心は大きくズレるため重心優先）。"""
    seg = ann.get("segmentation")
    try:
        m = mask_util.decode(seg)
        ys, xs = np.nonzero(m)
        if len(xs) > 0:
            return float(xs.mean()), float(ys.mean())
    except Exception:
        pass
    x, y, bw, bh = ann["bbox"]
    return x + bw / 2, y + bh / 2


def load_folds() -> tuple[dict[str, int], set[str]]:
    import csv
    qa = {}
    with open(QA_FOLD_CSV) as f:
        for r in csv.DictReader(f):
            qa[Path(r["videoID"]).stem] = int(r["fold"])
    inst_train = set()
    with open(INSTSEG_FOLD_CSV) as f:
        for r in csv.DictReader(f):
            if int(r["fold"]) != 0:
                inst_train.add(Path(r["video"]).stem)
    return qa, inst_train


def load_frames() -> list[dict]:
    """dump → フレームレコード（動画・時刻・インスタンス一覧）。qa fold0 動画は除外。"""
    qa_folds, inst_train = load_folds()
    frames: list[dict] = []
    excluded_val, skipped_gap = 0, 0
    for ds, dump in DUMPS.items():
        coco = json.load(open(dump / "coco/instances.json"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        by_video = defaultdict(list)
        for im in coco["images"]:
            by_video[im["video"]].append(im)
        for video, ims in sorted(by_video.items()):
            stem = Path(video).stem
            fold = qa_folds.get(stem)
            if fold == 0:
                excluded_val += len(ims)
                continue
            split = "qa_train" if fold is not None else "out_of_fold"
            last_t = -1e9
            for im in sorted(ims, key=lambda x: x["frame_number"]):
                t = im["frame_number"] / FPS[ds]
                if t - last_t < MIN_GAP_S:
                    skipped_gap += 1
                    continue
                insts = []
                for a in anns_by_img.get(im["id"], []):
                    cx, cy = centroid(a)
                    insts.append({"cls": CLASS_MAP[cats[a["category_id"]]][0],
                                  "cx": cx, "cy": cy, "area": a["area"]})
                if not insts:
                    continue
                last_t = t
                frames.append({
                    "dataset": ds, "video": video, "stem": stem,
                    "frame_number": im["frame_number"], "t": t,
                    "w": im["width"], "h": im["height"],
                    "path": str(dump / im["file_name"]),
                    "split": split,
                    "instseg_train": stem in inst_train,
                    "insts": insts,
                })
    log.info(f"フレーム採択 {len(frames):,}（qa-val 除外 {excluded_val:,} / "
             f"{MIN_GAP_S}s 間隔で間引き {skipped_gap:,}）")
    return frames


# ── family ごとの適用可否と (question, answer) 生成 ──────────────────────
def applicable(fr: dict) -> list[str]:
    insts = fr["insts"]
    classes = sorted({i["cls"] for i in insts})
    fams = ["inst_count", "list_all", "class_diversity", "combination", "class_count"]
    if len(insts) <= POSITIONS_MAX_INST:
        fams.append("positions_all")
    if len(insts) == 1:
        fams.append("single_object")
    if len(insts) >= 2:
        fams += ["same_class", "closest_center"]
    if len(classes) >= 2 or _absent_class(fr):
        fams.append("co_occur")
    if _quad_single(fr):
        fams.append("quad_to_class")
    if _cls_single(fr):
        fams.append("class_to_quad")
    return fams


def _quad_single(fr: dict) -> list[str]:
    q = Counter(quadrant(i["cx"], i["cy"], fr["w"], fr["h"]) for i in fr["insts"])
    return [k for k, v in q.items() if v == 1]


def _cls_single(fr: dict) -> list[str]:
    c = Counter(i["cls"] for i in fr["insts"])
    return sorted(k for k, v in c.items() if v == 1)


def _absent_class(fr: dict) -> list[str]:
    present = {i["cls"] for i in fr["insts"]}
    return sorted(set(PLURAL) - present)


_CO_STATE = {"yes": 0, "no": 0}


def gen_qa(fam: str, fr: dict, rng: random.Random) -> tuple[str, str] | None:
    insts = fr["insts"]
    classes = sorted({i["cls"] for i in insts})
    w, h = fr["w"], fr["h"]
    if fam == "inst_count":
        return T_INST_COUNT, str(len(insts))
    if fam == "class_diversity":
        return T_CLASS_DIVERSITY, str(len(classes))
    if fam == "class_count":
        cls = rng.choice(classes)
        n = sum(1 for i in insts if i["cls"] == cls)
        return T_CLASS_COUNT.format(plural=PLURAL[cls]), str(n)
    if fam == "list_all":
        return T_LIST_ALL, ", ".join(classes)
    if fam == "combination":
        return T_COMBINATION, ", ".join(classes)
    if fam == "single_object":
        return T_SINGLE, insts[0]["cls"]
    if fam == "same_class":
        return T_SAME_CLASS, "yes" if len(classes) == 1 else "no"
    if fam == "co_occur":
        # yes:no ≈ 45:55（公式 102:126）になるよう全体カウンタで制御。
        # 多クラスフレームは少数派なので、可能なら常に yes を優先する
        absent = _absent_class(fr)
        if len(classes) >= 2:
            a, b = rng.sample(classes, 2)
            ans = "yes"
        elif absent and _CO_STATE["no"] < (_CO_STATE["yes"] + 1) * 1.25:
            # 単クラスフレームは多数派なので、no は yes の 1.25 倍までに絞る
            a, b = rng.choice(classes), rng.choice(absent)
            ans = "no"
        else:
            return None
        _CO_STATE[ans] += 1
        a, b = sorted([a, b])
        return T_CO_OCCUR.format(pa=PLURAL[a], pb=PLURAL[b]), ans
    if fam == "quad_to_class":
        quads = _quad_single(fr)
        if not quads:
            return None
        qd = rng.choice(quads)
        cls = next(i["cls"] for i in insts if quadrant(i["cx"], i["cy"], w, h) == qd)
        return T_QUAD_TO_CLASS.format(quad=qd), cls
    if fam == "class_to_quad":
        singles = _cls_single(fr)
        if not singles:
            return None
        cls = rng.choice(singles)
        inst = next(i for i in insts if i["cls"] == cls)
        return T_CLASS_TO_QUAD.format(cls=cls), quadrant(inst["cx"], inst["cy"], w, h)
    if fam == "closest_center":
        d = sorted(((i["cx"] - w / 2) ** 2 + (i["cy"] - h / 2) ** 2, i["cls"]) for i in insts)
        diag = (w ** 2 + h ** 2) ** 0.5
        if len(d) < 2 or (d[1][0] ** 0.5 - d[0][0] ** 0.5) < CLOSEST_MARGIN_FRAC * diag:
            return None
        return T_CLOSEST, d[0][1]
    if fam == "positions_all":
        hms = hhmmss(fr["t"])
        ordered = sorted(insts, key=lambda i: (QUAD_ORDER.index(quadrant(i["cx"], i["cy"], w, h)),
                                               i["cls"]))
        ans = " ".join(f"{k}. {i['cls']}: {quadrant(i['cx'], i['cy'], w, h)}"
                       for k, i in enumerate(ordered, 1))
        return T_POSITIONS.format(hms=hms), ans
    return None


def parse_args(argv: list[str] | None = None) -> None:
    """モジュール定数を CLI で上書きする（旧 dump / 旧バージョンの再生成を残すため）。"""
    import argparse
    global VERSION, TARGET_TOTAL, INSTSEG_FOLD_CSV
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=VERSION, help="出力名・id 接頭辞（例: v2）")
    ap.add_argument("--target-total", type=int, default=TARGET_TOTAL)
    ap.add_argument("--dump-lapchole", default=str(DUMPS["lapchole"]))
    ap.add_argument("--dump-heico", default=str(DUMPS["heico"]))
    ap.add_argument("--instseg-folds", default=str(INSTSEG_FOLD_CSV))
    a = ap.parse_args(argv)
    VERSION = a.version
    TARGET_TOTAL = a.target_total
    DUMPS["lapchole"] = Path(a.dump_lapchole)
    DUMPS["heico"] = Path(a.dump_heico)
    INSTSEG_FOLD_CSV = Path(a.instseg_folds)


def main() -> None:
    setup_logging()
    log.info(f"version={VERSION} target={TARGET_TOTAL} dumps={ {k: str(v) for k, v in DUMPS.items()} }")
    rng = random.Random(SEED)
    frames = load_frames()
    rng.shuffle(frames)

    wsum = sum(QUOTA_WEIGHTS.values())
    quota = {f: max(1, round(TARGET_TOTAL * n / wsum)) for f, n in QUOTA_WEIGHTS.items()}
    log.info(f"quota: {quota}")

    rows = []
    for fr in frames:
        fams = [f for f in applicable(fr) if quota.get(f, 0) > 0]
        if not fams:
            continue
        # 残 quota に比例して family を抽選（大 family が枠を独占して希少 family が
        # 枯れるのを防ぐ。v1 初版は「残 quota 降順」で positions_all 54 / closest 2 に枯れた）
        picked = 0
        while fams and picked < MAX_Q_PER_FRAME:
            fam = rng.choices(fams, weights=[quota[f] for f in fams])[0]
            fams.remove(fam)
            qa = gen_qa(fam, fr, rng)
            if qa is None:
                continue
            q, a = qa
            quota[fam] -= 1
            picked += 1
            rows.append({
                "id": f"pf{VERSION[1:]}_{fr['dataset']}_{fr['stem'].split(' ')[0]}_{fr['frame_number']}_{fam}",
                "dataset": fr["dataset"], "video": fr["video"],
                "frame_number": fr["frame_number"], "timestamp": fr["t"],
                "question": q, "answer": a,
                "answer_format": FAMILY_FORMAT[fam],
                "primary_capability": FAMILY_CAPABILITY[fam],
                "family": fam, "track": "FRAME", "generation": f"pseudo_{VERSION}",
                "frame_path": fr["path"], "split": fr["split"],
                "instseg_train": fr["instseg_train"],
                "n_inst": len(fr["insts"]),
            })
        if all(v <= 0 for v in quota.values()):
            break

    df = pd.DataFrame(rows)
    assert df["id"].is_unique
    assert (df.groupby(["video", "frame_number"])["family"].nunique()
            == df.groupby(["video", "frame_number"])["family"].size()).all(), "同一フレーム同一familyの重複"
    # FRAME の number に GT=0 が無い公式性質を保っていること（expD14 の教訓）
    assert not ((df["answer_format"] == "number") & (df["answer"] == "0")).any()

    out = OUT / f"pseudo_frame_{VERSION}.parquet"
    df.to_parquet(out)
    log.info(f"書き出し {len(df):,} 問 → {out}")

    # ── stats.md ──
    lines = [f"# pseudo_frame_{VERSION} 統計\n",
             f"- 総数 **{len(df):,}** 問 / フレーム {df.groupby(['video','frame_number']).ngroups:,} / "
             f"動画 {df['video'].nunique()}（heico {df[df.dataset=='heico']['video'].nunique()} / "
             f"lapchole {df[df.dataset=='lapchole']['video'].nunique()}）",
             f"- split: {df['split'].value_counts().to_dict()} / "
             f"instseg_train フラグ: {df['instseg_train'].value_counts().to_dict()}\n",
             "## family 別（公式構成比との比較）\n",
             "| family | 生成 | 生成% | 公式% |", "|---|---|---|---|"]
    off_total = sum(QUOTA_WEIGHTS.values())
    for fam, n_off in sorted(QUOTA_WEIGHTS.items(), key=lambda x: -x[1]):
        n = (df["family"] == fam).sum()
        lines.append(f"| {fam} | {n} | {100*n/len(df):.1f} | {100*n_off/off_total:.1f} |")
    lines += ["\n## number 回答分布（inst_count）\n",
              str(df[df.family == "inst_count"]["answer"].astype(int).value_counts().sort_index().to_dict()),
              "\n## binary 回答分布\n",
              str(df[df.answer_format == "binary"].groupby("family")["answer"]
                  .value_counts().to_dict())]
    (OUT / f"stats_{VERSION}.md").write_text("\n".join(lines))
    log.info("family 別:\n" + df["family"].value_counts().to_string())
    log.info(f"stats → {OUT/f'stats_{VERSION}.md'}")


if __name__ == "__main__":
    parse_args()
    main()
