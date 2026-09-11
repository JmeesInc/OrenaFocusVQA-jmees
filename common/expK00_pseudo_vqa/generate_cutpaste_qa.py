"""RARP Needle crop を FOCUS 注釈フレームへ cut-paste して擬似 QA を生成する.

狙い（2026-08-23 調査の帰結）:
- SAR-RARP50 は Needle の動画多様性を 14→58 本に増やせる唯一の外部供給源だが、
  **針が注釈フレームの 77% に写る**ので素朴追加は「術式の見た目⇒針あり」を教えてしまう。
- → **背景を FOCUS フレームにする cut-paste** でショートカットを殺す。
- さらに、貼り付け元は pseudo_frame_v1 と同じフレーム集合なので、
  **同一背景の「針なし（v1）/針あり（本セット）」反実仮想ペア**が自然にできる。

リーク: 背景フレームは generate_frame_pseudo_qa.load_frames()（qa fold0 除外済み）を再利用。
Needle 素材は SAR-RARP50 train 46 本のみ（test 10 本は触らない）。
"""
from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import generate_frame_pseudo_qa as G
from paraphrase_bank import BANK  # noqa: F401  (学習側での言い換えは expK01 で適用)

HERE = Path(__file__).parent
OUT = HERE / "out"
FRAMES_OUT = OUT / "cutpaste_frames"
CROPS = OUT / "rarp_needle_crops"

N_FRAMES = 1500
MAX_Q_PER_FRAME = 3
P_TWO_NEEDLES = 0.25
SCALE_FRAC = (0.05, 0.13)      # 針の貼り付け幅 / フレーム幅
BRIGHT_JITTER = (0.85, 1.15)
MAX_IOU_EXISTING = 0.2
MIN_REGION_BRIGHTNESS = 25.0   # 内視鏡円外の黒帯に貼らない
MAX_TRIES = 12
SEED = 20260825

# needle 中心の family 配分（生成対象は既存 engine と needle 特化の混合）
FAMILY_WEIGHTS = {
    "class_count_needle": 0.22, "inst_count": 0.18, "list_all": 0.16,
    "class_to_quad_needle": 0.10, "positions_all": 0.10, "quad_to_class": 0.08,
    "combination": 0.06, "co_occur_needle": 0.05, "class_diversity": 0.05,
}

log = logging.getLogger("expK00.cutpaste")


def paste_one(img: np.ndarray, rgba: np.ndarray, existing: list[dict],
              rng: random.Random) -> dict | None:
    """1本貼り付けて {cls,cx,cy,bbox} を返す。制約を満たせなければ None。"""
    h, w = img.shape[:2]
    target_w = rng.uniform(*SCALE_FRAC) * w
    scale = target_w / rgba.shape[1]
    crop = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ang = rng.uniform(0, 360)
    ch, cw = crop.shape[:2]
    diag = int(np.ceil((ch ** 2 + cw ** 2) ** 0.5))
    canvas = np.zeros((diag, diag, 4), np.uint8)
    oy, ox = (diag - ch) // 2, (diag - cw) // 2
    canvas[oy:oy + ch, ox:ox + cw] = crop
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), ang, 1.0)
    rot = cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR,
                         borderValue=(0, 0, 0, 0))
    if rng.random() < 0.5:
        rot = rot[:, ::-1]
    a = rot[..., 3]
    ys, xs = np.nonzero(a > 10)
    if len(xs) == 0:
        return None
    rot = rot[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rh, rw = rot.shape[:2]
    if rh >= h - 4 or rw >= w - 4:
        return None
    jit = rng.uniform(*BRIGHT_JITTER)
    for _ in range(MAX_TRIES):
        x0 = rng.randint(2, w - rw - 2)
        y0 = rng.randint(2, h - rh - 2)
        region = img[y0:y0 + rh, x0:x0 + rw]
        m = rot[..., 3:4].astype(np.float32) / 255.0
        if float(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)[m[..., 0] > 0.5].mean()
                 if (m[..., 0] > 0.5).any() else 0) < MIN_REGION_BRIGHTNESS:
            continue
        bbox = (x0, y0, rw, rh)
        if any(_iou(bbox, e["bbox"]) > MAX_IOU_EXISTING for e in existing if "bbox" in e):
            continue
        fg = np.clip(rot[..., :3].astype(np.float32) * jit, 0, 255)
        img[y0:y0 + rh, x0:x0 + rw] = (m * fg + (1 - m) * region.astype(np.float32)).astype(np.uint8)
        ay, ax = np.nonzero(rot[..., 3] > 127)
        return {"cls": "Needle", "cx": x0 + float(ax.mean()), "cy": y0 + float(ay.mean()),
                "area": int((rot[..., 3] > 127).sum()), "bbox": bbox}
    return None


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def needle_qa(fam: str, fr: dict, rng: random.Random) -> tuple[str, str] | None:
    insts = fr["insts"]
    needles = [i for i in insts if i["cls"] == "Needle"]
    w, h = fr["w"], fr["h"]
    if fam == "class_count_needle":
        return G.T_CLASS_COUNT.format(plural="Needles"), str(len(needles))
    if fam == "class_to_quad_needle":
        if len(needles) != 1:
            return None
        n = needles[0]
        return G.T_CLASS_TO_QUAD.format(cls="Needle"), G.quadrant(n["cx"], n["cy"], w, h)
    if fam == "co_occur_needle":
        others = sorted({i["cls"] for i in insts} - {"Needle"})
        if not others:
            return None
        a, b = sorted(["Needle", rng.choice(others)])
        return G.T_CO_OCCUR.format(pa=G.PLURAL[a], pb=G.PLURAL[b]), "yes"
    return G.gen_qa(fam, fr, rng)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FRAMES_OUT.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.FileHandler(OUT / f"cutpaste_{ts}.log"),
                                  logging.StreamHandler()])
    rng = random.Random(SEED)
    crops_meta = json.loads((CROPS / "meta.json").read_text())
    frames = G.load_frames()
    # 既存インスタンスに bbox が無い（centroid のみ）ので、重なり回避用に疑似 bbox を付ける
    for fr in frames:
        for i in fr["insts"]:
            r = max(int(i.get("area", 900) ** 0.5), 20)
            i["bbox"] = (int(i["cx"] - r), int(i["cy"] - r), 2 * r, 2 * r)
    rng.shuffle(frames)
    frames = [f for f in frames if len(f["insts"]) <= 3][:N_FRAMES]
    log.info(f"背景フレーム {len(frames)} / needle 素材 {len(crops_meta)}")

    rows = []
    n_fail = 0
    for fr in frames:
        img = cv2.imread(fr["path"])
        if img is None:
            n_fail += 1
            continue
        n_needles = 2 if rng.random() < P_TWO_NEEDLES else 1
        pasted = []
        for _ in range(n_needles):
            cm = rng.choice(crops_meta)
            rgba = cv2.imread(str(CROPS / cm["file"]), cv2.IMREAD_UNCHANGED)
            if rgba is None:
                continue
            p = paste_one(img, rgba, fr["insts"] + pasted, rng)
            if p:
                p["src"] = cm["file"]
                pasted.append(p)
        if not pasted:
            n_fail += 1
            continue
        out_name = f"{fr['dataset']}_{fr['stem'].split(' ')[0]}_{fr['frame_number']}.jpg"
        cv2.imwrite(str(FRAMES_OUT / out_name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        fr2 = dict(fr)
        fr2["insts"] = fr["insts"] + pasted

        fams = list(FAMILY_WEIGHTS)
        picked = 0
        while fams and picked < MAX_Q_PER_FRAME:
            fam = rng.choices(fams, weights=[FAMILY_WEIGHTS[f] for f in fams])[0]
            fams.remove(fam)
            qa = needle_qa(fam, fr2, rng)
            if qa is None:
                continue
            q, a = qa
            picked += 1
            rows.append({
                "id": f"cp1_{fr['dataset']}_{fr['stem'].split(' ')[0]}_{fr['frame_number']}_{fam}",
                "dataset": fr["dataset"], "video": fr["video"],
                "frame_number": fr["frame_number"], "timestamp": fr["t"],
                "question": q, "answer": a,
                "answer_format": G.FAMILY_FORMAT[fam.replace("_needle", "")],
                "primary_capability": G.FAMILY_CAPABILITY[fam.replace("_needle", "")],
                "family": fam, "track": "FRAME", "generation": "pseudo_cutpaste_v1",
                "frame_path": str(FRAMES_OUT / out_name), "split": fr["split"],
                "instseg_train": fr["instseg_train"], "n_inst": len(fr2["insts"]),
                "n_pasted": len(pasted),
            })
    df = pd.DataFrame(rows)
    assert df["id"].is_unique
    assert not ((df["answer_format"] == "number") & (df["answer"] == "0")).any()
    out = OUT / "pseudo_frame_cutpaste_v1.parquet"
    df.to_parquet(out)
    log.info(f"書き出し {len(df):,} 問 / 合成フレーム {df['frame_path'].nunique():,} "
             f"(失敗 {n_fail}) → {out}")
    log.info("family:\n" + df["family"].value_counts().to_string())
    log.info("n_pasted: " + str(df.groupby('n_pasted')['frame_path'].nunique().to_dict()))


if __name__ == "__main__":
    main()
