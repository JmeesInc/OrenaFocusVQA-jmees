"""instseg（Mask2Former）の検出を VQA 入力へ渡すための **描画バリアント**.

expG00 の `m2f_hint.py` を置き換える位置づけ。2026-09-01 の見直しで直した点:

1. **クラス名を公式表記に正規化する**。旧実装は画像に `Silicon_Loop` / `Specimen_Bag` /
   `sponge` と焼いていたが、回答語彙（`prompts_seg.py:FO_CLASSES`）は
   `Silicone Loop` / `Specimen Bag` / `Sponge` で、**綴りまで違う**（Silicon vs Silicone）。
   `ALIASES` は text 経路にしか当たっていなかった。
2. **描画は「モデルが最終的に見るサイズ」で行う**。旧実装（コンテナ）は原寸
   (heico 960x540 / lapchole 1280x720) に描いてから 768px へ縮めていたので、
   **同じ設定でも文字の大きさが CV とコンテナで違い、heico と lapchole でも違った**
   （0.8倍 vs 0.6倍）。ここでは先に `out_width` へ縮めてから描く。
3. **検出0件なら重畳画像を作らない**（`render()` が None を返す）。旧実装は原画像と
   同一の画像を2枚目に足したうえで「2枚目には検出が描いてある」と説明していた。
   expG00 §39 の層別で **空の重畳は −0.0429**。

★クラス名・ラベル index は **ckpt の `config.json:id2label` から読む**（ハードコードしない）。
  expF38(8クラス) / expF40(7クラス, Gallstone なし) / expF39(clip 専用) が混在するため。

バリアント（`VARIANTS`）:
  r0  現行規約（青塗り35% + 太輪郭 + 緑のクラス名）を 1. 2. で直したもの
  r1  r0 + **インスタンス番号**（`Clip 1` `Clip 2` …）。計数の支援
  r2  **クラス別の色分け**・画像に文字を描かず、色の凡例をテキストで渡す
  r3  **2段 conf**（>=hi は塗り+太輪郭+ラベル / lo..hi は細い輪郭のみ + `?`）
  r4  ★**2026-09-02 以降の既定**（ユーザ指示）。検出器が持つ情報を落とさずに渡す:
      **クラスごとに色** + **インスタンス番号** + **confidence の数値**。
      色↔クラスの対応はテキスト側の凡例でも渡す。
      ⚠️ **これ以降に学習を始めるモデルは必ず r4（または同等の SEGMENT/PROCEDURE 版）を使う。**
         r0-r3 は既に学習済みのモデルとの互換のためだけに残してある。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

log = logging.getLogger("expM00.overlay")

# 内部クラス名 → 公式 FO 表記（`reference/src/focus/assets/FO_definitions.txt`）
OFFICIAL = {
    "sponge": "Sponge",
    "clip": "Clip",
    "Specimen_Bag": "Specimen Bag",
    "Silicon_Loop": "Silicone Loop",
    "External_Drain": "External Drain",
    "Needle": "Needle",
    "Gallstone": "Gallstone",
    "Specimen": "Specimen",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FILL_BGR = (255, 40, 0)      # 青（CLAUDE.md の規約色。r0/r1/r3 はこれ1色）
TEXT_BGR = (0, 255, 0)       # 緑

# r2 のクラス別色（BGR）。**画面に自然に出る色は使わない**（2026-09-01 の目視で確定）:
#   赤・ピンク＝組織/血液 / **シアン・ティール＝器具のシャフトやドレープに実在** /
#   黄＝脂肪 / 白＝ガーゼ・鉗子。→ 青・マゼンタ・緑・紫の系統だけを使う。
CLASS_BGR = {
    "Clip": (255, 40, 0),           # 青（最頻・最小クラスに一番見分けやすい色）
    "Sponge": (255, 0, 220),        # マゼンタ
    "Specimen Bag": (0, 255, 0),    # 緑
    "Silicone Loop": (255, 0, 100),  # 紫
    "External Drain": (180, 255, 0),  # 青緑（シアンより緑寄り）
    "Needle": (255, 120, 255),      # 明るいマゼンタ
    "Specimen": (0, 200, 120),      # 深緑
    "Gallstone": (200, 0, 255),     # 赤紫
}

VARIANTS = ("r0", "r1", "r2", "r3", "r4")

# ── ヒント文（user ターンに置く。system は LoRA がほぼ無視する）────────────
# ★r0 の文面は submit/v008 の DUAL_NOTE と**一字一句同じ**にする（既存の検証条件を保つ）。
NOTE_BASE = ("The second image is the same frame with the object detector's foreign-object "
             "masks drawn in blue (green labels). Use the **first (clean) image** to read "
             "the scene; use the second only as a location hint. The detector is imperfect "
             "and often misses instances — do not simply count the blue overlays.")
NOTE_R1 = (NOTE_BASE + " Each detected instance is numbered within its class, largest first; "
           "the class name is written next to instance 1 only (e.g. 'Clip 1', then '2', '3').")
NOTE_R2_HEAD = ("The second image is the same frame with the object detector's foreign-object "
                "masks drawn in colour. Use the **first (clean) image** to read the scene; "
                "use the second only as a location hint. The detector is imperfect and often "
                "misses instances — do not simply count the coloured overlays. Colour key: ")
NOTE_R3 = (NOTE_BASE + " Filled masks with a thick outline and a label are confident "
           "detections; thin unlabelled outlines are low-confidence detections that may be "
           "wrong or may be a different class.")


# ★★r4 = **2026-09-02 以降の既定**（ユーザ指示）。検出器が持つ情報を落とさずに渡す:
#   ① クラスごとに色  ② インスタンス番号  ③ **confidence を数値で**
NOTE_R4_HEAD = (
    "The second image is the same frame with the object detector's foreign-object masks "
    "drawn on it. Each foreign-object class has its own colour, every detected instance is "
    "numbered within its class, and the number after each label is the detector's confidence "
    "(0-1). Use the **first (clean) image** to read the scene; use the second as a location "
    "and identity hint. The detector is imperfect and often misses instances - do not simply "
    "count the overlays, and treat low-confidence instances with suspicion. Colour key: ")

def note_for(variant: str, drawn: list[str]) -> str:
    """バリアントごとのヒント文。`drawn` は実際に描かれた公式クラス名（重複なし・描画順）。"""
    if variant == "r1":
        return NOTE_R1
    if variant == "r3":
        return NOTE_R3
    if variant == "r2":
        key = ", ".join(f"{_colour_word(c)} = {c}" for c in drawn)
        return NOTE_R2_HEAD + key + "."
    if variant == "r4":
        key = ", ".join(f"{_colour_word(c)} = {c}" for c in drawn)
        return NOTE_R4_HEAD + key + "."
    return NOTE_BASE


_COLOUR_WORD = {
    "Clip": "blue", "Sponge": "magenta", "Specimen Bag": "green",
    "Silicone Loop": "purple", "External Drain": "blue-green",
    "Needle": "pink", "Specimen": "dark green", "Gallstone": "violet",
}


def _colour_word(cls: str) -> str:
    return _COLOUR_WORD.get(cls, "blue")


def count_text(dets: list[dict]) -> str:
    """T1 アーム用の**個数だけ**のテキスト（bbox も confidence も書かない）。"""
    per: dict[str, int] = {}
    for d in dets:
        per[d["cls"]] = per.get(d["cls"], 0) + 1
    body = ", ".join(f"{k} x{v}" for k, v in sorted(per.items(), key=lambda kv: (-kv[1], kv[0])))
    return f"Object detector output for this frame: {body}."


# ── 検出器 ──────────────────────────────────────────────────────────────
def _load_decode_instances():
    """expF00 の `postprocess.decode_instances` をファイル直読みで取り込む。

    ★`sys.path` に expF00 を足すと `dataset.py` / `postprocess.py` という一般名が
      expE01 側の同名モジュールと衝突しうる。
    """
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "expF00_fo_instseg/postprocess.py"
    spec = importlib.util.spec_from_file_location("expF00_postprocess", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decode_instances


class M2F:
    """1 つの Mask2Former ckpt。クラス名は `config.json:id2label` から読む。"""

    def __init__(self, ckpt: str | Path, conf: float = 0.5,
                 height: int = 512, width: int = 896, device: str = "cuda"):
        from transformers import Mask2FormerForUniversalSegmentation
        self.decode = _load_decode_instances()
        self.device = torch.device(device)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            str(ckpt)).to(self.device).eval()
        cfg = json.loads((Path(ckpt) / "config.json").read_text())
        id2label = cfg["id2label"]
        self.classes = [id2label[str(i)] for i in range(len(id2label))]
        if self.model.config.num_labels != len(self.classes):
            raise SystemExit(f"num_labels={self.model.config.num_labels} "
                             f"だが id2label は {len(self.classes)} 件")
        self.conf, self.h, self.w = conf, height, width
        # ★expP00(2026-09-04): fp16 を **opt-in** で足す。既定 False ＝ 既存実験は無変更。
        #   `build_fo_timeline` は元から .half() だが重畳側は fp32 のままだった。
        #   dl1(Turing) 実測: F40 6.89→18.20 f/s / F39 2.77→8.21 f/s（batch4 併用）。
        #   検出の一致は verify_fp16.py で確認してから使うこと。
        self.half = False
        log.info("M2F %s classes=%s conf>=%.2f in=%dx%d",
                 ckpt, self.classes, conf, height, width)

    def to_half(self):
        """モデルと入力を fp16 に揃える（呼んだときだけ有効）."""
        self.model = self.model.half()
        self.half = True
        return self

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray, conf: float | None = None) -> list[dict]:
        """BGR 画像 → [{cls(公式表記), mask(uint8 HxW), score}] （score 降順）。"""
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = (cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
             .astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        if self.half:
            x = x.half()
        out = self.model(pixel_values=x)
        if self.half:   # decode は fp32 前提（sigmoid/threshold の数値を既存に合わせる）
            out.class_queries_logits = out.class_queries_logits.float()
            out.masks_queries_logits = out.masks_queries_logits.float()
        thr = self.conf if conf is None else conf
        res = self.decode(out, [(h, w)], threshold=thr)[0]
        dets = [{"cls": OFFICIAL.get(self.classes[i], self.classes[i]),
                 "raw_cls": self.classes[i], "mask": m, "score": float(s)}
                for m, i, s in zip(res["masks"].numpy(), res["labels"].tolist(),
                                   res["scores"].tolist())]
        dets.sort(key=lambda d: -d["score"])
        return dets


class MergedDetector:
    """全クラス検出器 + クラス別 specialist の合成.

    `specialists={"Clip": M2F(...)}` を渡すと、**そのクラスは全クラス側の検出を捨てて
    specialist の検出に差し替える**（expF40 の clip AP50 0.315 → expF39 0.421）。
    """

    def __init__(self, allclass: M2F, specialists: dict[str, M2F] | None = None,
                 conf: float = 0.5, conf_lo: float | None = None,
                 cascade: bool = False):
        self.all = allclass
        self.spec = specialists or {}
        self.conf = conf
        # r3 用の低信頼しきい値。None なら 2 段目を出さない
        self.conf_lo = conf_lo
        # ★expP00(2026-09-04): カスケード。**opt-in**（既定 False ＝ 既存実験は無変更）。
        #   全クラス側が 1 件も出さないフレームでは specialist を省く。
        #   clip 専用(768x1344)が全体の 69.4% を占めるので効く。
        #   dl1 で 600 枚実測: 全クラス0 のフレームは 79.3%、そのうち specialist が
        #   拾うのは **2 枚(0.33%)** だけ → 失う情報はごく僅か、速度は 2.23x。
        self.cascade = cascade
        self.n_skipped = 0

    def predict(self, img_bgr: np.ndarray) -> list[dict]:
        lo = self.conf_lo if self.conf_lo is not None else self.conf
        raw = self.all.predict(img_bgr, conf=lo)
        dets = [d for d in raw if d["cls"] not in self.spec]
        if self.cascade and not raw:
            # 全クラス側が空 → specialist も出さない見込み（実測 99.67%）
            self.n_skipped += 1
            return dets
        for cls, m in self.spec.items():
            dets += [d for d in m.predict(img_bgr, conf=lo) if d["cls"] == cls]
        dets.sort(key=lambda d: -d["score"])
        return dets


# ── 描画 ────────────────────────────────────────────────────────────────
def _scaled(width: int) -> tuple[float, int, int]:
    """(fontScale, 文字太さ, 輪郭太さ)。**最終表示幅**から決める。

    旧実装は原寸に fontScale 0.6 固定で描いてから縮小していたので、
    768px に落とした時点で heico 0.8倍 / lapchole 0.6倍と**表示上の文字サイズが違った**。
    ここでは 1024px で 0.85（≒19px）を基準に線形にする。
    """
    fs = 0.85 * width / 1024.0
    th_text = max(1, int(round(2 * width / 1024.0)))
    th_cnt = max(2, int(round(4 * width / 1024.0)))
    return fs, th_text, th_cnt


def resize_to(img_bgr: np.ndarray, width: int | None) -> np.ndarray:
    if width is None or img_bgr.shape[1] == width:
        return img_bgr
    h = max(2, int(round(img_bgr.shape[0] * width / img_bgr.shape[1] / 2)) * 2)
    return cv2.resize(img_bgr, (width, h), interpolation=cv2.INTER_AREA
                      if width < img_bgr.shape[1] else cv2.INTER_LINEAR)


def _put_label(img, text, x, y, colour, fs, th, chip: bool = False):
    """ラベルを描く。`chip=True` なら**クラス色の塗り箱 + 黒文字**にする（r4）。

    ★2026-09-02 の目視: クラス色の文字をそのまま置くと、Clip=青が暗い術野に埋もれて
      ほとんど読めなかった（緑が既定だったのは「画面に無い色」だから）。
      色でクラスを伝えつつ可読性も保つには、**背景をクラス色で塗って文字を黒にする**のが確実。
    """
    ty = max(int(14 * fs / 0.85) + 2, y - 6)
    if chip:
        (tw, th_px), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.rectangle(img, (x, ty - th_px - 3), (x + tw + 6, ty + base), colour, -1)
        cv2.putText(img, text, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), th, cv2.LINE_AA)
        return
    for col, t in ((0, 0, 0), th + 2), (colour, th):
        cv2.putText(img, text, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, col, t, cv2.LINE_AA)


def render(img_bgr: np.ndarray, dets: list[dict], variant: str = "r0",
           conf: float = 0.5, conf_lo: float = 0.25):
    """重畳画像（PIL RGB）と描画メタを返す。**検出0件なら (None, meta)**。

    `img_bgr` は **既に最終表示サイズ**であること（`resize_to` を先に当てる）。
    """
    from PIL import Image
    if variant not in VARIANTS:
        raise ValueError(variant)
    hi = [d for d in dets if d["score"] >= conf]
    lo = [d for d in dets if conf_lo <= d["score"] < conf] if variant == "r3" else []
    use = hi + lo
    meta = {"n_hi": len(hi), "n_lo": len(lo),
            "labels": [d["cls"] for d in hi], "scores": [round(d["score"], 4) for d in hi]}
    if not use:                       # ★空の重畳は作らない（gate）
        return None, meta

    fs, th_text, th_cnt = _scaled(img_bgr.shape[1])
    base = img_bgr.copy()
    fill = img_bgr.copy()

    # ★インスタンス番号は**クラス内で面積の大きい順**（決定的）。
    #   r1 はクラス名を「1 番」にだけ書き、残りは番号だけにする。
    #   2026-09-01 の目視で、11 個の clip すべてに `Clip` と書くと**ラベル同士が重なって
    #   場面を隠す**ことが分かったため（カード row0）。
    areas = {id(d): int(d["mask"].sum()) for d in use}
    order: dict[int, int] = {}
    seen_cls: dict[str, int] = {}
    for d in sorted(hi, key=lambda x: (x["cls"], -areas[id(x)])):
        seen_cls[d["cls"]] = seen_cls.get(d["cls"], 0) + 1
        order[id(d)] = seen_cls[d["cls"]]

    drawn: list[str] = []
    per: dict[str, int] = {}
    for d in use:
        cls = d["cls"]
        m = d["mask"]
        mb = m.astype(bool)
        if not mb.any():
            continue
        low = d["score"] < conf
        colour = (CLASS_BGR.get(cls, FILL_BGR)
                  if variant in ("r2", "r4") else FILL_BGR)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if low:
            # r3 の低信頼: 塗らず細い輪郭だけ（文字も書かない）
            cv2.drawContours(base, cnts, -1, colour, max(1, th_cnt // 2))
            continue
        fill[mb] = colour
        cv2.drawContours(base, cnts, -1, colour, th_cnt)
        if cls not in drawn:
            drawn.append(cls)
        per[cls] = per.get(cls, 0) + 1
        if variant == "r2":
            continue                        # 画像に文字を描かない（凡例はテキスト側）
        if variant == "r1":
            k = order[id(d)]
            text = f"{cls} 1" if k == 1 else str(k)
        elif variant == "r4":
            # ★クラス色 + 番号 + confidence。クラス名は**そのクラスの最大インスタンス**に
            #   だけ書く（全部に書くと 11 個の clip でラベルが重なって場面を隠す。
            #   2026-09-01 の目視で確認済み）。色と凡例でクラスは判別できる。
            k = order[id(d)]
            # score は先頭 0 を落として短くする（ラベルが長いほど重なる）
            sc = f"{d['score']:.2f}".lstrip("0")
            text = f"{cls} {k} {sc}" if k == 1 else f"{k} {sc}"
        else:
            text = cls
        ys, xs = np.nonzero(m)
        # ★r4 はラベルも**クラス色**で書く（色↔クラスの対応を二重に伝える）
        lab_col = colour if variant == "r4" else TEXT_BGR
        _put_label(base, text, int(xs.min()), int(ys.min()), lab_col, fs, th_text,
                   chip=(variant == "r4"))
    out = cv2.addWeighted(fill, 0.35, base, 0.65, 0)
    meta["drawn"] = drawn
    meta["per_class"] = per
    meta["note"] = note_for(variant, drawn)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)), meta
