"""擬似 QA のフレームを貸しGPU 転送用にパックする.

- dump PNG（1280x720 等）と cutpaste JPEG を **幅 768px の JPEG q93** に統一
  （学習キャッシュ `frames_cache/768` と同じ規約。ffmpeg `scale=768:-2` 相当）
- frame_path を packed ルートからの**相対パス**に書き換えた `*_packed.parquet` を出力
  （dl1 と vast でルートが違うため。学習側は `data.pseudo.frames_root` で解決する）
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).parent
OUT = HERE / "out"
PACKED = OUT / "packed"
# sub（packed のサブディレクトリ名）→ parquet。
# ★サブディレクトリはバージョンごとに分ける。同じ sub に別バージョンを重ねると
#   末尾の `packed 枚数 == 参照ユニーク数` の assert が落ちる（残骸が混ざるため）。
SOURCES = {
    "v1": OUT / "pseudo_frame_v1.parquet",
    "cp": OUT / "pseudo_frame_cutpaste_v1.parquet",
}
# SEGMENT 擬似は 1問が複数フレームを持つので `frame_paths`(JSON 配列) を畳んで扱う
MULTI_SOURCES = {"sg": OUT / "pseudo_segment_v1.parquet"}
# dump PNG 由来（frame 番号が動画横断で衝突するので dataset+動画番号を前置する）
PREFIXED = {"v1"}
WIDTH = 768
QUALITY = 93

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("expK00.pack")


def select_version(version: str) -> None:
    """--version v2 のように、パックするバージョンを差し替える。
    v1 の packed は稼働中の学習（expK04 等）が参照しているので**上書きしない**。"""
    global SOURCES, MULTI_SOURCES, PREFIXED
    if version == "v1":
        return
    v = version.lstrip("v")
    SOURCES = {f"f{v}": OUT / f"pseudo_frame_{version}.parquet"}
    MULTI_SOURCES = {f"sg{v}": OUT / f"pseudo_segment_{version}.parquet"}
    PREFIXED = {f"f{v}"}


def main() -> None:
    for sub, pq in MULTI_SOURCES.items():
        import json as _json
        d = PACKED / sub
        d.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(pq)
        srcs = sorted({s for row in df["frame_paths"] for s in _json.loads(row)})
        rel = {}
        for src in srcs:
            sp = Path(src)
            # dump のフレーム番号は動画横断で衝突しうるので dataset+動画番号を前置する
            parts = sp.parts
            vid = parts[-3] if len(parts) >= 3 else sp.parent.name
            name = f"{vid.split(' ')[0]}_{sp.stem}.jpg"
            if (d / name).exists():
                rel[src] = f"{sub}/{name}"
                continue
            im = Image.open(sp).convert("RGB")
            if im.width > WIDTH:
                h = max(2, int(round(im.height * WIDTH / im.width / 2)) * 2)
                im = im.resize((WIDTH, h), Image.BICUBIC)
            im.save(d / name, quality=QUALITY)
            rel[src] = f"{sub}/{name}"
        df["frame_paths"] = df["frame_paths"].map(
            lambda row: _json.dumps([rel[s] for s in _json.loads(row)]))
        out = OUT / (pq.stem + "_packed.parquet")
        df.to_parquet(out)
        n = len(list(d.glob("*.jpg")))
        # ★再生成でフレーム集合が縮むことがある（例: MAX_FRAMES を 24→16 に変更）。
        #   ディレクトリに未使用の残骸があってもよいが、**参照先が全て実在すること**は必須。
        missing = [v for v in rel.values() if not (PACKED / v).exists()]
        assert not missing, f"{sub}: 参照先が無い {len(missing)} 件（例 {missing[:2]}）"
        if n != len(rel):
            log.info(f"{sub}: 未使用の残骸 {n - len(rel)} 枚（参照 {len(rel)} 枚は全て実在）")
        log.info(f"{sub}: {len(df)}問 / {n}枚（1問あたり median "
                 f"{int(df['n_frames'].median())}枚）→ {d}（parquet: {out.name}）")

    for sub, pq in SOURCES.items():
        d = PACKED / sub
        d.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(pq)
        rel = {}
        for src in sorted(df["frame_path"].unique()):
            sp = Path(src)
            name = sp.stem + ".jpg"
            # v1 の dump PNG は動画横断で frame 番号が衝突しうるので dataset+動画番号を前置
            if sub in PREFIXED:
                row = df[df["frame_path"] == src].iloc[0]
                name = f"{row.dataset}_{Path(row.video).stem.split(' ')[0]}_{row.frame_number}.jpg"
            im = Image.open(sp).convert("RGB")
            if im.width > WIDTH:
                h = max(2, int(round(im.height * WIDTH / im.width / 2)) * 2)
                im = im.resize((WIDTH, h), Image.BICUBIC)
            im.save(d / name, quality=QUALITY)
            rel[src] = f"{sub}/{name}"
        df["frame_path"] = df["frame_path"].map(rel)
        assert df["frame_path"].notna().all()
        out = OUT / (pq.stem + "_packed.parquet")
        df.to_parquet(out)
        n = len(list(d.glob("*.jpg")))
        assert n == df["frame_path"].nunique(), f"{sub}: packed {n} != unique {df['frame_path'].nunique()}"
        log.info(f"{sub}: {len(df)}問 / {n}枚 → {d}（parquet: {out.name}）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="v1", help="v1（既定, cutpaste 込み）/ v2 ...")
    select_version(ap.parse_args().version)
    main()
