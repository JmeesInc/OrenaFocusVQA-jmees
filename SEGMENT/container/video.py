r"""クリップ mp4 から必要な時刻のフレームだけ取り出す.

## 本番クリップの仕様（公式提出テンプレート `procedure-algorithm/inference.py` の docstring）
- H.264 MP4 / **厳密に 5 fps**（1フレーム 0.2s）/ 高さは最大 **576px**、幅は元動画のアスペクト比
  （**固定と仮定してはいけない**）/ **キーフレーム 5 秒ごと**
- **窓に切り出し済み**。`start_time` へシークしてはいけない（クリップ先頭 = start_time）
- **デコード時間は latency 予算に入る**

## なぜ ffmpeg ストリーミングなのか（2026-08-09 実測）
1200s クリップから 241 枚を取る比較:

| 方法 | 実測 |
|---|---|
| decord `get_batch` `num_threads=1`（テンプレ既定）| 29.1 s |
| decord `get_batch` `num_threads=4` | 9.0 s |
| **ffmpeg `-vf fps=..` ストリーミング** | **2.7 s** |

**ランダムアクセスよりストリーミングが速い**（全フレームのデコードでも seek より安い）。
SEGMENT クリップは最長 300s = 最大 1500 フレームなので、全走査しても十分収まる。

## サンプリング刻みとの噛み合わせ
SEGMENT の格子は 1.0s（`sampling.SEGMENT_GRID_S`）で、クリップは 5fps。
`fps=1` フィルタで**クリップ相対の秒ごとに1枚**取り出せば、要求時刻は必ず整数秒なので
そのままインデックスで引ける。
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)


def find_clip(input_dir: Path, qid: str) -> Path | None:
    """`/input` 配下から qID のクリップを探す。

    ★公式テンプレは `plain/<qID>.mp4` と `overlayed/<qID>.mp4` を置く。
      学習・CV は**オーバレイ無しの映像**で行っている（時刻はテキストで与える）ので
      `plain` を優先する。ディレクトリ名の差異で落ちないよう探索順を持たせる。
    """
    for rel in (f"plain/{qid}.mp4", f"{qid}.mp4", f"videos/{qid}.mp4",
                f"clips/{qid}.mp4", f"overlayed/{qid}.mp4"):
        p = input_dir / rel
        if p.exists():
            return p
    hits = sorted(input_dir.rglob(f"{qid}.*"))
    hits = [h for h in hits if h.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov")]
    return hits[0] if hits else None


def extract_frames(clip: Path, rel_times: list[float], width: int,
                   fps: float = 1.0) -> dict[float, Image.Image]:
    """クリップ相対秒 → PIL Image の辞書を返す（取れなかった時刻はキーごと欠落）。

    `rel_times` はクリップ先頭からの秒。`fps` の格子で一括デコードし、最も近い枚を割り当てる。
    """
    if not rel_times:
        return {}
    tmp = Path(tempfile.mkdtemp(prefix="clip_"))
    try:
        # ★`-vsync 0` で入力の間引きに任せる。`scale=W:-2` は学習時の ffmpeg と同一
        #   （幅を合わせ、高さは偶数に丸める）。
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(clip),
               "-vf", f"fps={fps},scale={width}:-2", "-vsync", "0",
               "-q:v", "3", str(tmp / "f_%06d.jpg")]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            log.warning("ffmpeg failed on %s: %s", clip.name,
                        r.stderr.decode("utf-8", "replace")[:300])
        files = sorted(tmp.glob("f_*.jpg"))
        if not files:
            return {}
        # 出力 i 枚目（1-origin）の時刻は (i-1)/fps 秒
        out: dict[float, Image.Image] = {}
        for t in rel_times:
            idx = int(round(t * fps))
            idx = min(max(idx, 0), len(files) - 1)
            with Image.open(files[idx]) as im:
                out[t] = im.convert("RGB")
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
