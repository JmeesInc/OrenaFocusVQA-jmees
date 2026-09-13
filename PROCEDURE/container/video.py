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
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

# showinfo が吐く "pts_time:123.456" を拾う
_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def find_clip(input_dir: Path, qid: str, overlay: bool = False) -> Path | None:
    """`/input` 配下から qID のクリップを探す。

    ★公式テンプレは `plain/<qID>.mp4` と `overlayed/<qID>.mp4` を置く。
      `overlayed` は **hh:mm:ss を映像に焼き込んだ**もの。
    ⚠️学習・CV は **`videos/` の素の映像**から抽出し、時刻は `[hh:mm:ss]` の**テキスト**で
      与えている（overlay は未使用）。したがって overlay に切り替えると
      **学習時と違う入力分布**になる。`time` 形式で効くかは CV で測ってから決めること。
    """
    order = (f"overlayed/{qid}.mp4", f"overlay/{qid}.mp4") if overlay else ()
    for rel in order + (f"plain/{qid}.mp4", f"{qid}.mp4", f"videos/{qid}.mp4",
                        f"clips/{qid}.mp4", f"overlayed/{qid}.mp4"):
        p = input_dir / rel
        if p.exists():
            return p
    hits = sorted(input_dir.rglob(f"{qid}.*"))
    hits = [h for h in hits if h.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov")]
    return hits[0] if hits else None


def extract_frames_keyframes(clip: Path, rel_times: list[float], width: int,
                             grid: float = 5.0) -> dict[float, Image.Image]:
    """★PROCEDURE 用: **キーフレームだけ**をデコードして時刻→画像を返す.

    ## なぜ `fps=1` ではだめか（2026-08-25 のローカル回帰テストで実測）

    `fps=1` は**クリップ全長を全フレームデコード**する。SEGMENT のクリップは最長 300s
    なので問題なかったが、**PROCEDURE は全長（実測 平均 3,507s / 最長 16,189s）**。
    16枚しか要らないのに 16,000 フレーム復号していた。
    結果 **1バッチ 1,044s / 予算 720s = 145% 超過**（枚数を 16 まで落としても超過）。
    ★**枚数を減らしてもデコード費用は減らない**ので、梯子では救えない。

    ## キーフレームで足りる理由

    本番クリップは **5fps・キーフレーム 5 秒ごと**（`-g 25`）。PROCEDURE のサンプリング格子も
    **5.0s**（CV と同一）なので、**キーフレーム集合＝欲しい時刻の集合**になる。
    ⚠️**`-discard nokey` に置き換えてはいけない**（2026-09-01 に検証して却下）。
      速度は 5.2 倍（5.88s → 1.14s）だが **取れるフレームが変わる**: 枚数は同じ 709 でも
      616/709 で時刻がズレ（最大 9.4 秒）、ffprobe のキーフレーム時刻と一致しなくなる。
      demuxer 側の「キーパケット」判定がデコーダ側と食い違うため。時刻ラベルが狂うので不可。
    `-skip_frame nokey` は P/B フレームの復号を省くので、全長 3,507s なら
    3,507 枚 → **702 枚**、しかも1枚あたりも安い（実測 ~90 kf/s）。
    ★repo 実測でも `ffmpeg -skip_frame nokey` 2.7s < `decord get_batch` 29.1s
      ＝**ランダムアクセスよりストリーミングが速い**。
    """
    if not rel_times:
        return {}
    tmp = Path(tempfile.mkdtemp(prefix="clipkf_"))
    try:
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-skip_frame", "nokey", "-i", str(clip),
               "-vf", f"scale={width}:-2", "-vsync", "0", "-q:v", "3",
               str(tmp / "f_%06d.jpg")]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            log.warning("ffmpeg(keyframe) failed on %s: %s", clip.name,
                        r.stderr.decode("utf-8", "replace")[:300])
        files = sorted(tmp.glob("f_*.jpg"))
        if not files:
            return {}
        # 出力 i 枚目（1-origin）の時刻 ≒ (i-1)*grid 秒
        out: dict[float, Image.Image] = {}
        for t in rel_times:
            idx = min(max(int(round(t / grid)), 0), len(files) - 1)
            with Image.open(files[idx]) as im:
                out[t] = im.convert("RGB")
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def keyframe_times(clip: Path) -> list[float]:
    """★クリップの**実際のキーフレーム時刻**を ffprobe で取る（秒・クリップ相対）。

    ⚠️「キーフレームは 5 秒ごと」という仕様の記述を信用してはいけない。
      ローカルのテストクリップ（本番と同じ ffmpeg 設定で生成）を実測すると
      **中央 2.2〜2.8 秒の不規則**（最小 0.40s / 最大 5.00s）だった。
      `-g 25` を指定してもシーンチェンジ検出で余分な I フレームが入るため。
      i 枚目 = i*5s と仮定すると **最大 972 秒（16 分）ズレる**（2026-09-01 に実測）。
    """
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
         "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(clip)],
        capture_output=True)
    if r.returncode != 0:
        log.warning("ffprobe failed on %s: %s", clip.name, r.stderr.decode("utf-8", "replace")[:200])
        return []
    out = []
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        v = line.strip().rstrip(",")
        if not v:
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def extract_all_keyframes(clip: Path, width: int, grid: float = 5.0) -> tuple[Path, list[Path]]:
    """★キーフレームを**全部**展開して (一時ディレクトリ, ファイル一覧) を返す。

    索引（Mask2Former）と工程（ConvNeXtV2）と VLM 入力の 3 用途で**同じデコード結果を使い回す**
    ため。`extract_frames_keyframes` は要求時刻ぶんだけ返すので、索引には使えない。
    ⚠️呼び出し側が `shutil.rmtree(tmpdir)` すること（クリップ 1 本で数千枚になる）。

    出力 i 枚目（0-origin）の時刻 ≒ i*grid 秒（本番クリップは 5 秒ごとにキーフレーム）。
    """
    tmp = Path(tempfile.mkdtemp(prefix="clipall_"))
    # ★時刻も**同じ 1 パスで**取る。別途 ffprobe を回すとクリップを 2 回走査することになり、
    #   3 時間動画で +31.4 秒（デコードと同額）かかる（2026-09-01 実測）。
    #   `showinfo` が各出力フレームの pts_time を stderr に出すので、それを拾う。
    # ★スケールしない（`width` は互換のため残すが無視する）。
    #   公式仕様: クリップは **高さ最大 576px・幅は元動画のアスペクト比**で
    #   「1024x576 / 720x576 / 640x360 など**固定解像度を仮定するな**」。
    #   実測でも 1024x576 / 960x540 / 854x480 / 720x576 / 720x480 が混在していた。
    #   幅を決め打ちで揃えると小さいクリップを**引き伸ばす**ことになるので、
    #   素のまま取り出して**用途ごとに縮小**する（索引 896x512 / VLM 幅448 / 工程 224）。
    #   `-an` は音声を無視する指示（公式仕様では音声トラックは無い）。
    cmd = ["ffmpeg", "-v", "info", "-nostdin", "-skip_frame", "nokey", "-i", str(clip),
           "-an", "-vf", "showinfo", "-vsync", "0", "-q:v", "3",
           str(tmp / "f_%06d.jpg")]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        log.warning("ffmpeg(all-keyframe) failed on %s: %s", clip.name,
                    r.stderr.decode("utf-8", "replace")[:300])
    times = [float(m) for m in _PTS_RE.findall(r.stderr.decode("utf-8", "replace"))]
    return tmp, sorted(tmp.glob("f_*.jpg")), times


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
