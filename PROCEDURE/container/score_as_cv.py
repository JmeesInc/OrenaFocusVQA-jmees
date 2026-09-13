"""コンテナの `answer.json` を CV と**同じ採点器**（eval_seg.py）に掛けて SCORE を出す。

★目的: 提出コンテナが CV 相当の値を出せるかを確認する。
  推論経路が違う（クリップからデコード vs 元動画から時刻指定抽出）ので、
  **同じ問・同じ採点で比べないと差の原因が分からない**。

★eval_seg.py は `responses.json` を読むだけなので**移植しない**。
  `answer.json`（qID / content / latency）に、採点に要るメタ（videoID / question /
  形式 / group など）を QA から補って `responses.json` の形に変換する。

⚠️`time` の後処理はコンテナ側で既に掛かっている。eval_seg.py は `raw` から
  掛け直すので、**`raw` にも後処理済みの文字列を入れて `--time-postproc off` で採点**する
  （on にすると二重適用になる）。

使い方:
    python score_as_cv.py <answer.json> <出力ディレクトリ>
    python ../../workspace/expE01_segproc_baseline/eval_seg.py <出力ディレクトリ> \
        --track PROCEDURE --time-postproc off
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "workspace" / "expE01_segproc_baseline"))
sys.path.insert(0, str(R / "reference" / "src"))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ans_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    import dataset_seg as D                                    # noqa: PLC0415

    idx = D._qa_index("PROCEDURE")
    by_qid = {str(q): (ds, req, ref) for (ds, q), (req, ref) in idx.items()}
    FMT = {"Binary": "binary", "Number": "number", "Percentage": "percentage",
           "FOClass": "fo_class", "OpenEnded": "open_ended",
           "MultipleChoice": "multiple_choice", "Time": "time", "Matching": "matching"}

    ans = json.loads(ans_path.read_text())
    out, miss = [], 0
    for a in ans:
        q = str(a["qID"])
        if q not in by_qid:
            miss += 1
            continue
        ds, req, ref = by_qid[q]
        content = a.get("content", "")
        # ★group / primary は CV と**同じ導出**にする（dataset_seg._group を使う）
        prim = getattr(ref, "primary", None)
        prim_name = getattr(prim, "name", str(prim) if prim else "").lower()
        grp_name = D._group(prim_name)
        out.append({
            "qID": q, "uid": f"{ds}:{q}", "dataset": ds, "videoID": req.videoID,
            "content": content,
            "raw": content,                    # ★後処理はコンテナ側で適用済み
            "latency": float(a.get("latency", 0.0)),
            # ★eval_seg.py の統計出力が参照する（無いと KeyError）
            "n_input_tokens": int(a.get("n_input_tokens", 0)),
            "n_frames": int(a.get("n_frames", 0)),
            "fmt": FMT.get(type(ref.format).__name__, "?"),
            # ★eval_seg.py は group / primary も要求する（無いと KeyError で落ちる）
            "primary": prim_name, "group": grp_name,
            "question": str(req.question),
            "start_time": float(req.start_time), "end_time": float(req.end_time),
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "responses.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (out_dir / "meta.json").write_text(json.dumps(
        {"source": str(ans_path), "n": len(out), "note": "container answer.json → responses.json"},
        ensure_ascii=False, indent=1))
    print(f"{len(out)} 件を書き出し（QA に無い qID: {miss}）→ {out_dir/'responses.json'}")
    print("次: eval_seg.py <dir> --track PROCEDURE --time-postproc off")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
