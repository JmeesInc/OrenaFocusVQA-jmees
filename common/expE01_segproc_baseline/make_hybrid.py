r"""形式ごとに「VLM の回答」と「映像を見ない prior」を振り分けたハイブリッドを作る.

## 根拠（matched + McNemar, SEGMENT fold0 val N=3925）
zero-shot は全体では prior に勝つ（0.4078 vs 0.3360）が、**形式によっては負けている**:

| 形式 | N | prior | zero-shot | 改善/悪化 | McNemar p |
|---|---|---|---|---|---|
| **number** | 292 | **0.2877** | 0.1815 | 25/56 | **0.0008\*\*\*** |
| **fo_class** | 940 | **0.2819** | 0.2415 | 123/161 | **0.0279\*** |
| multiple_choice | 569 | 0.4236 | 0.3673 | 136/168 | 0.0752 n.s. |
| time | 1574 | 0.0565 | **0.1626** | 236/69 | 0.0000\*\*\* |
| open_ended | 249 | 0.5663 | **0.7590** | 63/15 | 0.0000\*\*\* |

→ **prior が有意に勝つ形式だけ prior に差し替える**。

## ⚠️ この施策の位置づけ
- 「どの形式を差し替えるか」を **val の検定で選んでいる**ので、**val への過学習が入っている**。
  提出前には別 fold で確認すること
- **LoRA を当てると number / fo_class はまさに改善する見込み**（FRAME で fo_class 0.17→0.63）。
  その場合この振り分けは不要になる。**あくまで zero-shot 時点の暫定策**

Usage:
  .venv/bin/python workspace/expE01_segproc_baseline/make_hybrid.py \
     --vlm results/zs_seg_16f448 --prior results/prior_seg \
     --prior-formats number fo_class --out-tag hybrid_zs16f_prior_numfo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--prior-formats", nargs="+", default=["number", "fo_class"],
                    help="この形式だけ prior の回答に差し替える")
    ap.add_argument("--out-tag", required=True)
    a = ap.parse_args()

    vlm = json.loads((HERE / a.vlm / "responses.json").read_text()) \
        if not Path(a.vlm).is_absolute() else json.loads(Path(a.vlm).joinpath("responses.json").read_text())
    pri = json.loads((HERE / a.prior / "responses.json").read_text()) \
        if not Path(a.prior).is_absolute() else json.loads(Path(a.prior).joinpath("responses.json").read_text())
    # ★索引キーは (dataset, qID)。qID 単体は dataset 間で衝突する
    pmap = {(r["dataset"], r["qID"]): r for r in pri}

    swap = set(a.prior_formats)
    out, n_sw, n_miss = [], 0, 0
    for r in vlm:
        r = dict(r)
        if r["fmt"] in swap:
            p = pmap.get((r["dataset"], r["qID"]))
            if p is None:
                n_miss += 1
            else:
                r["content"] = p["content"]
                r["raw"] = p.get("raw", p["content"])
                # latency は VLM 側を残す（実運用でも VLM は動かすので時間は消費している）
                n_sw += 1
        out.append(r)

    d = HERE / "results" / a.out_tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "responses.json").write_text(json.dumps(out, indent=1))
    meta = json.loads((Path(a.vlm) if Path(a.vlm).is_absolute() else HERE / a.vlm)
                      .joinpath("meta.json").read_text())
    meta["adapter"] = f"(hybrid: {a.vlm} + prior for {sorted(swap)})"
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {d}/responses.json  n={len(out)}  差し替え {n_sw} 件  prior欠損 {n_miss}")


if __name__ == "__main__":
    main()
