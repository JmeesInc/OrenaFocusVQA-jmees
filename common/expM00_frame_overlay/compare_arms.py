"""スクリーニング各アームを **control と matched** で比較する.

`eval_seg.py` が出す `results_merged.csv`（uid 付き・judge 済み）を読んで:

  * pre-evaluation SCORE（公式定義: 5 capability group × ID/OOD の10バケット非加重平均）
  * control との **McNemar**（改善/悪化の対数）
  * 層別: 回答形式 / dataset / **重畳の有無** /
    **「instseg データセットに一度も居ない18動画」vs「検出器の model selection に使った8動画」**

を出す。★[[small-bucket-single-run-is-noise]]: 小さい層の差は n と p を必ず併記する。

    python compare_arms.py --arms control r0 r1 r2 r3 t1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def mcnemar(a: pd.Series, b: pd.Series) -> tuple[int, int, float]:
    """(b が勝った数, a が勝った数, 両側 p)。scipy が無ければ二項検定を自前で。"""
    win = int(((~a) & b).sum())
    lose = int((a & (~b)).sum())
    n = win + lose
    if n == 0:
        return win, lose, 1.0
    from math import comb
    k = min(win, lose)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return win, lose, p


_EV = None


def buckets(df: pd.DataFrame) -> pd.DataFrame:
    """公式 `pre_evaluation_score` が返す **バケット表**（group × ood）をそのまま返す。

    ★SCORE はこの表の accuracy の**非加重平均**。バケットごとの n が大きく違うので、
      「どのバケットが動いたか」を見ないと総合の増減の意味が読めない。
      FRAME の local val は全問 in-distribution なので 2 バケットしか埋まらない
      （本番 Validation は obj_id 747 / agg_id 553 / obj_ood 512 / agg_ood 47 の 4 つ）。
    """
    score(df)          # Evaluator を初期化するため
    _s, b = _EV.pre_evaluation_score(df)
    return b


def score(df: pd.DataFrame) -> float:
    """pre-evaluation SCORE。★**自作しない**（CLAUDE.md / eval_seg.py の注意書き）。

    `results_df.primary` は小文字・`Capability.name` は大文字で、自作の group マップは
    静かに壊れる。公式 `Evaluator.pre_evaluation_score` にそのまま食わせる
    （judge は不要 — 既に `correctness` 列が入っているので採点はしない）。
    """
    global _EV
    if _EV is None:
        import sys
        sys.path.insert(0, str(ROOT / "reference/src"))
        from focus import Evaluator
        _EV = Evaluator()
    s, _buckets = _EV.pre_evaluation_score(df)
    return float(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["control", "r0", "r1", "r2", "r3", "t1"])
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--ref", default="control")
    ap.add_argument("--out", default=str(HERE / "screen_summary.md"))
    a = ap.parse_args()

    # instseg での立場（det_val 8動画 / no_annot 18動画）
    arms_csv = pd.read_csv(HERE / "overlay_arm_v001.csv")
    det = {os.path.splitext(v)[0].strip(): s
           for v, s in zip(arms_csv.videoID, arms_csv.det_status)}

    dfs: dict[str, pd.DataFrame] = {}
    for arm in a.arms:
        # ★アーム名は `screen_<arm>` でも `<arm>`（本学習の eval_* など）でも受ける
        for _d in (Path(a.results) / f"screen_{arm}", Path(a.results) / arm):
            if (_d / "results_merged.csv").exists():
                break
        else:
            print(f"skip {arm}（results_merged.csv が無い）")
            continue
        p = _d / "results_merged.csv"
        d = pd.read_csv(p).set_index("uid")
        r = json.load(open(_d / "responses.json"))
        d["overlay"] = pd.Series({x["uid"]: bool(x.get("overlay")) for x in r})
        d["det_status"] = d.video.map(lambda v: det.get(os.path.splitext(str(v))[0].strip(), "?"))
        d["correct"] = d.correct.astype(bool)
        dfs[arm] = d
    if a.ref not in dfs:
        raise SystemExit(f"基準 {a.ref} が無い")

    ref = dfs[a.ref]
    # ★重畳の有無は **treatment アーム**から取る（control は当然すべて False なので、
    #   control のフラグで層別すると全問が同じ層に落ちて意味を成さない）。
    #   ゲートはフレームだけで決まるので、どの treatment アームでも同じ値になる。
    #   ⚠️ r3 だけは 2段しきい値(0.25)なのでゲート対象が狭い（設計どおり。バグではない）。
    #      層別の基準には **conf 0.5 のアーム**を使い、食い違うアームは件数を出して知らせる。
    _tr = next((d for k, d in dfs.items()
                if k not in (a.ref, "r3") and d.overlay.any()), None)
    if _tr is not None:
        ref = ref.assign(overlay=_tr.overlay.reindex(ref.index))
        for k, d in dfs.items():
            if k == a.ref:
                continue
            n = int((d.overlay.reindex(ref.index) != ref.overlay).sum())
            if n:
                print(f"※ {k} はゲート対象が {n} 問だけ他アームと違う"
                      f"（重畳あり {int(d.overlay.sum())} vs 基準 {int(ref.overlay.sum())}）")
    lines = [f"# expM00 スクリーニング（FRAME val, judge 込み）", "",
             f"基準アーム = `{a.ref}` / N = {len(ref)}", "",
             "## 総合", "",
             "| arm | N | SCORE | Δ vs ctrl | 正答率 | 改善 | 悪化 | p |",
             "|---|---|---|---|---|---|---|---|"]
    s_ref = score(ref)
    for arm, d in dfs.items():
        common = ref.index.intersection(d.index)
        w, l, p = mcnemar(ref.loc[common, "correct"], d.loc[common, "correct"])
        lines.append(f"| {'**' + arm + '**' if arm != a.ref else arm} | {len(d)} | "
                     f"{score(d):.4f} | {score(d) - s_ref:+.4f} | {d.correct.mean():.4f} | "
                     f"{w} | {l} | {p:.4f} |")

    for key, title in [("answer_format", "回答形式別"), ("dataset", "dataset 別"),
                       ("overlay", "重畳の有無別（★ゲート層は control と同一入力）"),
                       ("det_status", "検出器から見た動画の立場別")]:
        lines += ["", f"## {title}", "",
                  "| 層 | n | " + " | ".join(f"{x} Δ (改善/悪化, p)" for x in dfs
                                             if x != a.ref) + " |",
                  "|---|---|" + "---|" * (len(dfs) - 1)]
        for v, sub in ref.groupby(key):
            row = [str(v), str(len(sub))]
            for arm, d in dfs.items():
                if arm == a.ref:
                    continue
                common = sub.index.intersection(d.index)
                if not len(common):
                    row.append("-")
                    continue
                w, l, p = mcnemar(ref.loc[common, "correct"], d.loc[common, "correct"])
                delta = d.loc[common, "correct"].mean() - ref.loc[common, "correct"].mean()
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                row.append(f"{delta:+.4f} ({w}/{l}, p={p:.3f}){star}")
            lines.append("| " + " | ".join(row) + " |")

    # ── ★公式バケット別（group × ood）──────────────────────────────
    lines += ["", "## ★公式バケット別（group × ood — SCORE はこの非加重平均）", "",
              "| バケット | n | " + " | ".join(dfs) + " |",
              "|---|---|" + "---|" * len(dfs)]
    bk = {a: buckets(d).set_index(["group", "ood"]) for a, d in dfs.items()}
    ref_b = bk[a_ref := a.ref]
    for key in ref_b.index:
        n = int(ref_b.loc[key, "count"])
        row = [f"{key[0]}{' (OOD)' if key[1] else ''}", str(n)]
        for arm in dfs:
            v = float(bk[arm].loc[key, "accuracy"]) if key in bk[arm].index else float("nan")
            d0 = v - float(ref_b.loc[key, "accuracy"])
            row.append(f"{v:.4f}" if arm == a_ref else f"{v:.4f} ({d0:+.4f})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| **SCORE（非加重平均）** | — | "
                 + " | ".join(f"**{score(d):.4f}**" for d in dfs.values()) + " |")

    lines += ["", "## latency（FRAME 上限 5s）", "",
              "| arm | mean | max | 超過 |", "|---|---|---|---|"]
    for arm, d in dfs.items():
        lines.append(f"| {arm} | {d.latency.mean():.2f}s | {d.latency.max():.2f}s | "
                     f"{int(d.timed_out.sum())} |")

    txt = "\n".join(lines)
    Path(a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
