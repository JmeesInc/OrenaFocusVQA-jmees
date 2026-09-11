"""層化 GroupKFold で fold を切り直す（動画単位グループ + 多基準層化）.

## 設計意図（user 指示 2026-07-28 の優先順位）
1. **dataset比**（heico / lapchole）
2. **bucket比**（capability group × ood）— 特に**希少 capability leaf** の存在保証
3. **FOクラス比** — 希少クラス（Gallstone 等）が fold から消えないこと
4. **回答形式比** — 希少形式の存在保証 ＋ common 形式(number/fo_class)の比率均衡

### なぜこの順で、なぜ必要か（v001 実測）
| 基準 | v001 の最大幅 |
|---|---|
| dataset | 0.72pt（良好）|
| bucket(group) | 6.14pt |
| 希少 capability (SITUS/ATTRIBUTES) | 0.47 / 0.89pt（良好）|
| **FOクラス** | **Silicone loop 11.88pt(3.3倍) / Gallstone は f1,f4 が 0件** |
| 回答形式 | number 5.97 / fo_class 5.41pt |

- **回答形式の 6pt ズレは SCORE を約 0.015 動かす**（number 0.28-0.51 と fo_class 0.68-0.71 の差による）。
  これは比較したい効果量（27B vs 9B で 0.007〜0.030）と同じ桁なので、**fold 間比較の交絡になる**。
- **公式 SCORE はバケット非加重平均**なので bucket の「比率」自体は直接バイアスしないが、
  **バケットが空になると平均から落ちて比較不能**になる（judge ログの `only 2/10 buckets are populated`）。
  よって「比率合わせ」より **「存在保証」** が本質。

## アルゴリズム
動画は分割不可（同一動画が train/val に跨るとリーク）なので **video を原子単位**とし、
1. 各動画の「層化キーごとの出現数ベクトル」を作る
2. 質問数の多い動画から順に、**各 fold の現在の構成が全体構成からどれだけズレるか**（重み付き二乗誤差）
   が最小になる fold へ貪欲に割り当てる
3. 希少キー（全体シェアが `rare_threshold` 未満）は **「0件の fold を作らない」制約**を優先し、
   ペナルティを大きくする

貪欲法は初期順序に依存するので、seed を変えて `--restarts` 回試し、最良解を採用する。

## 使い方
    python workspace/fold/generate_folds_stratified.py --version v003 --restarts 30
    → workspace/fold/v003/folds.csv （既存 v001/v002 は消さない）
"""
from __future__ import annotations

import argparse
import collections
import csv
import logging
import random
import sys
from pathlib import Path

FOLD_DIR = Path(__file__).resolve().parent
ROOT = FOLD_DIR.parents[1]
sys.path.insert(0, str(FOLD_DIR))
sys.path.insert(0, str(ROOT / "reference/src"))

from focus import Capability, FocusConfig, FocusDataset, set_config  # noqa: E402
from focus.enums import DatasetSplit, Track  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("stratfold")
for n in ("httpx", "httpcore", "datasets", "huggingface_hub", "filelock", "fsspec"):
    logging.getLogger(n).setLevel(logging.WARNING)

# 優先順位に対応する重み（大きいほど強く揃える）
W = {"dataset": 100.0, "bucket": 30.0, "leaf": 30.0, "fo": 10.0, "fmt": 5.0}
RARE_BOOST = 6.0        # 希少キーはさらにこの倍率で重くする
RARE_THRESHOLD = 0.05   # 全体シェアがこれ未満なら「希少」扱い

# ★重複動画（2026-08-04 監査, workspace/expA00_eda/duplicate_videos/）。
#   0017=0027 と 0234=0242 は md5 までバイト完全一致、0137≈0183 と 0140≈0158 は同一素材の別エンコード。
#   別 fold に散ると学習フレームがそのまま検証側に出るので、**同一 fold に固定する**。
#   さらに層化上も1動画として数えるため、割当の原子単位を「動画」ではなく「このグループ」にする。
DUP_GROUPS = [
    ["0017 - Laparoscopic Cholecystectomy.mp4", "0027 - Laparoscopic Cholecystectomy.mp4"],
    ["0137 - Laparoscopic Cholecystectomy.mp4", "0183 - Laparoscopic Cholecystectomy.mp4"],
    ["0140 - Laparoscopic Cholecystectomy.mp4", "0158 - Laparoscopic Cholecystectomy.mp4"],
    ["0234 - Laparoscopic Cholecystectomy.mp4", "0242 - Laparoscopic Cholecystectomy.mp4"],
]


def collect(tracks: list[str]) -> tuple[dict, dict]:
    """videoID → Counter(層化キー) と videoID → dataset を作る。"""
    set_config(FocusConfig(root_dir=str(ROOT / "data/focus")))
    vec: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    vds: dict[str, str] = {}
    for tname in tracks:
        for ds in ("heico", "lapchole"):
            d = FocusDataset(ds, DatasetSplit.ALL, Track[tname])
            for req, ref in zip(d.requests, d.references):
                v = req.videoID
                vds[v] = ds
                leaf = getattr(ref.primary, "name", str(ref.primary))
                grp = getattr(getattr(Capability[leaf], "group", None), "name", "?")
                ood = int(bool(getattr(ref, "ood", False)))
                vec[v][("dataset", ds)] += 1
                vec[v][("bucket", f"{grp}|ood{ood}")] += 1
                vec[v][("leaf", leaf)] += 1
                vec[v][("fmt", ref._format)] += 1
                if ref._format == "fo_class":
                    for c in str(ref.answer).split(","):
                        c = c.strip()
                        if c and c.lower() != "none":
                            vec[v][("fo", c)] += 1
    return vec, vds


def build_units(vec: dict) -> tuple[dict, dict[str, list[str]]]:
    """重複動画をまとめて「割当の原子単位」を作る。

    返り値は (unit_vec, unit→動画リスト)。QA を持たない動画（unlabeled 側）は vec に居ないので
    メンバーから自然に落ちる。単独動画はその動画1本だけのユニットになる。
    """
    member: dict[str, list[str]] = {}
    seen: set[str] = set()
    for grp in DUP_GROUPS:
        present = [v for v in grp if v in vec]
        if len(present) < 2:
            continue
        member[present[0]] = present
        seen.update(present)
    for v in vec:
        if v not in seen:
            member[v] = [v]
    unit_vec = {}
    for u, vs in member.items():
        c = collections.Counter()
        for v in vs:
            c.update(vec[v])
        unit_vec[u] = c
    n_merged = sum(len(vs) for vs in member.values() if len(vs) > 1)
    if n_merged:
        log.info(f"重複動画 {n_merged} 本を {sum(1 for vs in member.values() if len(vs) > 1)} ユニットに固定: "
                 + ", ".join("+".join(v.split(" ")[0] for v in vs)
                             for vs in member.values() if len(vs) > 1))
    return unit_vec, member


def assign(vec: dict, k: int, seed: int) -> dict[str, int]:
    """質問数の多い動画から貪欲に、構成ズレが最小の fold へ入れる。"""
    total = collections.Counter()
    for c in vec.values():
        total.update(c)
    # キーごとの重み（希少キーは強化）
    fam_tot = collections.Counter()
    for (fam, _), n in total.items():
        fam_tot[fam] += n
    wt = {}
    for (fam, key), n in total.items():
        share = n / max(fam_tot[fam], 1)
        wt[(fam, key)] = W.get(fam, 1.0) * (RARE_BOOST if share < RARE_THRESHOLD else 1.0)

    vids = sorted(vec, key=lambda v: -sum(vec[v].values()))
    rnd = random.Random(seed)
    # 同数の動画は seed でシャッフル（貪欲の初期依存性を散らす）
    buckets = collections.defaultdict(list)
    for v in vids:
        buckets[sum(vec[v].values())].append(v)
    order = []
    for size in sorted(buckets, reverse=True):
        b = buckets[size][:]
        rnd.shuffle(b)
        order += b

    # ★コストは「シェアのズレ」ではなく **件数の目標(=全体/k)からのズレ** で測る。
    #   シェアで書くと「既に均衡している fold に足すのが常に最小」となり、
    #   全動画が1つの fold に吸い込まれる（実際にそうなった）。
    #   件数ベースなら fold が目標件数に達した時点で他へ流れる。
    cur = [collections.Counter() for _ in range(k)]
    out: dict[str, int] = {}
    tgt = {key: n / k for key, n in total.items()}   # 各 fold が持つべき件数

    def delta_cost(f: int, v: str) -> float:
        """動画 v を fold f に入れたときのコスト増分（相対二乗偏差の重み付き和）。"""
        c = 0.0
        for key, add in vec[v].items():
            t = max(tgt[key], 1e-9)
            before = cur[f].get(key, 0)
            c += wt[key] * (((before + add) - t) ** 2 - (before - t) ** 2) / (t ** 2)
        return c

    for v in order:
        best, best_cost = None, None
        for f in range(k):
            cost = delta_cost(f, v)
            # 希少キーを「まだ1件も持っていない fold」に入れるのは歓迎（存在保証を優先）
            for key, add in vec[v].items():
                fam_share = total[key] / max(fam_tot[key[0]], 1)
                if fam_share < RARE_THRESHOLD and add > 0 and cur[f].get(key, 0) == 0:
                    cost -= wt[key] * 0.5
            if best_cost is None or cost < best_cost:
                best, best_cost = f, cost
        cur[best] += vec[v]
        out[v] = best
    return out


def score_assignment(vec: dict, asg: dict[str, int], k: int) -> float:
    """評価指標: 各層化キーの fold 間シェア幅（max-min）を重み付き合計。小さいほど良い。"""
    per = [collections.Counter() for _ in range(k)]
    for v, f in asg.items():
        per[f] += vec[v]
    total = collections.Counter()
    for c in vec.values():
        total.update(c)
    fam_tot = collections.Counter()
    for (fam, _), n in total.items():
        fam_tot[fam] += n
    s = 0.0
    for (fam, key), n in total.items():
        share = n / max(fam_tot[fam], 1)
        w = W.get(fam, 1.0) * (RARE_BOOST if share < RARE_THRESHOLD else 1.0)
        vals = []
        for f in range(k):
            fam_f = sum(v2 for (f2, _), v2 in per[f].items() if f2 == fam)
            vals.append(per[f].get((fam, key), 0) / max(fam_f, 1))
        s += w * (max(vals) - min(vals))
        if any(per[f].get((fam, key), 0) == 0 for f in range(k)) and share < RARE_THRESHOLD:
            s += w * 0.5   # 希少キーが欠ける fold があるのは重大
    return s


def report(vec: dict, asg: dict[str, int], k: int, title: str) -> None:
    per = [collections.Counter() for _ in range(k)]
    for v, f in asg.items():
        per[f] += vec[v]
    total = collections.Counter()
    for c in vec.values():
        total.update(c)
    fam_tot = collections.Counter()
    for (fam, _), n in total.items():
        fam_tot[fam] += n
    print(f"\n########## {title} ##########")
    for fam in ("dataset", "bucket", "leaf", "fo", "fmt"):
        keys = sorted([kk for (f2, kk) in total if f2 == fam],
                      key=lambda kk: -total[(fam, kk)])
        if not keys:
            continue
        print(f"\n=== {fam} ===")
        print(f"  {'key':30s}{'全体%':>7}" + "".join(f"{'f'+str(i):>8s}" for i in range(k)) + f"{'幅':>7}")
        for kk in keys:
            vals = []
            for f in range(k):
                fam_f = sum(v2 for (f2, _), v2 in per[f].items() if f2 == fam)
                vals.append(100 * per[f].get((fam, kk), 0) / max(fam_f, 1))
            flag = "  ★0件fold有" if min(vals) == 0 else ""
            print(f"  {kk:30s}{100*total[(fam,kk)]/fam_tot[fam]:7.2f}"
                  + "".join(f"{x:8.2f}" for x in vals) + f"{max(vals)-min(vals):7.2f}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v003")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tracks", nargs="+", default=["FRAME"])
    ap.add_argument("--restarts", type=int, default=30)
    ap.add_argument("--compare-with", default="v001", help="比較表示する既存 fold バージョン")
    ap.add_argument("--dry-run", action="store_true", help="CSV を書かず比較だけ表示")
    ap.add_argument("--w", nargs="*", default=[],
                    help="層化重みの上書き 例: --w fo=40 fmt=10（優先順位の調整用）")
    args = ap.parse_args()

    for kv in args.w:
        k_, v_ = kv.split("=")
        W[k_] = float(v_)
    log.info(f"weights={W}")
    vec, vds = collect(args.tracks)
    log.info(f"videos={len(vec)}  questions={sum(sum(c.values()) for c in vec.values())//len(args.tracks) if args.tracks else 0}")

    # 重複動画は1ユニットに畳んで割り当てる（同一 fold に固定 ＋ 層化上も1動画として数える）
    unit_vec, member = build_units(vec)
    log.info(f"割当ユニット数 = {len(unit_vec)}（動画 {len(vec)} 本）")

    def expand(unit_asg: dict[str, int]) -> dict[str, int]:
        return {v: f for u, f in unit_asg.items() for v in member[u]}

    best, best_s = None, None
    for s in range(args.restarts):
        a = expand(assign(unit_vec, args.folds, seed=s))
        sc = score_assignment(vec, a, args.folds)
        if best_s is None or sc < best_s:
            best, best_s, best_seed = a, sc, s
    log.info(f"best seed={best_seed} score={best_s:.4f}（小さいほど均衡）")

    # 既存 fold との比較
    old = {}
    p = FOLD_DIR / args.compare_with / "folds.csv"
    if p.exists():
        for r in csv.DictReader(p.open()):
            old[r["videoID"].strip()] = int(r["fold"])
        old = {v: f for v, f in old.items() if v in vec}
        report(vec, old, args.folds, f"BEFORE: {args.compare_with}")
        log.info(f"{args.compare_with} の均衡スコア = {score_assignment(vec, old, args.folds):.4f}")
    report(vec, best, args.folds, f"AFTER: {args.version}")

    if args.dry_run:
        log.info("dry-run: CSV は書いていない")
        return
    out_dir = FOLD_DIR / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "folds.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["videoID", "dataset", "fold"])
        for v in sorted(best, key=lambda v: (vds[v], v)):
            w.writerow([v, vds[v], best[v]])
    log.info(f"wrote {out_dir/'folds.csv'}")


if __name__ == "__main__":
    main()
