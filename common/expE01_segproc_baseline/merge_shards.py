"""`run_infer.py --shard i/N` が吐いた `<tag>.shardK` を1つの `<tag>` に結合する.

## なぜ分割するのか
1問ずつの推論では GPU が遊ぶが、**32f@768 は 16.1GB/プロセス**必要で 24GB カードに
2プロセスは載らない（VRAM が先に尽きる）。→ **プロセスを増やすのではなく問題を割り**、
GPU ごとに 1 プロセスずつ、別々の 1/N を処理させる。

## 使い方
    python merge_shards.py results/eval_foo            # results/eval_foo.shard{0..N-1} を結合
出力: `results/eval_foo/responses.json` + `meta.json`（採点はこの結合結果に対して行う）

⚠️**必ず件数を検算する**。shard が1つでも欠けたまま採点すると、
未回答が「不正解」として静かに混ざり、スコアを過小評価する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: merge_shards.py <out_dir_without_shard_suffix> [expected_n] [--allow-dup-qid]")
    args = [a for a in sys.argv[1:] if a != "--allow-dup-qid"]
    allow_dup = "--allow-dup-qid" in sys.argv
    base = Path(args[0])
    expected = int(args[1]) if len(args) > 1 else None

    shards = sorted(base.parent.glob(f"{base.name}.shard*"))
    if not shards:
        raise SystemExit(f"shard が見つからない: {base}.shard*")

    responses: list = []
    metas: list = []
    for s in shards:
        rp, mp = s / "responses.json", s / "meta.json"
        if not rp.exists():
            raise SystemExit(f"❌ {s} に responses.json が無い（その shard は失敗している）")
        responses.extend(json.loads(rp.read_text()))
        if mp.exists():
            metas.append(json.loads(mp.read_text()))

    qids = [r["qID"] for r in responses]
    if len(qids) != len(set(qids)):
        # ★★FOCUS の qID は **heico と lapchole を跨いで一意ではない**（2026-09-05 実測）。
        #   SEGMENT val fold0 では qID=2392989 が
        #     lapchole 0059「clips が適用された時刻」 と heico 0024「Needle 挿入時刻」
        #   という**全く別の質問**に付いている。shard 分割とは無関係な**データ側の欠陥**。
        #   これまで表面化しなかったのは SEGMENT の評価が --shard を使っていなかったため。
        #   ★既定は従来どおり停止する（黙って捨てない）。`--allow-dup-qid` を明示したときだけ
        #     **先勝ちで dedupe** する。判定は失う問を明示してから行う。
        import collections
        dup = [q for q, n in collections.Counter(qids).items() if n > 1]
        if not allow_dup:
            raise SystemExit(
                f"❌ qID が重複している（{len(qids)} 件中 {len(set(qids))} ユニーク）: {dup}\n"
                f"   FOCUS は dataset を跨ぐと qID が衝突する。意図した重複なら "
                f"--allow-dup-qid を付けて先勝ち dedupe する")
        seen, kept = set(), []
        for r in responses:
            if r["qID"] in seen:
                print(f"  ⚠️ 重複 qID を捨てる: {r['qID']} content={r.get('content')!r}")
                continue
            seen.add(r["qID"]); kept.append(r)
        print(f"⚠️ ★qID 重複 {dup} を先勝ちで dedupe: {len(responses)} → {len(kept)} 問")
        responses = kept
    if expected is not None and len(responses) != expected:
        raise SystemExit(f"❌ 件数不一致: {len(responses)} != 期待 {expected}")

    base.mkdir(parents=True, exist_ok=True)
    (base / "responses.json").write_text(json.dumps(responses, ensure_ascii=False))
    if metas:
        m = dict(metas[0])
        m["n"] = len(responses)
        m["n_shards"] = len(shards)
        # latency は shard ごとの平均を件数で加重平均する
        tot = sum(x.get("n", 0) for x in metas) or 1
        for k in ("latency_mean", "input_tokens_mean", "n_frames_effective_mean"):
            if all(k in x for x in metas):
                m[k] = sum(x[k] * x.get("n", 0) for x in metas) / tot
        m["latency_max"] = max((x.get("latency_max", 0) for x in metas), default=0)
        (base / "meta.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))

    print(f"✅ {len(shards)} shard を結合 -> {base}  ({len(responses)} 問)")


if __name__ == "__main__":
    main()
