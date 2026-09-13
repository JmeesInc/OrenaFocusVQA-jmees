"""提出物 answer.json を公式仕様に照らして検証する（提出前チェック）.

チェック項目:
  1. answer.json が focus.Response のリストとして読める
  2. request.json の全 qID に 1:1 で回答がある（欠落・重複・余分なし）
  3. content が str（空でも可＝不正解扱いだが形式違反ではない）
  4. 空回答の割合を警告として表示
  5. 参考: 回答が各 answer format の verify を通るか（GT不要な形式チェック）

Usage: python validate_output.py <request.json> <answer.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    req_path, ans_path = Path(sys.argv[1]), Path(sys.argv[2])
    reqs = json.loads(req_path.read_text())
    ans = json.loads(ans_path.read_text())
    if isinstance(reqs, dict):
        reqs = reqs.get("items", reqs.get("requests", []))
    if isinstance(ans, dict):
        ans = ans.get("items", ans.get("responses", []))

    req_ids = [r["qID"] for r in reqs]
    ans_ids = [a["qID"] for a in ans]
    errs: list[str] = []

    if len(ans_ids) != len(set(ans_ids)):
        dup = [q for q, c in Counter(ans_ids).items() if c > 1]
        errs.append(f"duplicate qIDs in answer.json: {dup[:5]}")
    missing = set(req_ids) - set(ans_ids)
    extra = set(ans_ids) - set(req_ids)
    if missing:
        errs.append(f"missing {len(missing)} qID(s), e.g. {list(missing)[:5]}")
    if extra:
        errs.append(f"unexpected {len(extra)} qID(s), e.g. {list(extra)[:5]}")
    for a in ans:
        if not isinstance(a.get("content", None), str):
            errs.append(f"qID={a.get('qID')}: content is not str ({type(a.get('content'))})")
            break

    n_empty = sum(1 for a in ans if not str(a.get("content", "")).strip())
    print(f"requests={len(req_ids)}  answers={len(ans_ids)}  empty={n_empty} "
          f"({100 * n_empty / max(len(ans_ids), 1):.1f}%)")
    # ★空回答は「公式仕様違反ではない」が、こちらにとっては常に欠陥（不正解確定）。
    #   仕様チェックだけ見て ✓ を信じると OOM や生成打ち切りのバグを見逃す。
    if n_empty:
        errs.append(f"空回答 {n_empty}/{len(ans_ids)} 件 "
                    f"({100 * n_empty / max(len(ans_ids), 1):.1f}%) — 不正解確定なので原因を潰すこと")

    # 参考: 形式 verify（提出仕様違反ではないが、exact採点で落ちる回答を可視化）
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from prompts_seg import detect_format
        from focus.data.formats import get_format_class
        bad = Counter()
        by_fmt = Counter()
        qmap = {r["qID"]: r["question"] for r in reqs}
        for a in ans:
            q = qmap.get(a["qID"], "")
            fmt = detect_format(q)
            by_fmt[fmt] += 1
            if fmt in ("open_ended", "matching", "multiple_choice"):
                continue  # judge 採点なので verify 不要
            try:
                get_format_class(fmt)().verify(str(a["content"]))
            except Exception:
                bad[fmt] += 1
        print("per-format answers:", dict(by_fmt))
        if bad:
            print(f"⚠ exact採点形式で verify を通らない回答: {dict(bad)} "
                  "（不正解になるだけで提出は可能だが、プロンプト/後処理を見直す価値あり）")
    except Exception as e:  # 検証の付加機能なので失敗しても致命的でない
        print(f"(format check skipped: {e})")

    if errs:
        print("\n".join("✗ " + e for e in errs))
        return 1
    print("✓ answer.json は公式仕様を満たしています")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
