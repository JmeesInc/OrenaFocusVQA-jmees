# v017_router — FRAME track（★**バケット別ルーティング + 経路ごとの3本投票**）

**v011_m03_external のフォーク**。v011（LB **0.6260** / 4位）の3本投票を**そのまま経路A**として温存し、
**ID × aggregation の問だけ**を all-data 3本の投票（経路B）へ振り向ける。

## 設計根拠（FRAME LB 最終順位のバケット分解, N: obj_id 747 / agg_id 553 / obj_ood 512 / agg_ood 188）

| 提出 | 構成 | obj_id | agg_id | obj_ood | agg_ood | SCORE |
|---|---|---:|---:|---:|---:|---:|
| **v011** | r=16 / **r1** / train / 3本投票 | **538** | 324 | **262** | **129** | **0.6260**（4位）|
| v014 | r=64 / r4 / train | 515 | 322 | 213 | 113 | 0.5722 |
| v015 | r=16 / r4 / all-data / +SurgAtlas | 515 | **338** | 247 | 118 | 0.6027 |
| （1位）| — | 537 | 342 | 274 | 125 | 0.6343 |

- 1位との差は **agg_id −18問 と obj_ood −12問**だけ。**agg_ood は全参加者中トップ**
- ★**obj_id は「重畳が r4 かどうか」だけで決まる**（rank 4倍でも SurgAtlas 追加でも val 追加でも 515、r1 のときだけ 538）
- ★**v015 が勝つのは agg_id ただ1つ**（+14問）
→ **r1 を保ったまま、agg_id の問だけ all-data 系に投げる**のが実測バケットの組み替えで最良。
  期待値 (0.7202 + 0.6112 + 0.5117 + 0.6862)/4 = **0.6323**

## ルーティング（モデルを走らせる前に、質問文と属性だけで確定する）

```
判定1  ID/OOD    : procedure_type ∈ {Proctocolectomy, Rectal Resection,
                                    Sigmoid Resection, Laparoscopic Cholecystectomy}
判定2  capability: detect_format → number/binary = AGGREGATION / 他 = OBJECT_RECOGNITION
経路A（OOD 全部 + ID×object）  → v011 の3本（実測構成に手を入れない）
経路B（ID×aggregation）        → all-data 3本
```

- FRAME val 実測で **形式 → group の純度は 1.000 / 1.000 / 0.980**、`detect_format` は FRAME で 99.76%
- ローカル 20,000 問での振り分けは **A 58.5% / B 41.5%**（CV のバケット比 58.0:42.0 とほぼ一致）
- ★**誤判定の向きを非対称に倒してある**: `procedure_type` が未知・空なら **OOD 扱い＝経路A**（実測構成）。
  「OOD を ID と誤る」と agg_ood を落とす（−11問相当）が、逆向きは v011 に戻るだけで無害

## メンバー

| 経路 | member | 幅 | 重畳 | 出典 | md5 |
|---|---|---|---|---|---|
| A | m1_m03b_overlay | 1024 | r1 | expM03B | `c0f2795a…` |
| A | m2_m03b_768 | 768 | r1 | expM03B（同一 adapter・解像度違い）| `c0f2795a…` |
| A | m3_v06_r32 | 768 | なし | expV06_frame_r32 | — |
| B | b1_q02b_r1 | 768 | r1 | **expQ02B**（v011 レシピ + all-data）| `7a5b2ba9…` |
| B | b2_q00b_r4 | 768 | **r4** | **expQ00B**（r4 + SurgAtlas + all-data = v015 の中身）| `646acb71…` |
| B | b3_q03b_r1sa | 768 | r1 | **expQ03B**（r1 + SurgAtlas + all-data）| `297a1fd9…` |

★経路Bは **別々に学習した3本**（「同一 adapter 2本の投票は負ける」実測 0.6528 < 単独 0.6581 に従う）。
★重畳キャッシュは**検出結果**を持ち `render(variant=…)` はメンバーごとなので、**r1 と r4 の混在は安全**。

## パス構成（anytime 性質を維持）

**ラウンド単位**で回す: round1 で経路A・Bとも pass1 を終える → **全問に回答がある状態が先に完成**。
round2/3 は純粋な上積みで、予算が尽きて途中で止めても投票メンバーが減るだけ。

## 手順

```bash
bash stage_resources.sh                       # 6 adapter を配置（md5 検算つき）
bash ../build_test_on_dl2.sh v017_router 0    # dl2 でビルド+回帰テスト
BUILD_ID=r1 bash do_save.sh                   # tar.gz
```

**アップロードはユーザー手動**。⚠️ 再提出時は必ず `BUILD_ID` を上げる（Failed でも digest は消費される）。

## 既知の注意

- 回収時に `tokenizer.json`（0バイト）と `processor_config.json`（欠落）が b3 で欠けていたため、
  **b1 から補完**した（両ファイルは b1/b2 で md5 完全一致 = ベースモデルのトークナイザのコピーで学習に依らない）
