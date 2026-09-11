# expK00_pseudo_vqa — instseg アノテーションからの擬似 VQA 生成

**目的**: survis-anno の FO instseg アノテーション（2026-08-25 dump）から、公式 FRAME テンプレと
byte 一致の擬似 VQA を生成し、学習データ増強（特に計数=知覚ボトルネックと未ラベル動画の視覚多様性）に使う。
ユーザ提案（2026-08-25）: 擬似QA生成 / 入力プロンプト置換 / insseg 重畳入力 / 他術式データ活用 + **instseg リーク対策**。

## データソース

- `/mnt/data/data4/input/survis-anno/{focus-heico,focus-lapchole}/*_20260825_001`
- heico: 13動画 / 812枚 / 1,126 inst（30秒グリッド疎）
- lapchole: **58動画 / 37,847枚 / 71,965 inst**（1秒バースト密。0818比で激増）
- ★dump 71動画のうち **30動画は qa fold v003 の外＝未ラベル lapchole**（0021〜0246）。
  「動画多様性を増やす道は未ラベル70動画しかない」（claudeSummary）に対する実供給。
- instseg_v003（49動画）に対し dump は +25動画（大半が未ラベル群）

## リーク設計

- **qa fold v003 の fold0（VQA val）動画のフレームは生成から除外**（5,058枚捨て）。
  val 動画の画で学習すると CV が過大評価になるため。
- 各行に `instseg_train`（その動画のアノテーションが instseg_v003 の train に入っているか）を付与。
  → ユーザ提案の分岐「instseg 学習に使った動画は overlay なし入力で学習（GT暗記 overlay を見せない）/
  使っていない動画は overlay あり」を、このフラグで実装できる。
- 既知の残リーク（今回の生成とは別件）: instseg_v003 train ∩ qa fold0 val = 6動画
  （0003/0024-Heico, 0027/0036/0061/0079-LapChole）。overlay/索引系の CV はこの6動画を除外して測る
  （expG00 の leak-aware と同じ扱い）。

## 生成仕様（`generate_frame_pseudo_qa.py`）

- 質問文: 公式 lapchole FRAME train parquet のテンプレを **byte 一致**で使用（12 family、
  属性系 1c/1e の 490 問だけ生成不能）。回答語彙・連結順（アルファベット順）も公式一致。
- family 構成比は公式に一致させる（quota 残数比例の抽選。降順貪欲だと希少 family が枯れる）。
- フレーム間引き: 同一動画内 3 秒間隔（1秒バーストの冗長除去）。1フレーム最大3問。
- 中心座標は **RLE マスク重心**（bbox 中心だと細長い loop/drain でズレる）。
- closest_center は 1位2位の距離差が対角の 3% 未満なら曖昧としてスキップ。
- positions_all の列挙順は象限順（TL→TR→BL→BR）→クラス名順（公式例と整合、judge は LLM なので寛容）。
- **expD14 の教訓を assert で強制**: FRAME の number に GT=0 を作らない（陽性フレームのみ使用なので自然に満たす）。
- co_occur の no は「同フレームに無いクラス」で作る（网羅性仮定に依存）。yes:no は公式比 102:126 に制御。

## 出力（`out/`）

- `pseudo_frame_v1.parquet` — **7,668問 / 2,556フレーム / 64動画**（heico 10 / lapchole 54）
  - split: qa_train 4,680 / **out_of_fold（未ラベル動画）2,988**
  - instseg_train フラグ: True 3,702 / False 3,966
  - 列: id, dataset, video, frame_number, timestamp, question, answer, answer_format,
    primary_capability, family, track, generation, frame_path, split, instseg_train, n_inst
- `stats.md` — family 構成比（公式との比較表）・回答分布
- `qa_cards.png` — 全 family×2 の目視カード（青=マスク重畳/緑=ラベル、正しさ目視確認済み）

## ★アノテーション網羅性の検算（`validate_against_official.py`）

公式 FRAME 問題と同一フレーム（±0.5s）に dump アノテーションがある 317 問で、公式正解と直接照合:

| family | 一致率 |
|---|---|
| class_diversity | **0.895** |
| class_count | 0.731 |
| inst_count | 0.703 |
| list_all/combination | 0.686 |

- 計数のズレは**両方向**（dump−公式 = −3〜+4）。「dump が数え漏らしている」だけでなく
  「公式が visible と見なさない物を dump が拾っている」も相当数。可視性の判定基準差が主因とみられる。
- → **擬似 QA の教師は公式基準に対して ~30% ノイズ**（計数・列挙系）。象限・単一物体系は構造上これより清潔。
- v2 の改善候補: expF26 検出器のカウントとアノテーションが一致するフレームだけに計数問題を絞る
  （agreement filter）。

## 追記 2026-08-25（ユーザ回答を受けた拡張）

### QA 生成方式の決定

- **答え（事実）は必ずアノテーションから機械導出**。LLM に答えを作らせない
  （注釈ノイズ ~30% にさらに幻覚が乗るため）。
- **質問文の言い換えはテンプレ単位**（12 family × 十数変種しかない）なので、
  Gemini API / Qwen3.8-27B を回すまでもなく **Claude Code が直接執筆**した
  → `paraphrase_bank.py`（各 family 先頭要素=公式文、意味不変条件をヘッダに明記。
  multiple_choice の選択肢列挙と positions_all の形式規定文は一言一句固定）。
  適用は学習側（expK01 の dataset で `sample_question(family, rng, paraphrase_prob)`）。
- LLM 生成が必要になるのは**注釈から導出できない属性系（1c/1e）を視覚 LLM に作らせる場合のみ**。
  answer 検証ループ（別モデルで答え合わせ）とセットで v2 以降。

### RARP cut-paste（`harvest_rarp_needles.py` → `generate_cutpaste_qa.py`）

- SAR-RARP50 train 46本の 1Hz マスク（class4=Suturing needle）から **827 crop** を収穫
  （closing(15px) 後 1 CC のフレームのみ＝分断/複数針の曖昧さ除外、10s 間引き、RGBA feather）
- **FOCUS 注釈フレーム（pseudo_v1 と同一集合）へ貼り付け**て QA 生成:
  **4,473問 / 合成 1,491 フレーム**（1本貼り 1,112 / 2本貼り 379）→ `pseudo_frame_cutpaste_v1.parquet`
- 設計意図: ①針の「術式の見た目⇒針あり」ショートカットを背景差し替えで殺す
  ②**同一背景の針なし(v1)/針あり(cutpaste) 反実仮想ペア**になる ③貼り付け後の全インスタンスが
  既知なので inst_count / list_all / positions_all も汚染なく生成できる
- 制約: 貼り付け位置は輝度>25（黒帯回避）・既存インスタンスと IoU<0.2・幅 5〜13%・回転/反転/輝度jitter
- `cutpaste_cards.png` で目視確認済み。v2 候補: FOCUS 側のボケに合わせる軽い blur マッチング
- ⚠️ RARP **native** フレームからの QA は v1 では見送り
  （sponge 等が未ラベルなので存在主張系が作れず、「針は常にある」prior を教えるリスクが残るため。
  作るなら class_to_quad 限定）

### insseg 重畳の学習入力（ユーザ確認: **LB で向上済み**・joint 学習方針）

- 推論側は v007 で実装済み（instseg 重畳 dual, leak-aware CV +0.0081, D/C/D 採用）。
  **学習側も同じ描画実装・同じ detector（v007 は expF23_m2f_cvsinit_ps1sN）を使う**こと
  （学習と推論の入力分布を揃える。学習だけ GT マスク重畳にするのは禁物）。
- **リーク分岐（本 parquet の `instseg_train` フラグで実装）**:
  - instseg_v003 train の動画 → **重畳なし arm のみ**（detector 予測が GT 暗記で綺麗すぎるため）
  - それ以外（未ラベル30本含む）→ **重畳あり/なしの両 arm**（同一質問を2サンプル化）
- 事前計算: detector を qa_train 実フレーム + 擬似フレーム（v1 2,556 + cutpaste 1,491）に推論して
  overlay 画像をキャッシュ（M2F 実測 6.8fps → 1万枚 ≈ 25分, dl2 4090）

## 未実装 / 次の一手

- [ ] **expK01**: expE10 レプリカ + {pseudo_v1 + cutpaste_v1 + paraphrase} 混入の A/B。
      dataset 側に frame_path 直読み経路 + paraphrase 適用（`sample_question`）を追加
- [ ] **expK02**: overlay joint 学習（上記リーク分岐）。detector overlay の事前計算スクリプトから
- [ ] expK00 v2: 計数問題の agreement filter（expF26 検出器と注釈が一致するフレームに限定）
- [x] SEGMENT 向け擬似 QA → **SAM3 は不要と判明**し FRAME 注釈だけで 3,742問を生成（下記）。
      time 系は実測で棄却（`last_visible` 一致率 0.000）
- [x] 質問パラフレーズ（paraphrase_bank.py, Claude 直筆）/ SAR-RARP50 cut-paste（4,473問）

## SEGMENT 擬似 QA（`pseudo_segment_v1`, 2026-08-26）— ★SAM3 は使わない

expK01S が負けた原因は「擬似が FRAME 形式しかなく混入比が崩れた」ことだったので、
**SEGMENT 形式の擬似 QA** を作った。ユーザ指摘のとおり **SAM3 疑似ラベルは不要**で、
**FRAME 注釈だけ**から作れる。

### 設計の核: 「クリップの証拠 = 我々が渡すフレーム」

擬似クリップの入力フレームを **注釈フレームそのもの**に限定する。
するとモデルが見る証拠と我々が知る事実が一致し、「この動画に何が写っているか」は**厳密**になる
（アノテータは各フレームの全 FO を塗るので、その瞬間の集合は完全）。
窓 = 同一動画の注釈イベント列から取った duration<=300s（SEGMENT 上限）の連続部分列、
stride 1 イベントのスライディング。バースト（<=2s 間隔）は代表1枚に畳む。

★**SAM3 を使わなかった理由**（検討して棄却）: GT シード ±15s しか埋めず、
**シードされた個体しか追えない**（窓の途中で入る物体が見えない）。score ゲートで早期に切れる
個体もあり、「消失時刻」の教師にできない。一方 GT 注釈は瞬間の集合が完全。
→ 完全性が要る用途では GT 注釈のほうが強い。（RLE デコードのコストも無駄になる）

### ★★★時刻系は作らない（実測で棄却）

| 検証 | 結果 |
|---|---|
| 公式 GT との一致率 `last_visible` | **0.000（0/6）** |
| 公式 GT との一致率 `first_visible` | **0.217（5/23）** |
| 誤差の符号中央（last）| **−16s** ＝ 我々の答えは**常に早すぎる** |
| 論理的に保証できる条件B（隣接イベントで不在確認、間隔<=許容）| 全 995 窓で **first 32 / last 5 問**しか成立しない |
| 条件A（クリップ端で見えている→答えは端）| 1,215 問取れるが、**公式の答えは 80〜95% がクリップ中間**（端は last 13% / first 5%）|

機序: 注釈イベント間隔は median ~40s なのに、`time` の許容は `min(5, 1+dur*4/360)` で
**300s クリップでも ±4.3s** しかない。イベント間に本当の消失があると必ず外す。
条件Aばかり作ると「端を答える」退化を教えて temporal_grounding を壊しかねない。
→ **時刻系は密注釈が取れるまで作らない**。テンプレと `safe_boundary_time()` は v2 用に残置。

### 生成した family と教師品質（公式 GT との一致率, N=258 の被覆問で実測）

| family | 生成数 | 公式一致率 |
|---|---|---|
| classes_between | 1,446 | 0.671 (49/73) |
| also_appear | 611 | —（公式に対応問なし）|
| first_quad | 256 | **0.778 (14/18)** |
| nth_unique | 244 | **0.750 (24/32)** |
| classes_in_video | 243 | 0.592 (42/71) |
| positions_all | 242 | —（FRAME 側で検証済み）|
| last_quad | 238 | **0.720 (18/25)** |
| quad_not_populated | 194 | 0.667 (2/3) |
| quad_populated | 193 | 0.500 (1/2) |
| single_in_video | 75 | **1.000 (7/7)** |

**合計 3,742問 / 窓 1,684 / 動画 64**（qa_train 2,249 / 未ラベル動画 1,493）。
形式: fo_class 2,008 / multiple_choice 881 / binary 611 / open_ended 242。
1問あたり median 9 フレーム（duration median 270s）。

★**狙いが K01S の傷と一致している**: K01S で有意に落ちたのは
`fo_class`(−0.0298\*\*) と `open_ended`(−0.0442\*) ＝ SEGMENT 形式の同定・記述で、
本セットはまさにその形式（1a/1b/1d）を SEGMENT 入力で教える。`time` には触らない。

### 実装

- `generate_segment_pseudo_qa.py` — 窓生成 + 10 family。公式テンプレ byte 一致。
  時刻回答がクリップ外に出ないことを assert（v1 では時刻系そのものを出さない）
- `validate_segment_pseudo.py` — (A) 公式 GT 一致率 / (B) 疎化バイアス（密区間が1窓しか無く測定不能）
- `pack_pseudo_frames.py` — SEGMENT は `frame_paths`(JSON 配列) を畳んで 768px 化（2,519枚）
- `expE01_segproc_baseline/pseudo_data.py` — **マルチフレーム対応**（track で単一/複数を分岐）
- `make_segment_cards.py` — クリップ全フレームを帯で並べたカード（`segment_qa_cards.png`）で目視確認済み

---

## 擬似 QA v2 — 0828 dump（新規 41 動画）で再生成（2026-08-28）

### 生成

```bash
python generate_frame_pseudo_qa.py                      # 既定が v2 / 0828 dump / instseg_v004
python generate_segment_pseudo_qa.py
python pack_pseudo_frames.py   --version v2             # packed/f2, packed/sg2 に 768px JPEG
python make_cards.py           --version v2             # out/qa_cards_v2.png
python make_segment_cards.py   --version v2             # out/segment_qa_cards_v2.png
python validate_against_official.py --dump .../focus-lapchole_20260828_001
python validate_segment_pseudo.py
```

CLI で dump / version / instseg fold を差し替えられるようにした（v1 の再生成も可能）。
**v1 の parquet と `packed/{v1,cp,sg}` は触っていない**（expK04 が参照中のため）。

### 結果

| | v1 | **v2** | 新規動画ぶん |
|---|---|---|---|
| FRAME | 7,668問 / 64動画 / 2,556フレーム | **11,433問 / 104動画 / 3,811フレーム** | +40動画 **3,765問** |
| SEGMENT | 3,854問 / 62動画 | **4,252問 / 98動画 / 窓 1,712** | +37動画 **1,446問** |

- FRAME split: qa_train 5,499 / out_of_fold 5,934（未ラベル動画）
- SEGMENT split: qa_train 1,987 / out_of_fold 2,265
- 形式（FRAME v2）: fo_class 42.3 / **number 40.9** / binary 8.3 / multiple_choice 4.5 / open_ended 3.9 %
- 形式（SEGMENT v2）: fo_class 48.9 / multiple_choice 25.2 / binary 13.7 / **number 6.8** / open_ended 5.4 %

→ **K04 の `format_mix`（FRAME number 44%→31%）は v2 でもそのまま必要**（構成比は v1 と同じ）。
SEGMENT の number 6.8% は公式 7% にほぼ一致（v1 で追加した `max_at_once` / `class_count_video` 由来）。

### ★網羅性は注釈が 2 倍になっても変わらない

公式 FRAME train と同一フレーム（±0.5s）で直接照合（`validate_against_official.py`）:

| family | 0825 dump | **0828 dump** |
|---|---|---|
| 一致した公式問 | 317 | **351** |
| class_diversity | 0.895 | **0.900** |
| class_count | 0.731 | **0.743** |
| inst_count | 0.703 | **0.735** |
| list_all/combination | 0.686 | **0.685** |

差は依然**両方向**（dump−公式 = −3〜+4）。→ v1 の結論
「**ズレの正体は数え漏れではなく『可視』判定基準の違い**」が**量では動かない**ことが確定した。
計数・列挙系の教師ノイズ ~25〜30% は、注釈を増やしても消えない性質のもの。

SEGMENT 側（`validate_segment_pseudo.py`, 被覆 267 問）も v1 とほぼ同値:
classes_between 0.667 / classes_in_video 0.589 / nth_unique 0.758 / first_quad 0.778 /
last_quad 0.704 / single_in_video 1.000。
⚠️ 疎化バイアス（B）は**密窓が 3 個しか取れず**依然測定不能（密注釈動画 0027/0036 が
VQA val で除外されるため）。

### ★⚠️ `instseg_train` フラグが v2 では全行 True になった

`instseg_v004` は fold0 以外を全部 instseg の train に使うので、
**擬似 QA の生成元動画は定義上いつも instseg train** になる
（v1 は instseg_v003 が 49 動画しか無かったので False が 3,966 行あった）。

→ 「instseg 学習に使った動画は overlay なしアーム / 使っていない動画は overlay ありアーム」
という重畳 joint 学習の分岐は、**擬似 QA 側では成立しなくなった**。
実 QA 側は instseg 注釈の無い 86 動画が残るのでそちらで賄える。
両立させるなら instseg を 2 分割して cross-fit で out-of-fold 予測を作る（未実施）。

### 目視

`out/qa_cards_v2.png`（12 family × 2、青=マスク重畳/緑=ラベル）と
`out/segment_qa_cards_v2.png`（クリップ全フレームの帯）で確認済み。
新規動画（out_of_fold）が各 family に混ざっていること、`max_at_once` / `class_count_video`
（number）が正しく生成されていることを確認した。
