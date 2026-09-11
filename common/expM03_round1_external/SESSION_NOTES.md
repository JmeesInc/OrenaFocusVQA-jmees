# expM03_round1_external — Round 1: 外部 Surgical VQA 混合学習（FRAME）

開始: 2026-09-01
目的: OOD（未知術式）対策。FOCUS 全量 + 擬似 + **expM00_frame_overlay の重畳設計** に
外部 VQA（SurgMLLMBench v1 15,525 / MultiBypass v1 16,000）を「事前学習的」に混ぜる。
検証は LB（ユーザ方針。提出枠は充分）。

## 設計（ユーザと合意済み 2026-09-01）

1. **2段アニーリング**: Stage A = 混合 1ep → Stage B = FOCUS+擬似のみ 1ep
   （既存 `lora.init_from_adapter` の warm start をそのまま使用。B の lr は A の半分 5e-5）
2. **外部は loose に効かせる**: loss_weight 0.5（サンプル別重み）。
   全系列LM loss 化は見送り（外部質問文はテンプレで情報が薄い。SurgAtlas キャプションを
   使う Round 2 で導入検討）
3. **術式文ドロップアウト 0.5**: 外部QAの質問先頭の術式文を確率で除去
   （未知術式名が来ても壊れない+あれば使う、の両立。expE21 の「学習込みで反転」リスク緩和）
4. **expM00 の重畳設計を同梱**: overlay r1（暫定勝者）+ arm ゲート + 擬似v2。
   ⚠️ expM00 のスクリーニング最終決定に `overlay.variant` / `expect_dual` を追随させること

## 実装（フォーク方針）

- `train_lora_ext.py` = expE01/train_lora_seg.py のフォーク（**2026-09-01 01:38 時点 35,210B**）。
  **dataset_seg / pseudo_data / prompts_seg / overlay_attach は expE01 の生きたコードを import**
  （expM00 側の改良を自動追随）。差分は3点のみ:
  ① `data.external` レーン注入（擬似の後・shuffle）
  ② collator が `meta["loss_weight"]` を持ち回り → `WeightedTrainer.compute_loss` が適用。
     **重み全て1.0のバッチは親クラス経路＝既存実験と数値一致**（Stage B も同一挙動）
  ③ probe で loss_weight を pop
- `external_data.py`: parquet ローダ（pseudo_data 準拠、paraphrase 無し・system prompt は
  answer_format から直接構成・術式文ドロップは seed 固定で決定的・フレーム欠落は既定で fail）
- configs: `config_M03A_frame_stageA.yaml` / `config_M03B_frame_stageB.yaml`

## 単体テスト済み（2026-09-01, GPU なし）

- external レーン: SMB 15,525 件ロード / dropout 実測 0.499 / weight 0.5 付与 /
  cholec80 の head `Frame at [00:06:12]`（焼き込みと一致）・時刻不明系は `[00:00:00]`
- MB FRAME 8,000 は frames 抽出中のため欠落扱い（watch_and_extract.sh 完了後に解消）
- py_compile 通過

## SMB parquet の時刻整合（2026-09-01）

`Frame at [hh:mm:ss]` と焼き込みタイムスタンプを一致させるため、SMB parquet に実時刻を追加:
cholec80 = fidx/実fps / autolaparo(phase) = idx-1 / misaw = fidx/30 → **10,833/15,525 行に時刻あり**。
導出不能（autolaparo tool / endovis2018 / mavis = 焼き込み無し）は空 → [00:00:00] 扱い。

## TODO

- [x] MB フレーム 69,042/69,042 完了（FINAL rc=0）→ external 厳格ロード 23,525件・欠落0 ✅
- [x] smoke rc=0（vast上, --train-limit 160。ROOTバグをフォークにも適用修正）→ **本走チェーン起動済み**（instance 49566575, train=55,423, 14.3s/it, A+B≈$10）
- [ ] expM00 スクリーニング最終決定を variant に反映
- [x] SEGMENT 版 config = joint A/B 作成済み（K04ベース、下記）
- [x] vast チェーンスクリプト（xfer_m03.sh / vast_m03.sh）+ instance 49563990 賃借（orena_expM03_frame）
- [ ] 学習後: LB 提出は Round1 vs 現ベストの差分で読む

## joint（SEGMENT 系）config 追加（2026-09-02 03:00）

- `config_M03A_joint_stageA.yaml` / `config_M03B_joint_stageB.yaml`: K04（joint F+S + 擬似
  format_mix）ベース。★**track 比 1:1 を external でも維持**（K01S の教訓）:
  external SEGMENT = MB 8,000 全量 / external FRAME = 23,525 → 8,000 cap
- 見積り: Stage A ~79h/ep on 1×4090 → **DDP 4×4090 で ~20h**。B は ~15h
- 4 config とも yaml パース・external/warm start/lr の整合を検証済み

## DL 事故の教訓（expM01 mb05, 2026-09-02 01:30 検死）

- ★**`curl -C -`（resume）と `-r`（range）は併用禁止**: リトライ時に -C - が range 終端を
  無視して EOF まで読み続け、part が 89GB/42.5GB に暴走した。v2 = resume なし・
  サイズ不一致 part は捨てて丸ごと取り直し、で解決（全 part 一発一致）
- ★ABORT する設計にしたら**再起動係（リトライループ）を必ず同時に置く**（5.5h 空転の実害）
