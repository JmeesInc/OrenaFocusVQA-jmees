# expQ00_frame_alldata — FRAME: val 込み all-data 学習

開始: 2026-09-04（ユーザ指示）

## 狙い

CV で選ぶ段階を終えて、**fold0 val 4,008 問も学習に入れた提出用モデル**を作る。
`train_part: all` なので **CV は測れない。判定は LB のみ。**

## Stage A の構成（ユーザ確定 2026-09-04）

| 項目 | 値 | 由来 |
|---|---|---|
| base | Qwen/Qwen3.5-9B（4bit NF4 / bf16 compute） | FRAME で 27B/32B/Gemma31B に 5 連勝 |
| LoRA | r=64 / α=128 | expN03（CV 0.6780 = FRAME 単独ベスト）と同じ容量 |
| 重畳 | r4 / conf 0.25 / `expM00_frame_overlay/cache_r4` | 2026-09-02 のユーザ規約（クラス色+個体番号+conf） |
| 解像度 | 768px（学習＝推論） | 1024 は expD25 で無効と確定 |
| lr | 1e-4 cosine / warmup 0.03 / 1 epoch | base からの 1ep なので expN02 と同じ |
| データ | FOCUS **20,000**（train 15,992 + **val 4,008**）+ 擬似 15,906 + 外部 **40,170** = **76,076** | — |
| 外部 | SurgMLLMBench 15,525 + MultiBypass 8,000 + SurgAtlas 16,645、`loss_weight 0.5` / 術式文 50% ドロップ | ユーザ指示「外部 all」 |
| step | 4,755（bs1 × accum16）≈ 20h（1×4090, 15.1 s/it） | — |
| eval | **なし**（`eval_steps: 0`）→ 最終 step を採用 | held-out が無く best 選択が無意味（[[load-best-picks-noise-argmin]]） |

Stage B（FOCUS+擬似のみで焼き付け）を回すかは他実験の結果待ちで**未定**。
このフォルダは A のみを規定する。

## 重畳の「あり/なし」（2026-09-04 実測。乱択は無い）

FOCUS train+val 20,000 問に対し `attach(cache_r4, r4, arms=arm_csv)` の実測:

| | 件数 | 決まり方 |
|---|---:|---|
| 重畳あり（dual） | **11,032** | DUAL arm かつ検出 1 件以上（train 7,881 + val 3,151） |
| CONTROL arm | 6,095 | 検出器が注釈を学習に使った 36 動画 → **OFF のまま**（ユーザ確定） |
| 検出 0 件 | 2,873 | 空の重畳は expG00 で −0.0429 |
| 欠け | **0** | val 3,087 フレームも含めてレンダ済み |

→ `overlay.expect_dual: 11032` にピン留めした。**train のみの 7,881 とは別の数字**。

参考: arm ゲートを外すと `cache`（dl2 レンダ・全動画）で dual 15,411 / 欠け 0 にできるが、
「検出器が GT を暗記した重畳」を教師にするため**採用しない**（ユーザ確定）。

## expM04 との関係（★重複に注意）

**expQ00A = expM04A の「rank 64 + val 込み」版**。データレーン・lr・EOS 保護は同一。

| | expM04A（走行中） | expQ00A |
|---|---|---|
| LoRA | r=16 / α=32 | **r=64 / α=128** |
| FOCUS | train のみ 15,992 | **train+val 20,000** |
| 重畳キャッシュ | `cache`（dl2 レンダ, train dual 7,403） | `cache_r4`（dl1 レンダ, train dual 7,881） |
| CV | 測れる | **測れない** |

→ SurgAtlas の是非は **expM04 の CV/LB でしか判定できない**。Q00A はそれを先取りして焼く形。

## 実装したこと

1. `train_lora_ext.py` の**単一トラック分岐**に `train_part` を実装（従来は `data.tracks` の
   multitrack 分岐にしか無く、FRAME では**黙って無視されていた**）。
   併せて `eval_steps: 0` のとき val を作らないガードを追加。
   ⚠️ 既定（`train_part` 省略）では連結もシャッフルもしないので**既存実験と数値一致**。
2. HF 中継 repo に **`pseudo_frames` part を追加**（0.82GB / packed 14,017 枚 + 擬似 parquet 5 本）。
   frame track の bundle に**擬似フレームが入っていなかった**（expK00 の packed は誰も上げていなかった）。
   `--track frame` は 15.3GB になった。
3. `vast_q00.sh`（pull / preflight / fetch / smoke / run / watch）。

## 手順

```bash
# 機体（vast 1×4090, ラベル orena_expQ00_frame_alldata）
REPO=/workspace/Orena bash workspace/expE01_segproc_baseline/setup_vast.sh
export HF_TOKEN=... HF_HOME=/workspace/Orena/.hf
bash workspace/expQ00_frame_alldata/vast_q00.sh pull       # 15.3GB / 約4分
bash workspace/expQ00_frame_alldata/vast_q00.sh preflight  # 件数で検算
bash workspace/expQ00_frame_alldata/vast_q00.sh fetch      # Qwen3.5-9B
bash workspace/expQ00_frame_alldata/vast_q00.sh smoke      # --train-limit 160
bash workspace/expQ00_frame_alldata/vast_q00.sh run        # 本走（デタッチ）
```

## 健全性の見かた（CV が無いので）

- 学習 loss 曲線が expN02/expN03 と同型か
- smoke ログの `★学習セット計 20000 問（part=all: train+val を連結）` と
  本走ログの `★重畳(train, r4): dual 11032`（assert が通ること）
- 完了後: コンテナ回帰テストで **empty 0 / 多行 0**（外部混合モデルは EOS が薄いので
  `stop_strings=["\n"]` + 1行目切断が必須。[[empty-count-is-not-format-compliance]]）

## 本走（2026-09-04 14:04 JST 開始）

**4×4090 DDP**（instance 49822949, 台湾, $1.509/h）で `config_Q00A_frame_alldata_ddp4.yaml`。

| | 実測 |
|---|---|
| s/it | **4.6**（1×4090 は 16.2 → **3.52×**, スケール効率 88%）|
| step | 4,755 → **6.1h**（完了見込み 20:10 JST）|
| VRAM | 19.3〜19.7 GiB × 4（揃っている = ddp_wrap が効いて rank0 への居候なし）|
| 検算 | `dual 11032 / 空 2873 / CONTROL 6095 / 全 20000` が事前実測と一致。合計 76,076 |

## 転送で踏んだ落とし穴（次に借りるときのために）

1. **HF bundle に擬似フレームが無かった** → `pseudo_frames` part を新設（0.82GB）
2. **HF 上の tar のメンバ名が古かった**（`frames_frame/…`）→ 展開先違いで丸ごと欠落。再アップロード済み
3. **検算パターンが `*.jpg` 固定**で endovis2018（PNG 1,438枚）を数え落とし PULL_FAIL
4. **転送するモジュールを手で列挙**して `paraphrase_bank` を落とした → ディレクトリ単位に変更
5. `$SSH` に `-n` が入っているとヒアドキュメントが届かない（リモートスクリプトが 0 バイトに）

## 完走（2026-09-04 22:14 JST）— expQ00B

| | 実測 |
|---|---|
| step | **4,755/4,755 完走**（5h46m49s / 4.38 s/it / rc=0）|
| 最終 loss | 0.2275（epoch 0.970, lr 2.5e-07 まで cosine で落ちきり）|
| adapter | `results/expQ00B_frame_alldata_r16_r4_ddp4/fold0/adapter` md5 `646acb71529a063b4c48cafeb350757c`（205MB）|
| 機体 | 49822949 破棄済み |

⚠️ expQ00A（r=64）は 1,758/4,755（38%）で**打ち切り**。v014 の LB 後退（−0.0538, 損失の84%がOOD）を
受けたユーザ判断で rank を 16 に戻したため。A の checkpoint は機体ごと破棄した。

## 提出物

`submit/v015_q00b_alldata`（v014 のフォーク・**差分は adapter 1点のみ**）。
回帰テスト（dl2/A4000, 20問, 5ケース）全合格: empty 0 / 重畳失敗 0 / hernia 差分一致 /
validate_output ✓ / 予算 39%（3.69s/問）。**tar.gz 書き出しとアップロードは未実施**。
