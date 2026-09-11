# expR00 — SEGMENT ベスト構成を expP00C の流儀で学習する

開始: 2026-09-04 13:45

## 狙い

SEGMENT の現ベストは **expN00 s1rules = CV 0.7065**（fold0 val 3,925問, judge込み）。
ただし作り方が旧世代のまま（r=16 / 3スタイル 1/3 混合 / 外部なし / 学習 16f@448 uniform・anchorなし）。

**expP00C の型**（scratch・r=64/α=128・s3 重畳 two_way・外部込み・貸し4GPU DDP・HF 中継）を
SEGMENT に当てる。同時に **P00C の本命施策＝学習と推論のフレームの渡し方のズレの解消**を
SEGMENT でもやる:

| | 従来の学習（expN00/expE06g）| 推論（CV / 提出 v013）|
|---|---|---|
| 枚数 | **16** | 32（router B）/ 64（router A）|
| anchor | **なし** | **あり** |

→ 学習を **32f@448 + anchor** にする。

## ユーザ確定事項（2026-09-04）

1. 外部データを **入れる**（P00C 準拠）
2. SEGMENT は **32f@448 + anchor**
3. **`train_part: all`**（fold0 val も学習に入れる。**CV は測れず判定は LB のみ**）
4. 遊休の **4×RTX6000Ada（vast 49813999）** を使う → ラベルを `orena_expR00_seg_s3` に貼り替え済み

入れないもの（すべて実測で負けている）: PROCEDURE / 擬似ラベル / mask 塗り(s2) / 33枚以上

## 構成

| 項目 | 値 |
|---|---|
| base | Qwen3.5-9B 4bit nf4 / bf16 / **scratch**（init_from_adapter なし）|
| LoRA | **r=64 / α=128** / dropout 0.05 / vision+language 16 種 |
| tracks | FRAME 1f@448 / **SEGMENT 32f@448 + anchor** |
| 重畳 | **s3**（bbox・クラス色・conf・番号なし）`two_way: true`（s0:s3 = 1:1）|
| 外部 | SMB + MultiBypass + SurgAtlas / loss_weight 0.5 / `track_limits {FRAME: 8000}` |
| max_seq_len | 8192（expN04 実測 32f@448: p50 4756 / p99 6039 / max 6158）|
| DDP | 4 GPU / bs1 × accum4 × 4 = **有効バッチ 16** |
| eval | `eval_steps: 0`（held-out が無い）/ `load_best_model_at_end: false` / `save_total_limit: 8` |

config: `config_R00_seg_s3_r64.yaml`

## 作業ログ

### 13:45 機体を確認 → **セットアップ済みだった**
vast 49813999（4×RTX6000Ada 49,140MiB / disk 250G・空き 204G）は expP00C 用に
立ち上げ済みで 1 step も回していなかった:
- `.venv` = torch 2.6.0+cu124 / **flash-attn 2.7.4.post1** ✓
- `frames_cache/448` **344,128 枚**（期待値一致）
- `$HF_HOME` 19GB（Qwen3.5-9B 取得済み）
- `cache_p00_448` s3 **47,558 枚** + index 3本
- 外部 parquet 3本

**足りないのは「expR00 用の重畳キャッシュ」と「外部フレーム tar 4本」だけ。**
主要ソース 8 本（train_lora_ext.py / dataset_seg.py / overlay3.py / ddp_wrap.sh / …）は
**md5 が全部一致**＝コードは同期済み。

### 13:52 列挙と被覆の確認（`render_cache_r00.py --dry-run`）

```
[train/FRAME]   15,992 問 / 実効フレーム mean 1.00
[train/SEGMENT] 15,994 問 / 実効フレーム mean 30.57   anchor 時刻あり 2,729問 (17.1%)
[val/FRAME]      4,008 問 / 実効フレーム mean 1.00
[val/SEGMENT]    4,006 問 / 実効フレーム mean 30.67   anchor 時刻あり 681問 (17.0%)
★列挙完了: ユニークフレーム 122,173 枚（DUAL のみ）
★生 JPEG の欠落 0 / build_samples の dropped 0
★既存キャッシュ 148,846 件 → **新規レンダ対象 30,770 枚**
```
★**実効枚数が 32 でなく 30.6 なのは正常**。SEGMENT の格子は 1s なので
29s のクリップからは最大 30 枚しか取れない（`dropped 0` が転送漏れでないことの裏取り）。

### 13:54 s3 差分レンダ開始（dl1 3シャード）
`cache_p00_448/s3` を **ハードリンク**で種付け（同一 FS なので即時・容量ゼロ）→
index を `index.p00seed{0,1,2}.json` に**別名で**置いてから `--resume` で差分だけ描く。
★index を元の `index.<i>of3.json` のまま置くと、新規シャードが同名で上書きして種を消す。

実測 **6.38 f/s（1GPU）** — 事前見積り 1.9 f/s の 3.4 倍。cascade が効いている
（空検出 88% → 全クラス側 0 件なら clip 専用を省ける）。1 シャード 10,280 枚で ETA 約 26 分。

★**chain と手動起動の二重化を潰した**: 最初 `chain_render_r00.sh` に
「shard0 を今 / shard1,2 を評価終了後に」と書いたが、GPU2 が先に空いたので手で shard1 を足すと
chain が後で **同じ shard1 を二重起動**して `index.1of3.json` を取り合う形になっていた。
→ chain を kill して、**shard2 だけを起動する** `wait_shard2.sh` に置き換えた。
（[[dual-safety-net-shares-write-target]] と同型。安全網を後付けするときは書き込み先を必ず見る）

## 次にやること
- [ ] 3 シャードの `完了 frames=` を確認 → index の key 合計 = 122,173 を検算
- [ ] `hf_push_bundle.py --parts overlay_r00_448` で HF に push
- [ ] 機体で `setup_r00.sh`（overlay_r00_448 + 外部4本 ≒ 10.6GB）
- [ ] smoke（`train_ddp_r00.sh 1 --train-limit 24`）→ attach3 の内訳で `expect_drawn` を確定
- [ ] 本走（`train_ddp_r00.sh 4`）約 3,500 step / 12〜13h
- [ ] s3@768 の val レンダ → 形式サニティ
- [ ] `submit/v015_segment_r00`（v013 のフォーク・`OVERLAY_STYLE` を s1→s3）

### 14:46 s3 差分レンダ完了 — 検算パス

| shard | frames | empty | drawn |
|---|---:|---:|---:|
| 0 | 10,280 | 6,984 (67.9%) | 3,296 |
| 1 | 10,211 | 7,027 (68.8%) | 3,184 |
| 2 | 10,279 | 7,009 (68.2%) | 3,270 |
| 計 | **30,770** | 21,020 | **9,750** |

```
★index key(ユニーク) = 179,616（列挙全数 122,173 以上。種の P00C 分を含む）
★s3 jpg = 57,308 = 種 47,558 + 新規 9,750
★file_s3 あり = 57,308 → jpg 実数と一致: True（索引に幽霊エントリなし）
```
実測 4〜7 f/s（見積り 1.9 の 2〜3.5 倍）。空率が 68% あり cascade がほぼ毎フレーム効いた。

### 14:12〜14:35 機体の乗り換え（★今日 5 回中 4 回が失敗）

ユーザ判断で「レンダ中は遊休の機体を持たない」→ RTX6000Ada(49813999) を破棄。
その後 **もっと安い/速い機体を探す**方針に切り替え。

| 試行 | 結果 |
|---|---|
| 4x4090-48GB 48463810 | ✓ 合格 → **49824344**（$2.631/h・47.4GiB・flash-attn 2.7.4.post1 実走）|
| 4x5090 48860467 ($1.388) | ★起動せず → 自動破棄 |
| 4x5090 47697095 ($2.136, DLP 548.7) | ✓ 合格 → **49825687**（HF 36MB/s）|

★`rent_probe.sh` の自動破棄ゲートが 4 回とも正しく働いた。**手で判断していたら
壊れた機体にデータを送っていた**。

### ★★flash-attn は「import が通る」を根拠にしてはいけない（本セッションの誤り）

8x5090 箱で `import flash_attn` が 2.8.3.post1 で成功したのを見て
「5090 の flash-attn は解決済み」と報告したが、**誤り**。ユーザ経由で判明した実態:

| 段階 | 結果 |
|---|---|
| torch cu128 | ✅ 2.11.0+cu128 / arch_list に sm_120 |
| flash-attn(PyPI) | ❌ **import は通るが sm_120 カーネルが無く実行時クラッシュ** |
| sdpa 代替 | ❌ PROCEDURE 64f で OOM（27.79 + 8.12 = 35.9 GiB / 31.4 GiB）|
| ソースビルド | 🔄 別レーンで実行中（nvcc 12.8.93）|

★落ち方が2種類ある: **①ABI 不一致 → import で落ちる（cu124/Ada）/ ②カーネル欠落 →
forward で落ちる（sm_120）**。①だけ知っていると import 確認で満足して②を見逃す。
★`train_lora_ext.py` の `attn_implementation` フォールバックは
**ロード時例外しか拾わない**（`except` がモデル構築を包んでいる）。②は走り出してから落ちる。
→ 5090 用 config は **`attn_implementation: sdpa` を明示**する。
メモリ [[flash-attn-abi-mismatch-falls-back-to-sdpa]] を上書き更新済み。

### 判定方法: 2機体で同一 smoke を打って VRAM を実測する

推定（4.0MB/token 換算で約 34GB）で 32GB の可否を決めない。**両方で測る**:

| 箱 | config | 狙い |
|---|---|---|
| 4x4090-48GB | `config_R00_seg_s3_r64.yaml`（flash-attn） | 本命。ここは確実に走る |
| 4x5090 | `config_R00_seg_s3_r64_5090.yaml`（**sdpa 明示**） | flash-attn 無しで 32GB に載るか |

★smoke は `--train-limit 200`。24 件では系列長の裾（p99=6,039）に当たる確率が 1/4 しかなく、
**VRAM ピークは最長サンプルで決まる**ので判定材料にならない
（[[tail-settings-need-tail-sized-samples]]）。

### 15:06 ★5090(32GB) は **2.0 GiB 足りない** — 実測で決着

ユーザ提案（4x5090 は 4090-48GB より安い・速い）を受けて実機で測った。

**まず flash-attn を forward で検証**（import では駄目、という直前の教訓を適用）:
```
flash_attn 2.8.3.post1 / torch 2.11.0+cu128 / RTX 5090 sm_120
  L=512  ★NG CUDA error: no kernel image is available for execution on the device
  L=6158 ★NG 同上
```
★uv が**ソースビルド**した（4m12s）にもかかわらず sm_120 カーネルが無い。
4 分で終わる時点で怪しく、実際 CUDA カーネルを焼いていない
（`TORCH_CUDA_ARCH_LIST` に 12.0 が入らず Python グルーだけビルドされた形）。
**別レーンが PyPI 版で踏んだのと同じ結論に、別経路で到達した。**

**sdpa 明示で smoke（`config_R00_seg_s3_r64_5090.yaml`）**:
```
attn_implementation=sdpa / 系列長 実測(n=32) min=291 p50=948 max=6197 / max_seq_len=8192
★重畳3(train, two_way=True): s3 299 / s0 321 / 空 36 / CONTROL 144 / 全 800   ← 割当は正常
Tried to allocate 5.73 GiB          ← logits.float()（事前推定 5.7 GiB と一致）
GPU total 31.36 GiB / free 3.75 GiB / process in use 27.60 GiB
→ 必要 27.60 + 5.73 = 33.33 GiB  vs  31.36 GiB = **2.0 GiB 不足**
```

**判断**: flash-attn が通っても節約は attention 側の 1〜2 GiB 程度で、2.0 GiB を埋められるかは五分。
埋まっても VRAM 99% 運用になり、過去に 98.5% で 33.96 s/it まで落ちている。
→ **4090-48GB(49824344) で本走**。5090(49825687) は破棄。

★外挿: 32f/seq 6,197 で 33.33 GiB なので **64f/seq 8,793 なら約 40 GiB**。
PROCEDURE を 32GB の 5090 で回すのは flash-attn が通っても厳しい（別レーンへの参考値）。

### ★私のスクリプトのバグ: 40GB を上り 1.9MB/s に流していた

`xfer_code_r00.sh` の router ステップが `submit/v005_segment_hybrid` を**丸ごと** tar していた。
この dir は **40GB**（`orena-focus-segment-hybrid{,-r2}.tar.gz` が 19GB ずつ + test 3.5GB）で、
**約 5 時間コース**。8 分気づかなかった。
→ expR00 は `router_groups` を使わないので router 自体が不要。要るのは
`router.py` + `resources/group_templates.json` = **327KB** だけ。**40GB → 327KB**。
[[verify-tar-size-before-sending]] を、ディレクトリ名だけ見て送るという形で踏んだ。

★片付けで `pkill -f "submit/v005_segment_hybrid"` を打って**自分のシェルを撃った**
（[[pgrep-f-matches-itself]] を同日2回目）。PID 直指定に切り替えて解決。

### 15:25 ★高速化の診断 — GPU が 20〜30% 遊んでいる（DDP ストラグラー）

smoke（48GB 箱・63 step）完走: OOM 0 / 定常 ~15 s/it / adapter 保存（`fold0` → `fold0_smoke200` に退避。
本走が `fold0_001` になる罠 [[resolve_run_dir]] を避けるため）。

学習中の GPU util を 2 秒おきに 5 回サンプリング:
```
100/100/ 99/100   100/100/100/100   100/100/ 99/100
  5/ 97/100/  0   ← rank0,3 が待ち
  0/  0/100/ 95   ← rank0,1 が待ち
```
CPU 256 vCPU で 98% idle / load 4.7 → **データ供給は律速でない**。
系列長 p50 948 / max 6,197（6.5 倍）なので、長いサンプルを引いた rank に他 3 rank が
all-reduce で待たされる。step 間隔が 12〜21 s と暴れる原因もこれ。

### 対策（学習の中身を変えない 2 本）

| # | 手 | 変わるもの | 実装 |
|---|---|---|---|
| 1 | **長さグループ化 sampler** | 訪問順だけ | `WeightedTrainer._get_train_sampler` を上書き → HF `LengthGroupedSampler(bs×accum×world, lengths=proxy)`。proxy = 180×枚数 + 400 + len(question)/3。config `train.group_by_length: true`（既定 off） |
| 2 | **`logits_to_keep`** | 数学的に等価 | P00 レーンが 15:22 に `train_lora_ext.py` へ追加（常時 on）。**検証記録が無い**ので env `R00_KEEP_EQUIV_TEST=1` で全 logits と末尾 K の loss を同一入力で突き合わせる仕組みを足した |

★accelerate は batch i を rank `i % world` に配る（`BatchSamplerShard`、確認済み）ので、
連続 16 index を同程度の長さに揃えれば 4 rank × accum 4 が揃って終わる。
★注記の訂正: `group_by_length` は transformers 5.x で「削除」ではなく
`train_sampling_strategy="group_by_length"` に**改名**。ただし datasets.Dataset 以外は長さを
自動推定できないので sampler 上書きが要る。

★`train_lora_ext.py` は P00 レーンが同日 15:22 に編集した共有ファイル。
追加は config gate / env gate で**既定の挙動を変えない**形にした（バックアップ `.bak_pre_r00_*`）。

★2 が入ると 5090 の OOM 原因（`logits.float()` 5.73 GiB）が消える → 33.2 → 約 27.7 GiB で
**32GB に載る見込み**。ユーザ許可を得て 5090 を引き直し中（offer 47697095）。

### 15:37 等価性テスト 合格 → 15:38 本走開始（48GB 箱 49824344）

`R00_KEEP_EQUIV_TEST=1` で 4 rank × 2 step = **32 件**を同一入力で比較（FOCUS w=1.0 / 外部 w=0.5、
系列長 288〜6,089、K 4〜11）: **|diff| = 0 が 29 件、1.19e-07 が 3 件**（fp32 丸め）。
`logits_to_keep` は重み付き経路含めて数学的等価。`★group_by_length: LengthGroupedSampler(batch=16)` も起動。

本走の起動ログ（4 rank 一致）:
```
★学習セット計 40000 問（part=all）
★重畳3(train, two_way=True): s3 11939 / s0 14007 / 空 1870 / CONTROL 12184 / 全 40000  missing 0
★group_by_length: LengthGroupedSampler(batch=16) proxy min=588 p50=831 max=6361 n=56000
attn_implementation=flash_attention_2
```
dl1 の空回し（`count_drawn_r00.py`）は **s3 11956 / 空 1853**（s0・CONTROL は一致）。
**17 問（0.14%）が箱では空扱い**。index 6 本・s3 57,308 枚・arm csv・QA Arrow の fingerprint は
両機体で完全一致なのでデータ差ではない。サンプル構築側（フレーム欠落）を疑い `dropped` を確認中。

`expect_drawn` は **11956（dl1 基準）** だが本走は `null` で起動済み（assert なし）。
上の 11939 と 17 差の説明が付けば config に `expect_drawn: 11939` を書いて再現性を固定する。

### 15:44 ★★本走を一度止めた — 箱に 768 キャッシュが無く SEGMENT train のフレームが 9% 落ちていた

本走ログ `built 15994 SEGMENT ... 実効フレーム数 mean=27.9 ... distinct frames 26887, dropped 0`
（dl1 の空回しは mean 30.6 / distinct 0）。フレーム総数は両機体とも 344,128 で同一・index/QA も同一。

原因: `dataset_seg.resolve_frame` は 448 が無いと **`FALLBACK_SIZES=(768,)` から縮小して供給**する。
dl1 には 768 キャッシュ（280,014 枚）があるので train 26,887 枚 / val 388 枚を黙って補っていた。
箱には 448 しか送っていないので、その分が**フレーム単位で静かに落ちる**
（`distinct frames` = 未抽出 job 数、`dropped` はサンプル単位なので 0 のまま）。
→ 17 問が「dl1 では s3・箱では空」だったのもこれ（落ちたフレームにしか検出が無かった）。

★教訓: **`dropped 0` はフレーム欠落の証拠にならない。`distinct frames` が 0 でなければ落ちている。**
　`extract: false` の機体では **448 と 768 の両方**を置く（HF part `focus_frames_768` 9.34GB）。

対処: 6/3500 で停止（`expR00_train_aborted_no768fallback.log` / `fold0_aborted_no768` に退避）→
`focus_frames_768` を pull → `check_frames_r00.py` で mean 30.6 / s3 11956 を検算 → 再起動。
config に `expect_drawn: 11956` を固定した。
★torchrun へ SIGTERM しても worker 37 本が GPU を掴んだまま残った → **PID 直指定で kill -9** が要る。
★hf_pull_bundle の 2 回目以降は同じ `--cache` を使うと `IncompleteSnapshotError`（削除済み tar を
不完全 snapshot と判定）→ **part ごとに別 cache dir** を使う。

5090: 47697095 は上り 0.233MB/s で probe 不合格、48860467 は 2 回とも起動せず。**打ち切り**（今日 6 敗）。

### 15:50 768 投入 → 検算一致 → **本走 再起動（pid 20033, log `expR00_train.log`）**

`focus_frames_768` 280,014 枚（134 MB/s・1.2 分）。`check_frames_r00.py`:
```
★SEGMENT 実効フレーム mean = 30.59（dl1 30.6）
★重畳内訳: s3 11956 / s0 14007 / 空 1853 / CONTROL 12184 / missing 0   ← dl1 と完全一致
```
config `expect_drawn: 11956` の assert 付きで再起動。2 回目の pull 失敗は私が pull コマンドにも
`HF_HUB_OFFLINE=1` を付けていたため（pull はオンライン・検算はオフライン、に分離して解決）。

### 15:52〜 5090 へ乗り換え（ユーザ判断: `logits_to_keep` で 32GB に載る）

48GB smoke で `logits_to_keep` 込みの実使用が 37〜39 → **22〜23.5 GB** に下がった（5090 31.4GB に余裕）。
48GB 本走は安全側として走らせたまま、5090 を並行で立ち上げ、smoke が通った方へ乗り換える。

| 箱 | id | $/h | 構成 | 見込み |
|---|---|---|---|---|
| 4x4090-48GB | 49824344 | 2.631 | accum 4 | 10.5h / $28（走行中） |
| 4x5090 台湾 | 49830901 | 2.136 | accum 4 / DLP 548.7 | 7.5h / $16（setup 中）|
| **8x5090 US** | 49829574 | 2.899 +帯域 | **accum 2**（1×2×8=16） | **約 4h / $12**（ユーザ指名, 借用中）|

★rent_probe の上りゲートを `MIN_UP=100000`（0.1MB/s）に緩和。データは全て HF なので上りは
60MB のコード転送にしか使わない（台湾 host は上り 0.23MB/s で一度落とされていた）。
★8 GPU 用: `config_R00_seg_s3_r64_5090x8.yaml`（grad_accum 2）/ `train_ddp_5090x8.sh` /
`smoke_5090x8.sh`（CVD 0-7、ORIG_CVD 0-7）。`group_by_length` の batch は bs×accum×world = 16 で不変。
★setup v2 は **448+768+overlay+外部+model** を HF から取る（448 だけだと 9% 落ちる教訓を反映）。

### 16:05 `group_by_length` の実測 — 効果は約 8%（見積り 20〜25% より小さい）

48GB 本走 step 12〜32 の step 間隔（秒）:
`21 19 22 18 21 21 27 13 14 14 13 13 22 16 | 6 7 9 8 7 6 7`
→ median 14.0 / mean **14.5 s/it**（smoke の定常 15.7）。3,500 step ≈ **14.1 h**。
長いサンプルの塊と短い塊に分かれて処理される（megabatch 内ソートの狙いどおり）が、
長い塊の中でも系列長が ±30% ばらつくので 1 step 内のストラグラーは残る（util 79/100/10/57）。
★見積りを外した理由: 「GPU idle 20〜30%」は短いサンプルが混ざった step の観測で、
グループ化後は長い塊の中の分散が支配的になる。訪問順の変更だけでは限界。
→ 乗り換え（8x5090 で step あたり約 1/2）の価値が上がった。

### 16:20 ★4x5090 台湾で本走 — **実測 5.90 s/it → 全 3,500 step ≈ 5.7 時間**

`logits_to_keep` で VRAM が 19 GB / 31.4 GB に収まり、5090 が使えるようになった（ユーザ判断）。

step 間隔（step2-52, 秒）:
```
15 11 10 11 10 12 11 11 12 11 11 10 11 11 10 10 11 | 7 7 6 6 6 7 5 | 3 2 3 4 2 3 2 2 3 2 3 2 3 2 3 2 4 2 3 2 2 3 2 3 2 3 | 11 12
   ← 長い側（megabatch 先頭）          ← 中間        ← 短い側                                                    ← 次の megabatch
```
★**megabatch 1周(50 step) mean = 5.90 s/it**。3,500 step = **5.74 h**（$2.272/h → 約 $13）。
序盤 12 step だけ見て 11.25 s/it → 10.9 h と見積もったのは**上振れ**だった。
`get_length_grouped_indices` は megabatch(mega_batch_mult 50 × batch 16 = 800 サンプル)内を
長さ順にし、**最長バッチを先頭に置く**（OOM を早く出すため）。
→ **s/it は 1 周(50 step)を均さないと意味を持たない**。ノコギリの底(2s)も頂点(15s)も代表値でない。

機体の最終構成:
| 箱 | 判断 |
|---|---|
| 4x5090 台湾 49830901 $2.272/h | **本走中**（5.74h / $13）|
| 4x4090-48GB 49824344 | 破棄（5090 と二重実行・14.5 s/it で遅く高い）|
| 8x5090 Zhejiang 49832592 | 破棄（探索打ち切り・ユーザ判断。probe が確保済みだった残骸）|

★8x5090 の探索は打ち切り。5.7h なら 4x で十分という判断。
★探索時の学び: `reliability > 0.98 disk_space >= 150 inet_down >= 300` の絞りが厳しく
  8x5090 が 2-4 件しか見えていなかった。**条件を外すと 19 件**あった。

### 22:14 ★本走 完了（4x5090 台湾, r=16）

```
3500/3500 [5:48:47<00:00, 5.98 s/it]   expR00 rc=0
最終 step の重みを保存（global_step=3500）   OOM/例外 0
adapter 205MB  md5 c88fde1d59a3716fd93271d2e6723005（機体と dl1 で一致）
→ workspace/expR00_seg_s3/results/expR00_seg_s3_r16_alldata_5090/fold0/adapter
```
| step | loss | lr |
|---|---|---|
| 10 | 0.9153 | 8.57e-06 |
| 870 | 0.3061 | 8.80e-05 |
| 1730 | 0.2102 | 5.34e-05 |
| 3020 | 0.1583 | 4.87e-06 |
| 最後50点 mean | **0.2213** | — |

**実測 5.81h / 約 $13**（当初の 4090-48GB 見積り 14.1h / $37）。内訳:
`logits_to_keep`（VRAM 37→19GB で 5090 が使えるように）＋ `group_by_length`（約 8%）＋ 5090 乗り換え。

### 22:17 ★機体の run_infer.py が古く `--overlay3-style s3` を弾いた

`invalid choice: 's3' (choose from s0, s1, s2, mix)`。md5 照合すると **run_infer.py と overlay3.py の
2 本が古い**世代だった（xfer は本走前に済ませていたが、その後 dl1 側で s3/mix2 対応が入った）。
★学習には影響なし（学習が使うのは attach3 だけで、重畳内訳は dl1 と完全一致していた）。
★**評価に入る前に、推論経路のファイルも md5 照合する**。学習が通ったからコードが揃っている、
　とは言えない（[[eval-must-share-dataset-args-with-train]] の同型）。
→ run_infer/dataset_seg/eval_seg/prompts_seg/overlay3/overlay_render の 6 本を同期して再実行。

### 03:00-04:10 コンテナ v018_segment_r00 を作成

★ナンバリング: 既存は v017 まで（v015 が 2 つ、v016/v017）→ **v018**。

**土台の選択**: v015_segment_framerouter（LB 0.5989）で組みかけたが、**v013 の inference.py が
segrules + 重畳を持つ唯一の版**（v015 は segrules すら無い）で、私の A/B 測定条件と一致するので
**v013 を土台に切り替えた**。差分は adapter / OVERLAY_STYLE / router / cascade の 4 点。

#### ★重畳を積む判断（当初は「積まない」と誤判断していた）
v013（重畳 s1, LB 0.5338）vs v015（重畳なし, 0.5989）から「重畳は OOD で害」と結論しかけたが、
**交絡していた**: v013 だけ**外部データなし**（v012/v013/v015 の OOD を最もよく説明する変数）。
しかも v013 の重畳は学習の 1/3 のみ（expR00 は 1/2）。重畳そのものの効果は
**同一モデル・同一問題で s0 0.6381 → s1 0.7047（+0.0666）** と直接測られている。→ 積む。

#### ★移植で見つけた欠落 3 点
1. **v013 の renderer は s3 を描けない**（`STYLES=("s1","s2")`）。手で書くと学習時とズレるので
   dl1 `overlay3.py` の描画部 215 行を**丸ごと移植**。`CLASS_BGR` / `_put_label as _put_label_chip`
   の import が抜けて `NameError` → **描画まで実行するテスト**で即検知（import だけなら通っていた）
2. **s3 の色凡例が note に付いていなかった**。学習時 `attach3` は `note += s3_legend(sorted(seen_cls))`
   → 付けないとプロンプトが学習と変わる。`apply()` で描いたクラスを集めて `note()` で足す実装に
3. **stage_resources.sh に検出器の配置が無かった**（v015 由来のため）→ v013 から移植

#### ★cascade は不採用（ユーザ判断）
cascade =「F40 が 0 件なら F39 を省く」で 2.20x 速いが、**F39 が要る理由はまさに F40 が
clip に弱いこと**（AP50 0.315→0.421）＝**省く条件と必要な条件が重なる**。
val の質問は **27.4% が Clip に言及**、FO 名なしの計数問 24.8% でも clip が関わる。
★ただし **expR00 の学習用重畳は cascade=True で焼かれている**（clip 省略 6,773/10,280 枚）。
　推論の方が検出が濃くなるが、**より完全な検出を見せる方向は害にならない**（ユーザ指摘。
　私は当初「学習より良い検出を見せるとズレ」と誤って述べた）。
★**A/B にも非対称があった**: arm A(560) は cascade=True、arm B(768) は cascade=False で焼かれていた。
　**より劣化した重畳の A が勝った**ので A 優位は保守側だが、公平な比較ではなかった。

#### latency（本番 L40S 実測から積算）— 単価を 2 回間違えた
- 誤り①: v015 の dl2 実測(7.34)と v013 の本番実測(14.25)を混ぜて外挿した
- 誤り②: v013 の 7.36s を「32枚ぶん」と誤解 → 実際は router で **59.0 枚平均**
- 正: **両検出器 125 ms/枚**。v018 は 63.1 枚平均 → 検出器 7.87 + モデル 6.89 = **14.76 s/問 = 96.9%**
- ★**rush は発動しない**: 条件は「残り予算÷残り問数 < 11.25s」で 14.76 < 15 なら単調増加。
  発動は実効 15.2 s/問 超から。**v013 は 93.5% で forfeit 0 で着地**しており前例がある

#### 高速化の実測（cascade 以外）
| 手 | 結果 |
|---|---|
| バッチ推論 | ❌ F40 bs=4 で 1.07x / F39 は bs=2 で 0.36x と**悪化**（GPU が 1 枚で飽和）|
| decode 最適化 | ❌ decode は **5 ms/枚**のみ。forward が 98/154 ms で支配的 |
| TRT | 保留。F40 の既存 engine は **mask を出さず**（fo_index 用）、F39 は engine 無し。再エクスポート×2 |

#### ビルド
`orena-focus-segment-r00:latest` **29.1GB** `sha256:6ff0be9906114d93…` BUILD_ID=r1。
イメージ内検算: cascade=False / OVERLAY_STYLE=s3 / STRIDE=1 / STYLES に s3 /
GROUP_A に AGGREGATION+EVENT / adapter md5 `c88fde1d…` / detector md5 `8f749fbbb46da`+`7d07da340ea3f` /
scipy 1.17.1 + `Mask2FormerLoss` import OK。
★`do_save.sh` の build_id 検算が dl1 の壊れた nvidia runtime を踏んだ → `--runtime=runc` を明示。

---

## 2026-09-08 expR00B — 重畳の是非 と FOCUS fine-tune を同一土俵で決着

**機体**: vast 50212335（4×RTX5090 32GB, 香港, $1.716/h）7時間29分 ≒ **$12.8**

### 3アームを同一 3,924 問で比較（64f@560 + anchor + segrules, fold0 val）

| アーム | SCORE | 正答数 |
|---|---:|---:|
| expR00 + **s3**（重畳あり・既存）| 0.7612 | 2983 |
| expR00 + **s0**（重畳なし）| 0.7650 | 2969 |
| **expR00B（FOCUS-ft）+ s0** | **0.7733** | **3015** |

⚠️ expR00 は `train_part: all` なので **絶対値は暗記込み**。判断に使うのは対応のある差だけ
（[[paired-config-compare-survives-leakage]]）。expR00B は expR00 から **FOCUS の train のみ**で
継続学習しており、val への追加露出が無いので両者の暗記量が揃う（Stage B に不利側＝保守的）。

### (a) 重畳 s3 → s0: **差なし**

| | Δ | 不一致 | p |
|---|---:|---:|---:|
| **全体** | −0.0036 | 396 | **0.51** |
| temporal_grounding | −0.0123 | 198 | 0.18 |
| aggregation | +0.0190 | 47 | 0.38 |
| object_recognition | −0.0018 | 133 | 0.86 |

★**SCORE では s0 が +0.0038 だが問数では s3 が +14問と符号が逆**。in-dist では
**重畳は効きも害もしない**。にもかかわらず latency は 6.9 → 14.26 s/問（予算95%）に倍増する。
→ LB で v018(重畳) の OOD が最下位 0.5110 だったことと合わせ、**重畳を外す判断を支持**。
⚠️ ただし OOD 側は依然として測れていない（val に ood=1 が 0 件）。

### (b) FOCUS fine-tune: **効く**

| | Δ | 不一致 | p |
|---|---:|---:|---:|
| **全体** | **+0.0117** | 352 | **0.016** ✅ |
| **object_recognition** | **+0.0148** | 129 | **0.034** ✅ |
| aggregation | +0.0298 | 43 | 0.13 |
| complex_reasoning | +0.0306 | 5 | 0.38 |
| temporal_grounding | +0.0080 | 157 | 0.34 |
| event_understanding | −0.0417 | 18 | 0.24 |

SCORE 0.7650 → **0.7733**、正答 **+46問**。

### ★★16枚系で見えた「FOCUS-ft が temporal を壊す」は 32枚系では再現しない

| | 16枚系（expN01 A→B）| **32枚系（今回）** |
|---|---:|---:|
| temporal | **−0.0755（p<0.001）** | **+0.0080（n.s.）** |
| object | +0.0176（p=0.082） | **+0.0148（p=0.034）** |
| 全体 | −0.0163（p=0.066） | **+0.0117（p=0.016）** |

→ **「temporal だけ元モデルに戻す合成」は不要**。expR00B を全問に使えばよく、構成が単純化される。
16枚系の temporal 劣化は**16枚特有**だった可能性が高い（機序は未特定のまま）。

### 運用で踏んだ落とし穴（4件、いずれも私の実装ミス）

1. **`.toml` が転送対象外** → `pyproject_box_5090.toml` 欠落で `uv sync` 即死。
   `xfer_code_r00.sh` の glob に `*.toml` `*.json` を追加した
2. **`train_lora_ext.py` に `--max-steps` は無い**（`--config/--train-limit/--epochs/--track` のみ）。
   smoke が6秒で落ちた。**smoke を先に置いた設計が効いて学習4時間前に露見**
3. **merge が qID 重複で停止** → `--allow-dup-qid` が要る（`2392989` が heico/lapchole 双方に在る）
4. ★★**smoke が本走と同じ out_dir に adapter を書く** → ②のスキップ判定がそれを掴み、
   **32件学習のゴミ adapter で③が10分走った**。smoke 用 config で out_root を隔離し、
   実行後に丸ごと削除 + 「smoke 後に本走 adapter が在ったら停止」の assert を追加

⚠️ judge が2回落ちた（`time_count_prior.json` 未転送 / 判定モデル Qwen3.5-4B 未取得）。
**setup の検算が学習側に偏っていた**のが原因。judge を回す機体では setup 時点で
判定モデルと後処理ファイルの存在まで確認すること。

### 成果物（dl1 へ回収済み・検証済み）
- adapter: `results/expR00B_seg_focusft_trainonly/fold0/adapter`（md5 `b361916844ed0ac85e5dcfca319b8007`）
- `eval_r00_s0_64f560` / `eval_r00B_s0_64f560`（各 n=4005・空回答0）
- 中間 checkpoint 4.9GB は回収せず（eval を切っているので best 選択の余地が無い）

---

## 2026-09-10 ★2モデルの confidence 選択が効く（+0.0120, p=0.0009）

expR00（A）と expR00B（B, FOCUS-ft）の**予測が食い違う問だけ**、生成トークンの
log-prob を比べて良い方を採る。judge 済みの同一 3,924 問で評価。

### 上限と基準
- 両方正解 2,816 / A のみ 153 / B のみ 199 / 両方不正解 756
- **オラクル SCORE 0.8157（+153問）**。選択が効くのは **correctness が食い違う 352 問だけ**
- 基準「常に B」= 352 問で **56.5%** 的中。これを超えないなら採用しない

### 結果

| 選択器 | 的中率 | 不一致での正答 | 常にB比 | p |
|---|---:|---:|---:|---:|
| 常に B（基準）| 56.5% | 199 | — | — |
| **`logp_mean`（長さ正規化）** | **65.9%** | **232** | **+33** | **0.000** |
| `logp_sum` | 65.1% | 229 | +30 | 0.001 |
| `logp_min` | 63.4% | 223 | +24 | 0.010 |

全体: **SCORE 0.7733 → 0.7853（+0.0120）/ 正答 +33 / 選択勝64・B勝31 / p=0.0009**。
オラクルの **22% を回収**。

| group | n | B | 選択 | Δ | p |
|---|---:|---:|---:|---:|---:|
| **temporal_grounding** | 1620 | 0.6994 | **0.7105** | **+0.0111** | **0.003** |
| event_understanding | 144 | 0.8194 | 0.8542 | +0.0347 | 0.062 |
| aggregation | 369 | 0.6125 | 0.6233 | +0.0108 | 0.289 |
| object_recognition | 1693 | 0.8576 | 0.8612 | +0.0035 | 0.461 |

★**最大バケット temporal で有意**。小バケット頼みではない。

### ★予想を外した（記録として残す）
事前に「B は1epoch 多く自信過剰なので選択が B に退化し 56〜60% 止まり」と予測した。
**自信過剰は実在した（73.3% で B の logp_mean が高い）が、それでも 65.9%**。
**閾値を持たない大小比較なので較正が不要**だったのが効いた。汚染された val しか
較正データが無いという懸念も、この形なら無関係。

### 実装
`run_infer.py --with-conf`（opt-in）。`compute_transition_scores` で
`logp_sum / logp_mean / logp_min / n_gen_tokens` を responses.json に追加。
★**greedy のまま**で生成は変わらない（4問で content 一致を assert 済み）。

### ⚠️未解決: latency
2モデル分の forward が要り **約13.8 s/問＝予算92%**（v013/v018 が 95% で綱渡りだった領域）。
「B の conf が低い問だけ A も回す」選択的2パスにしたいが、**B の conf は不一致 931 問しか
測っていない**ので損得を評価できない。全 3,924 問の B conf が要る（dl2 で約4時間・無料）。
⚠️一度「不一致の下位X%を再走」という表を作ったが、**どの問が不一致かを知るには
既に2パス走っている**ので循環しており無効。ゲートは B の1パスだけで決まるものに限る。

---

## expR00C（B' = expR00 + FOCUS-ft all-data）— 2026-09-11 完走

`config_R00C_focusft_alldata.yaml` / `chain_R00C.sh`。vast 50533409（台湾 4×RTX5090, $1.989/h）。

- **2,500 / 2,500 step 完走**（4h42m54s、≈6.8 s/step、08:50→13:34）
- データ 40,000（train 31,986 + val 8,014、`train_part: all`、overlay3 `two_way: true` / `val: true`）
- lr 3.0e-5 / r16 α32 / 有効バッチ 16 固定（grad_accum は GPU 数から chain が自動設定）
- 起点 `init_from_adapter` = `expR00_seg_s3_r16_alldata_5090/fold0/adapter`
- ログ末尾 `最終 step の重みを保存（global_step=2500）` → **adapter は最終 step**（load_best ではない）
- **adapter md5 `09a229b19b760ae3319ba7948f050a1a`**（機体側 chain 報告値と dl1 回収後で一致）
- 回収物: `checkpoint-200 〜 2400` の 12 本 + `adapter` + `train_20260910_235050.log` + `config.yaml`
  （2200 md5 `b7afb498a22e8307309fa4683ddd1383` / 2400 md5 `12c2a91ad66927748909aa9ef609a758`）
- 回収完了後、**ユーザ許可を得て機体を破棄**。他セッションの `orena_expP06_video_alldata` は非接触

★**CV は測れない**。fold0 val を学習に入れているため。ユーザ判断で測定はスキップ
（同じ130動画を切り直しているだけなので別 fold でも汚染は解けない）。

## latency の未解決（上の⚠️）はここまで詰めた

- v018 の本番 14.76 s/問（96.9%）の内訳のうち **7.87s は検出器**。expR00C / expN04 は
  どちらも s0 なので **検出器・重畳を v019 から削除**でき、1パス相当は 6.89 s/問になる
- dl2 4090 で同一17問の対応比較: 2パス mean 12.05s / 1パス mean 6.80s → **比 1.82×（問ごと中央値）**
- → 本番換算 6.89 × 1.82 = **12.5 s/問 = 予算 84%**（v018 の 96.9% より軽い）
- 「選択的2パス」は**循環していて設計できない**ので捨て、**両枝とも常に2パス**にした
