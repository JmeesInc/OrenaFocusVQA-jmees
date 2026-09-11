# workspace/fold/

fold 割り当ての単一の情報源。全実験がここの `folds.csv` を参照する（実験ごとに fold を切り直さない）。

## 構成

```
fold/
├── README.md            # このファイル（設計意図を必ず更新する）
├── generate_folds.py    # fold 生成スクリプト（バージョン管理付き）
├── splits.py            # ★split の単一情報源。CV(5-fold) と LOPO(OOD) を返す
└── v001/
    └── folds.csv        # 列: videoID, dataset, fold（動画単位で割当）
```

## split の取得方法（全学習コード共通）

`folds.csv` を直接パースせず **`splits.py` を経由**する:

```python
import sys; sys.path.insert(0, "workspace/fold")
from splits import cv_folds, lopo_splits, cv_fold, filter_by_videos

for s in cv_folds("v001"):          # in-distribution 5-fold CV
    tr_reqs, tr_refs = filter_by_videos(requests, references, s.train_videos)
    va_reqs, va_refs = filter_by_videos(requests, references, s.val_videos)
    # s.name = "cv_fold0".. / train 104 video, val 26 video

for s in lopo_splits("v001"):       # OOD: 術式跨ぎ
    ...  # lopo_train_heico (train heico30→val lapchole100) / lopo_train_lapchole (逆)
```

## 本コンペの fold 設計方針

**動画単位の GroupKFold が大原則。** 同一動画の質問が train/val に分かれると重度のリークになる（同じシーン・同じ異物インスタンスを見て答える質問が多数あるため）。

考慮すべき層化軸（folds.csv 作成時に分布を確認して README に記録する）:
- **術式**: heico (colorectal) と lapchole (cholecystectomy) の混合比
- **track**: FRAME / SEGMENT / PROCEDURE の質問数バランス
- **capability group**: 5グループの分布（評価は group 単位のバケット平均なので偏りは CV を歪める）
- **OOD 汎化の検証**: テストは未知の術式を含む。「lapchole のみで学習 → heico で検証」（またはその逆）の leave-one-procedure-out 検証を、通常 CV とは別に用意する価値が高い

## ルール

- 前処理やデータを変更したら新バージョン（`v002_...`）を作る。旧バージョンは削除しない
- 実験の `config.yaml` の `cv.folds_csv` で使用バージョンを指定
- 各バージョンの切り方・分布・意図をこの README に追記する

## 採用する CV 設計（LoRA 以降）

2系統を **併用**する（決定: 2026-07-22）:

### (1) in-distribution 5-fold CV — `cv_folds("v001")`
- 動画単位 GroupKFold、術式(heico/lapchole)層化の round-robin。**full 5-fold**（5回学習）で頑健に測る。
- 各 fold: train 104 video / val 26 video。
- **分布（検証済み・均衡良好）**:
  | fold | heico | lapchole | heico サブ術式(Prokto/Rektum/Sigma) | QA数(全track) |
  |------|-------|----------|-------------------------------------|--------------|
  | 0 | 6 | 20 | 3/2/1 | 10010 |
  | 1 | 6 | 20 | 2/1/3 | 10025 |
  | 2 | 6 | 20 | 2/3/1 | 9923 |
  | 3 | 6 | 20 | 1/2/3 | 10021 |
  | 4 | 6 | 20 | 2/2/2 | 10021 |
  - 動画数・QA数はほぼ完全均衡。heico サブ術式(各10)は round-robin で多少ばらつくが実用上問題なし。

### (2) LOPO（leave-one-procedure-out, OOD）— `lopo_splits("v001")`
- **OOD-A**: train=heico(30) → val=lapchole(100) / **OOD-B**: train=lapchole(100) → val=heico(30)
- 目的: **未知術式への汎化を推定**。テストは cholecystectomy + 未知術式を含む。
- **根拠**: expB02 で「プロンプト改善(baseline→formatcond)が heico で +0.051 だが lapchole で −0.005 と転移しない」と判明。
  単一術式(heico)だけの in-dist CV は術式非依存性を見逃す → LOPO を必ず併用する。

**⚠️ 分類器ヒント併用時の注意**: expC00 の FO 分類器は現在 fold0 holdout のみ学習。full 5-fold で LoRA+ヒント
複合系を clean 評価するには **分類器も fold ごとに再学習**が必要（現状 fold0 の val のみ clean）。

## カバー範囲
- v001 = **130 動画**（heico 30 + lapchole labeled 100）。lapchole 未ラベル70はVQAが無いため fold 対象外
  （LoRA では pseudo-label / SSL の補助素材として別途検討）。

## バージョン履歴

| バージョン | 作成日 | 切り方 | 備考 |
|-----------|--------|--------|------|
| v001 | 2026-07-17 | 動画単位・術式層化 round-robin 5-fold（130動画） | expC00/expB02 で使用。2026-07-22 に splits.py で 5-fold CV + LOPO を確定。LoRA もこれを再利用 |
| v002 | 2026-07-2x | v001 ＋ 希少 FO クラス再割当（FO 分類器 expC00 用） | Gallstone 評価可能 fold 2→3、macro-AP std 0.049→0.036 |
| **v003** | **2026-07-28** | **多基準層化 GroupKFold**（`generate_folds_stratified.py`） | **VQA 用の推奨 fold**。下記参照 |

## v003 — 多基準層化 GroupKFold（VQA の推奨）

### なぜ作ったか（v001 の実測問題）
v001 は「術式層化 round-robin」なので dataset 比は良い（幅 0.72pt）が、**それ以外が揃っていなかった**:

| 基準 | v001 の最大幅 | 問題 |
|---|---|---|
| dataset | 0.72pt | ✅ |
| bucket(capability group × ood) | 6.14pt | |
| 希少 capability leaf | 0.47 / 0.89pt | ✅ |
| **FO クラス** | **Silicone loop 11.88pt（17.0% vs 5.2% = 3.3倍）** / **Gallstone は f1・f4 が 0 件** | ❌ |
| 回答形式 | number 5.97 / fo_class 5.41pt | ⚠️ |

**回答形式の 6pt ズレは SCORE を約 0.015 動かす**（number は最弱 0.28-0.51、fo_class は 0.68-0.71 のため）。
これは比較したい効果量（27B vs 9B の差は 0.007〜0.030）と**同じ桁**で、fold 間比較の交絡になる。
また **Gallstone が 2 fold から欠けている**ため、その 2 fold では Gallstone を一切評価できない。

### 層化の優先順位（user 指示 2026-07-28）
1. **dataset 比** 2. **bucket 比（特に希少 capability の存在保証）** 3. **FO クラス比（希少クラス）** 4. **回答形式比**

重みは `W = {dataset:100, bucket:30, leaf:30, fo:100, fmt:5}`。
※`fo` は当初 10 にしていたが、重み掃引（10/40/100）の結果 **100 が最悪ケース最小**だったため採用
（fo=10 では Silicone loop 7.35pt が残る）。**FO クラスは動画固有（その動画に silicone loop があるか否か）**なので、
動画単位グループ制約下では最も揃えにくく、重みを厚くする必要がある。

### 結果

| 基準 | v001 | **v003** |
|---|---|---|
| dataset | 0.72 | 0.63 |
| bucket(group) | 6.14 | **1.36** |
| 希少 capability (SITUS / ATTRIBUTES) | 0.47 / 0.89 | **~0.1** |
| **Gallstone** | 1.78（**f1・f4 が 0 件**）| **0.30（全 fold に存在）** |
| Silicone loop | 11.88 | **2.41** |
| Sponge | 7.27 | **2.70** |
| **number** | 5.97 | **2.33** |
| **fo_class** | 5.41 | **1.54** |
| 動画数 | ほぼ均等 | **全 fold 26 本ちょうど**（heico 6 + lapchole 20）|

→ **形式構成による SCORE 交絡が約 0.015 → 約 0.006 に縮小**。希少 FO クラスの欠損 fold も解消。

### 注意
- **v001 の結果と v003 の結果は直接比較できない**（fold が違う）。過去の fold0 系の数値は v001 のまま解釈する
- **v001 の fold0 は 5 fold 中もっとも不利**だった（number 最多 34.7% / fo_class 最少級 42.5%）。
  これまでの fold0 CV 値は**悲観側にバイアス**していた
- アルゴリズムは貪欲法（動画を質問数の多い順に、目標件数からのズレが最小の fold へ）。
  **コストは「シェアのズレ」でなく「件数の目標(=全体/k)からのズレ」で測ること**。
  シェアで書くと「既に均衡している fold に足すのが最小コスト」となり全動画が 1 fold に吸い込まれる
- seed を変えて `--restarts` 回試し最良解を採る（貪欲の初期順序依存性を散らすため）

## instseg 用 fold — `instseg_v001..v003`（独自 LPT）と `instseg_v004`（VQA 揃え）

| version | 切り方 | 用途 |
|---|---|---|
| instseg_v001〜v003 | 動画単位 LPT（positive 数を fold 間で均す）3-fold | 検出器単体の CV |
| **instseg_v004**（2026-08-28）| **VQA fold v003 の fold0 を instseg の val に固定**（2値・K-fold ではない）| **検出器の出力を VQA に流す用途**（FRAME 重畳入力 / PROCEDURE FO 索引）|

**v004 を作った理由**: instseg_v001〜v003 は VQA fold と独立に切られており、
`instseg_v003 train ∩ qa fold v003 fold0(val)` に **8 動画**
（0007/0027/0036/0061/0079-LapChole, 0003/0012/0024-Heico）が残っていた。
検出器が VQA の val 動画を学習していると、重畳入力・索引系の CV が過大評価になる。
擬似 QA 生成（expK00）は最初から「qa fold0 は生成元から除外」で設計してあるので、
instseg の学習もその境界に揃えたのが v004。

- 生成: `python generate_folds_instseg_qaaligned.py --dataset <prepared> --version instseg_v004`
- 中身（`s30dall_20260828` 基準）: val 8 動画 1,779枚 1,540inst / train 104 動画 12,112枚 11,742inst
  （train には**公式 VQA が存在しない未ラベル動画 68 本**を含む）
- 列: `video, fold, dataset, qa_fold, n_pos, n_neg, n_inst`
- ⚠️ **最終提出用の検出器は全動画で学習した方が強い**（テストは未知動画）。
  v004 は「CV を honest にするための split」であって、提出用の重みとは別に持つ


## 問題単位 split — `qa_{version}/qa_split.csv`

`folds.csv` は**動画単位**しかないため、学習サブセットが `build_samples(limit=N, shuffle_seed=42)` という
**暗黙の定義**になっていた（中身を検査できない・入れ子性を確認できない・他トラックを混ぜられない）。
`generate_qa_split.py` で **qID 単位に materialize** する。

- 生成: `python workspace/fold/generate_qa_split.py --version v003 --fold-version v003 --tracks FRAME`
- 読み出し: `import qa_split; qa_split.load(fold=0, part="train", limit=8000)`
- `order` 列（seed 固定シャッフル順位）により **N を増やすと必ず入れ子**（n1000 ⊂ n4000 ⊂ n8000）
- `track` / `duration` 列があるので **SEGMENT / PROCEDURE を足すのは `--tracks FRAME SEGMENT` だけ**

**★一意キーは `uid = "{dataset}:{qID}"`。qID はデータセット間で衝突する**（FRAME で 20 件確認:
heico と lapchole に同じ qID が存在し、別動画の別質問）。qID 単体で索引を作ると片方が黙って上書きされる。
