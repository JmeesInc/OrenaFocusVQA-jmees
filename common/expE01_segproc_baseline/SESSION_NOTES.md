# expE01 — SEGMENT / PROCEDURE ベースライン

前提の構造把握は `workspace/expE00_segproc_eda/` を参照（配点構造・prior 床・`time` の許容誤差）。

## この実験でやること
1. 形式判定器を SEGMENT/PROCEDURE 用に作り直す（`prompts_seg.py`）
2. 複数フレーム入力のデータセットを作る（`dataset_seg.py`）
3. Qwen3.5-9B zero-shot で床を測る（`run_infer.py` → `eval_seg.py`）

## 決めたこと と その理由

### 1. 形式判定器を作り直した（expB01 の流用は不可）
expB01 の `detect_format` は **FRAME 99.76% だが SEGMENT 81.5% / PROCEDURE 84.0%**。
形式判定を外すと誤った OUTPUT RULE を渡し、`verify` に落ちて **0点**になる。
実測した誤判定の内訳と、それを分離する弁別子（50,000問で検証）:

| GT → 誤判定 | n | 弁別子 | 精度 |
|---|---|---|---|
| fo_class → open_ended | 3355 | **引用符なし** `answer (with) none` = fo_class / **引用符つき** `'none'` = open_ended | 11531/11554, 263/263 |
| multiple_choice → open_ended | 1482 | `please select`（`select one answer` だけでは足りない。`select one or multiple` 形式がある）| — |
| open_ended → number | 328 | `provide a single integer` = **open_ended** / `provide a/the number` = number | 192/192, 9862/9862 |
| percentage → number | 135 | `in %` / `xx%` / `percentage` を `how many` より**先に**判定 | 135/135 |

→ `prompts_seg.py` の `_RULES` は **順序そのものが仕様**。結果:

| track | expB01 | **prompts_seg (v2)** |
|---|---|---|
| FRAME | 0.9976 | **0.9998** |
| SEGMENT | 0.8148 | **0.9998** |
| PROCEDURE | 0.8398 | **0.9978** |

残る誤りは「GT は open_ended だが文面はクラス名を要求している」30問で、fo_class 規則で出しても
judge が通す見込みなので実害は小さい。

### 2. 公式 `FocusVideoDataset` を使わない
1サンプルごとに ffmpeg でクリップを再エンコードして /tmp に MP4 を書く実装。
`[0, 17780]s` の PROCEDURE で毎回やると学習が回らない。
FRAME と同じく **必要な時刻のフレームだけ JPEG 抽出してキャッシュ**する。

**キャッシュキーは (videoID, 時刻)。qID ではない。**
同じ動画の重なったクリップへの質問が大量にあるので、時刻を **1秒グリッドにスナップ**して共有する。
実測: SEGMENT fold0 val 4,006問 × 16フレーム = 64,096 スロット → **distinct 26,472枚（2.4倍の圧縮）**。

### 3. タイムスタンプは焼き込まず、各フレームの直前にテキストで置く
公式の `VideoTimestampOverlayPreprocessor` は全動画を再エンコードする（数百GB）。
そもそも **`Request.start_time` は推論時にも与えられる**＝時刻はこちらが知っている情報なので、
モデルに OCR させる理由がない。テキストで渡す方が正確でコストもゼロ。
※提出コンテナでも同じ渡し方にすること（学習と推論で入力形式を揃える）。

### 4. time 回答は [start, end] にクランプする
expE00 実測で **正解が区間内にある割合は SEGMENT 0.966 / PROCEDURE 1.000**。
区間外の予測は確実に外れなので、端に寄せる後処理は**常に非負の期待値**を持つ。
`run_infer.py --no-clamp` で対照が取れる。

## 落とし穴（実際に踏んだもの）

### ★dl1 の GPU はこの venv では使えない（2026-08-05）
`.venv` の torch は **2.13.0+cu130**（CUDA 13 系）。ドライバは
**dl1 = 560.35.05（CUDA 12.6）→ `torch.cuda.is_available() == False`**、
**dl2 = 580.82.09 → OK**。
`nvidia-smi` は GPU を出すので気づきにくく、`device_map="cuda"` まで進んでから落ちる。
→ **学習・推論は dl2 の 4090（`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2`）で回す**。
dl1 は **フレーム抽出（ffmpeg = CPU/IO）専用**に使う（共有 FS なので dl2 から見える）。

### ★並列抽出の tmp ファイル名には pid+tid を必ず入れる
`out.with_suffix(".tmp.jpg")` の固定名だと、**同じ抽出を2プロセスで走らせたとき**互いの tmp を
rename し合って `FileNotFoundError` で落ちる。キャッシュ埋めは並列に流したくなるので、
衝突しない名前が前提条件。

### 5. Collator の検証（GPU 不要で先に潰した）
`train_lora_seg.py` の `VideoCollator` を CPU だけで検証（`check_seq_lengths`）:

- **系列長 実測 (n=32): min 2138 / p50 2830 / max 3296**（16フレーム@448px）→ `max_seq_len: 4096` で足りる
- `pixel_values (7168, 1536)` / `image_grid_thw (16, 3)` ＝ **16フレームが正しく入っている**
- **教師トークンは `'00:37:18<|im_end|>\n'` の10トークンのみ**（GT `00:37:18`）＝ prompt マスクが正しい

★複数フレームでは系列長がフレーム数×解像度で大きく変わり、`max_seq_len` 不足だと
**answer が切り落とされて静かに学習が壊れる**。学習開始前に必ず実測すること（`check_seq_lengths`）。

⚠️ **collator は processor を1サンプルあたり2回呼ぶ**（prompt 用と full 用）ので、
16フレームだと画像前処理が 32枚ぶん走る。CPU 実測 5.3s/sample（抽出と IO を食い合う状況下）。
GPU step とオーバーラップさせるため `dataloader_num_workers` は 4 以上を維持する。

### 6. GPU の割り当て（dl2 は3枚あるが 4090 は1枚だけ）
| GPU | 用途 |
|---|---|
| idx2 RTX 4090 24GB | **推論・学習を直列に**（`chain_dl2.sh` が順番に流す）|
| idx0 RTX A4000 16GB | **judge 採点**（Qwen3.5-4B なので載る。`chain_judge.sh`）|
| idx1 RTX 4000 8GB | 空き |

judge を素直に 4090 で回すと推論を止めてしまうので、**A4000 に逃がして並行**させる。

### 7. 採否判定は必ず matched + McNemar（`compare_runs.py`）
FRAME で「バケット平均 +0.046」だけ見てヒント注入を採用し LB −0.020 を食った前例がある
（matched で再検算すると 改善51/悪化38, p=0.203 で最初から非有意だった）。
`compare_runs.py` は **(改善数, 悪化数, McNemar p)** と **バケット非加重平均**を同時に出す。
★SEGMENT/PROCEDURE は小バケットの N が二桁なので、**バケット別の差には必ず N を併記**する。

## 実行

```bash
# フレーム抽出（dl1 で可。CPU/IO のみ）
.venv/bin/python workspace/expE01_segproc_baseline/dataset_seg.py \
  --track SEGMENT --fold 0 --part val --limit 0 --n-frames 16 --size 448 --workers 32

# 推論（dl2 の 4090）
ssh dl2 'cd /mnt/data/data4/src/shunsuke/MICCAI2026/Orena && setsid nohup env \
  PYTHONPATH=$PWD/reference/src CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  .venv/bin/python workspace/expE01_segproc_baseline/run_infer.py \
  --track SEGMENT --fold 0 --n-frames 16 --size 448 --out-tag zeroshot_16f448 \
  > workspace/expE01_zs.log 2>&1 < /dev/null &'

# 採点（judge 込み）
ssh dl2 '... .venv/bin/python workspace/expE01_segproc_baseline/eval_seg.py \
  workspace/expE01_segproc_baseline/results/zeroshot_16f448 --track SEGMENT'
```

## 結果（SEGMENT fold0 val, 公式 Evaluator + LLM judge, 重複動画0027は除外）

| アーム | N | heico | lapchole | **SCORE** | 備考 |
|---|---|---|---|---|---|
| prior（映像を一切見ない）| 3925 | 0.3118 | 0.3780 | **0.3360** | 採用の下限 |
| zero-shot 16f@448 | 3925 | 0.3734 | 0.4511 | **0.4078** | latency 1.29s |
| hybrid（number/fo_class を prior に）| 3925 | 0.4060 | 0.4788 | **0.4396** | +0.0318（AGG +0.089\*\*\*）|
| **FRAME LoRA(expD11) 16f@448** | 1958 | 0.5017 | 0.4606 | **0.4946** | **matched +0.0860** |

### 主要な知見

1. **FRAME で学習した LoRA は SEGMENT にそのまま効く**（追加学習ゼロで +0.0860）。
   obj_recognition **+0.312\*\*\***、agg +0.113\*、temporal +0.038\*。
   形式別 fo_class **+0.376\*\*\*** / MC **+0.327\*\*\*** / number **+0.196\*\*\***
2. **`time` は壊れなかった**（+0.027 n.s.）。FRAME に `time` は1問も無いのに劣化しない
3. **唯一の劣化 open_ended −0.121\* は「`none` と言えない」1つの失敗モード**:
   GT=`none` の12問で zero-shot 0.917 → LoRA **0.000**。空フレームに物体を捏造する。
   FRAME の空間定位問題は必ず対象物が在るため「FO は常に在る」を学んだ。
   **転移元に存在しない答えは転移先でも出せない**
4. **書式正規化だけで +0.035**（`'1.'` / `'Yes.'` が `verify` に落ちて自動0点だった）
5. **`time` の失点は「個数」と「刻み」に分解できる**:
   個数不一致で17.9%、刻みが±5sより粗い問題が16fで75.4%（64fで0%）

### 反証可能な予測（学習後に検証する）
- **SEGMENT 学習で `none` が言えるようになる**（SEGMENT には GT=`none` の同型質問がある）
- **64フレーム化で長クリップ帯の time が 0.23 → 0.45 前後**（刻みが±5s以下になるため）
