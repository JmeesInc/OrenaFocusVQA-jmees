#!/usr/bin/env bash
# PROCEDURE をキーフレーム格子（5s）に縛った学習と、その対照（1s 格子）。
#
#   A = expE05a_proc_from_joint_kf5          （config_G, grid 5.0）★本命
#   B = expE05b_proc_from_joint_grid1_ctrl   （config_H, grid 1.0）対照
#   E = 推論だけの格子比較（学習なし。expE02c を PROCEDURE val に当てる）
#
# ★A と B の config 差分は **name と grid の2行だけ**（`diff` で確認済み）。
# ★どちらも expE02c（qa_v004 = fold v003 の fold≠0 で学習）から warm start するので、
#   qa_v004 fold0 val で評価すれば clean。
# ★推論側の対照 `eval_expE02c_joint_16fproc448_n2000`（8/8 実施, grid 1s）は既にあるので、
#   E は grid=5 のアームだけを回して matched 比較する。greedy(do_sample=False)なので
#   同一入力なら出力は決定的＝差はフレーム選択だけに帰着する。
#
# Usage: GPU=0 run_proc_kf.sh <A|B|E>
set -uo pipefail
cd /data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv-dl1/bin/python"     # ★dl1 の GPU は cu126 の .venv-dl1 でないと見えない
D="$PWD/workspace/expE01_segproc_baseline"
R="$D/results"
step(){ echo "=== $(date '+%F %T') $* ==="; }

case "${1:-}" in
  A) CFG=$D/config_G_proc_from_joint_kf5.yaml ;;
  C) CFG=$D/config_I_proc_ep2.yaml ;;   # expE05a の続き（2エポック目、8/10 打ち切り）
  J) CFG=$D/config_J_frame_specialist_v003.yaml ;;  # FRAME 特化を fold v003 で（joint との決着用）
  L) CFG=$D/config_L_frame_3ep.yaml ;;   # expE10 を 3ep へ継続（checkpoint-2000 から resume）
  M) CFG=$D/config_M_frame_visionrank.yaml ;;   # FRAME: vision へ rank 再配分（params 据え置き）
  K) CFG=$D/config_K_frame_from_fs.yaml ;;   # FRAME を expE06g(F+S) から warm start
  B) CFG=$D/config_H_proc_from_joint_grid1_ctrl.yaml ;;
  E)
    AD="$R/expE02c_joint_frame_segment/fold0/adapter"
    TAG="eval_expE02c_joint_16fproc448_kf5_n2000"
    step "INFER PROCEDURE val 2000 @ grid=5（対照は既存の $TAG の grid=1 版）"
    stdbuf -oL -eL "$PY" "$D/run_infer.py" --track PROCEDURE --fold 0 --part val \
      --limit 2000 --n-frames 16 --size 448 --version v004 --grid 5 \
      --adapter "$AD" --out-tag "$TAG"
    rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && exit 1
    step "JUDGE"
    stdbuf -oL -eL "$PY" "$D/eval_seg.py" "$R/$TAG" --track PROCEDURE
    step "JUDGE rc=$?"
    step "COMPARE grid=1 vs grid=5 (matched, McNemar)"
    stdbuf -oL -eL "$PY" "$D/compare_runs.py" \
      "$R/eval_expE02c_joint_16fproc448_n2000" "$R/$TAG" --labels grid1 grid5
    exit $? ;;
  F)
    # expE03f（全量 joint 39,986 = F15992+S15994+P8000、8/9 13:10 完了、best=checkpoint-1000）を
    # PROCEDURE で評価する。★expE03f も qa_v004 fold0 学習なので v004 fold0 val は clean。
    # ★grid=5 で回す（E で「格子は SCORE に無影響」を確認済み。以後の標準に揃える）。
    #   対照は同じ grid=5 の expE02c 版 = eval_expE02c_joint_16fproc448_kf5_n2000。
    AD="$R/expE03f_joint_all_full/fold0/adapter"
    TAG="eval_expE03f_joint_16fproc448_kf5_n2000"
    step "INFER expE03f PROCEDURE val 2000 @ grid=5"
    stdbuf -oL -eL "$PY" "$D/run_infer.py" --track PROCEDURE --fold 0 --part val \
      --limit 2000 --n-frames 16 --size 448 --version v004 --grid 5 \
      --adapter "$AD" --out-tag "$TAG"
    rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && exit 1
    step "JUDGE"
    stdbuf -oL -eL "$PY" "$D/eval_seg.py" "$R/$TAG" --track PROCEDURE
    step "JUDGE rc=$?"
    step "COMPARE expE02c vs expE03f (どちらも grid=5, matched)"
    stdbuf -oL -eL "$PY" "$D/compare_runs.py" \
      "$R/eval_expE02c_joint_16fproc448_kf5_n2000" "$R/$TAG" --labels E02c E03f
    exit $? ;;
  N)
    # ★フレーム数掃引。PROCEDURE の temporal は**学習データでなくフレーム数で動く**
    #   （expE03f: 16f 0.0676 → 32f 0.1092 / expE02c: 0.0585 → 0.1014）。
    #   latency は 32f でも 4.90s / 上限 30s ＝ 6倍の余裕があるので上を見る。
    #   Usage: GPU=0 NF=64 run_proc_kf.sh N
    NF="${NF:?NF（フレーム数）を指定すること}"
    # RUN で評価対象を、SHORT でタグの短縮名を切り替える（既定は expE03f）。
    # ★既存タグ `eval_expE03f_joint_<NF>fproc448_kf5_n2000` を変えないこと
    #   （compare_runs.py の参照先とディレクトリ名が壊れる）。
    RUN="${RUN:-expE03f_joint_all_full}"; SHORT="${SHORT:-expE03f_joint}"
    AD="$R/$RUN/fold0/adapter"
    TAG="eval_${SHORT}_${NF}fproc448_kf5_n2000"
    [ -f "$AD/adapter_model.safetensors" ] || { echo "!!! adapter が無い: $AD"; exit 1; }
    step "INFER expE03f PROCEDURE val 2000 @ ${NF}f grid=5"
    stdbuf -oL -eL "$PY" "$D/run_infer.py" --track PROCEDURE --fold 0 --part val \
      --limit 2000 --n-frames "$NF" --size 448 --version v004 --grid 5 \
      --adapter "$AD" --out-tag "$TAG"
    rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && exit 1
    step "JUDGE"
    stdbuf -oL -eL "$PY" "$D/eval_seg.py" "$R/$TAG" --track PROCEDURE
    step "JUDGE rc=$?"
    # 対照は BASE で指定（既定 = expE03f の 32f）。フレーム掃引なら同じモデルの隣の枚数、
    # モデル比較なら同じ枚数の expE03f を指すこと。
    BASE="${BASE:-eval_expE03f_joint_32fproc448_kf5_n2000}"
    step "COMPARE $BASE vs $TAG (grid=5, matched)"
    stdbuf -oL -eL "$PY" "$D/compare_runs.py" \
      "$R/$BASE" "$R/$TAG" --labels base "${SHORT}_${NF}f"
    exit $? ;;
  FR)
    # ★FRAME トラック評価。joint に PROCEDURE を混ぜて FRAME が落ちたかを見るために必要。
    #   対照は既存の `eval_C_joint_on_FRAME_n2000`（expE02c / N=1962 / 448px / v004 fold0）。
    #   Usage: GPU=1 SIZE=448 run_proc_kf.sh FR   /   GPU=3 SIZE=768 ... FR
    #   ★768 は「学習は 448 のまま、推論だけ高解像度」の条件。SEGMENT では
    #     448→768 で fo_class +0.038** が出た前例がある（expE01）。FRAME 用の 768 フレームは
    #     キャッシュに無ければ自動抽出される（1問1枚なので約2,000枚＝数分）。
    SIZE="${SIZE:-448}"
    RUN="${RUN:-expE03f_joint_all_full}"; SHORT="${SHORT:-expE03f_joint}"
    AD="$R/$RUN/fold0/adapter"
    TAG="eval_${SHORT}_frame${SIZE}_n2000"
    [ -f "$AD/adapter_model.safetensors" ] || { echo "!!! adapter が無い: $AD"; exit 1; }
    step "INFER FRAME val 2000 @ ${SIZE}px（$RUN）"
    stdbuf -oL -eL "$PY" "$D/run_infer.py" --track FRAME --fold 0 --part val \
      --limit 2000 --n-frames 1 --size "$SIZE" --version v004 \
      --adapter "$AD" --out-tag "$TAG"
    rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && exit 1
    step "JUDGE"
    stdbuf -oL -eL "$PY" "$D/eval_seg.py" "$R/$TAG" --track FRAME
    step "JUDGE rc=$?"
    BASE="${BASE:-eval_C_joint_on_FRAME_n2000}"
    step "COMPARE $BASE vs $TAG (matched)"
    stdbuf -oL -eL "$PY" "$D/compare_runs.py" "$R/$BASE" "$R/$TAG" \
      --labels C_joint "${SHORT}_${SIZE}px"
    exit $? ;;
  SG)
    # ★SEGMENT トラック評価。対照は既定で expE03f（F+S+P joint 全量、SCORE 0.5674）。
    #   expE06g（F+S のみ）と比べることで「PROCEDURE を混ぜた害の有無」が出る。
    #   Usage: GPU=1 RUN=expE06g_joint_frame_segment_FULL SHORT=expE06g_joint $0 SG
    SIZE="${SIZE:-448}"; NF="${NF:-16}"
    RUN="${RUN:-expE03f_joint_all_full}"; SHORT="${SHORT:-expE03f_joint}"
    AD="$R/$RUN/fold0/adapter"
    TAG="eval_${SHORT}_seg${NF}f${SIZE}_n2000"
    [ -f "$AD/adapter_model.safetensors" ] || { echo "!!! adapter が無い: $AD"; exit 1; }
    step "INFER SEGMENT val 2000 @ ${NF}f/${SIZE}px（$RUN）"
    stdbuf -oL -eL "$PY" "$D/run_infer.py" --track SEGMENT --fold 0 --part val \
      --limit 2000 --n-frames "$NF" --size "$SIZE" --version v004 \
      --adapter "$AD" --out-tag "$TAG"
    rc=$?; step "INFER rc=$rc"; [ "$rc" != "0" ] && exit 1
    step "JUDGE"
    stdbuf -oL -eL "$PY" "$D/eval_seg.py" "$R/$TAG" --track SEGMENT
    step "JUDGE rc=$?"
    BASE="${BASE:-eval_expE03f_joint_seg16f448_n2000}"
    step "COMPARE $BASE vs $TAG (matched)"
    stdbuf -oL -eL "$PY" "$D/compare_runs.py" "$R/$BASE" "$R/$TAG" \
      --labels base "$SHORT"
    exit $? ;;
  *) echo "usage: GPU=N $0 <A|B|C|J|E|F>  /  NF=64 $0 N  /  SIZE=448 $0 FR  /  $0 SG"; exit 2 ;;
esac

# ★smoke は --train-limit で件数だけ絞る。config の形（train_limit: null）は本番と同一に保つ
#   （2026-08-07 の実害: smoke が limit を書き換えて壊れた経路を隠し、本番が 1.5h 後に落ちた）。
# ★共有 GPU なので、起動前に空きを確認する（2026-08-10 に実害: 停止から 8 分の隙に
#   別プロセスが 45.6GB を確保しており、smoke がモデルロードで OOM した）。
#   ここで弾けば「本番だと思っていたら落ちていた」を防げる。
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU:-0}")
if [ "${free_mb:-0}" -lt 25000 ]; then
  echo "!!! GPU ${GPU:-0} の空きが ${free_mb}MB しかない（25,000MB 必要）。別 GPU を指定すること"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
  exit 1
fi
echo "=== GPU ${GPU:-0} 空き ${free_mb}MB を確認 ==="

SMOKE=/tmp/smoke_prockf_$1.yaml
sed -e 's/^  name: \(.*\)$/  name: SMOKE_\1/' \
    -e 's/eval_subset: [0-9]*/eval_subset: 4/' \
    -e 's/save_steps: [0-9]*/save_steps: 2/' \
    -e 's/eval_steps: [0-9]*/eval_steps: 2/' \
    -e 's/grad_accum: [0-9]*/grad_accum: 2/' "$CFG" > "$SMOKE"
step "SMOKE $1"
stdbuf -oL -eL "$PY" "$D/train_lora_seg.py" --config "$SMOKE" --epochs 1 --train-limit 8
rc=$?; step "SMOKE $1 rc=$rc"
[ "$rc" != "0" ] && { echo "!!! smoke 失敗。本番は流さない"; exit 1; }

step "TRAIN $1 PROCEDURE 全量 8,000問（RTX8000 で推定 25〜30h）"
stdbuf -oL -eL "$PY" "$D/train_lora_seg.py" --config "$CFG"
step "TRAIN $1 rc=$?"
