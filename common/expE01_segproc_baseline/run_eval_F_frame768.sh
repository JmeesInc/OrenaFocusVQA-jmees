#!/usr/bin/env bash
# expE03f（joint 全量, FRAME は 448px 学習）を **FRAME 768px** で評価する。
# 狙い: PROCEDURE で「学習を4.4倍にしても動かなかった temporal が、推論時にフレームを増やすだけで
#       1.7倍になった」のと同じ論理。FRAME は1枚入力なので増やせるのは**解像度**。
# ★train/test ミスマッチを意図的に許可（--allow-resolution-mismatch）。出力タグに 768 を明記。
# ★対照は同一モデルの 448px 評価（0.5439/0.5586）＝ paired 比較になる。
set -uo pipefail
cd /data4/src/shunsuke/MICCAI2026/Orena
export PYTHONPATH="$PWD/reference/src"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY="$PWD/.venv-dl1/bin/python"
AD=workspace/expE01_segproc_baseline/results/expE03f_joint_all_full/fold0/adapter
OUT=workspace/expE03f_frame_eval; TAG=f_joint_full_frame768
step(){ echo "=== $(date '+%F %T') $* ==="; }
step "EVAL expE03f on FRAME @768px (qa_v004, 学習は448px)"
stdbuf -oL -eL "$PY" workspace/expD00_lora_frame/eval_lora.py \
  --adapter "$AD" --qa-version v004 --qa-fold 0 --qa-part val \
  --frame-size 768 --allow-resolution-mismatch --limit 1500 --per-dataset \
  --out-dir "$OUT" --out-tag "$TAG"
step "EVAL rc=$?"
for ds in heico lapchole; do
  d=$(ls -d $OUT/*${TAG}_$ds 2>/dev/null | head -1)
  [ -n "$d" ] && stdbuf -oL -eL "$PY" workspace/expB00_qwen35_zeroshot/eval_with_judge.py "$d" --dataset $ds --split train
done
step "FRAME768 DONE"
