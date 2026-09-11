#!/usr/bin/env bash
# 3メンバーの LoRA adapter を resources/ へ配置する（git 管理外）。
# ★base_model は v003 イメージのレイヤーを再利用するのでコピー不要（Dockerfile 参照）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RES="$ROOT/workspace/expE01_segproc_baseline/results"
M03="$ROOT/workspace/expM03_round1_external/results"
W="$ROOT/workspace/expF00_fo_instseg/vast_48078326_results/results"
F="$ROOT/workspace/expF00_fo_instseg/results"
declare -A SRC=(
  # ── 経路A: v011（LB 0.6260 / 4位）の3本を**そのまま**。実測構成に手を入れない ──
  [m1_m03b_overlay]="$M03/expM03B_frame_stageB/fold0/adapter"
  [m2_m03b_768]="$M03/expM03B_frame_stageB/fold0/adapter"
  [m3_v06_r32]="$RES/expV06_frame_r32/fold0/adapter"
  # ── 経路B: all-data 3本（ID×aggregation 専用）。別々に学習した重みであること ──
  [b1_q02b_r1]="$ROOT/workspace/expQ02_v011_alldata/results/expQ02B_alldata_stageB/fold0/adapter"
  # ★2026-09-11: 1段（expQ00B）→ **FOCUS 焼き付け済み（expQ00C）** に差し替え。
  #   単一因子の実測で焼き付けは +0.0157（expM04A 0.6344 → expM04B 0.6501、段数だけが違う比較）。
  #   これで経路B の3本すべてが「外部で事前学習 → FOCUS で焼き付け」の2段構造に揃う。
  [b2_q00b_r4]="$ROOT/workspace/expQ00_frame_alldata/results/expQ00C_q00b_bake_r4/fold0/adapter"
  [b3_q03b_r1sa]="$ROOT/workspace/expQ03_alldata_sa/results/expQ03B_alldata_sa_stageB/fold0/adapter"
  # ── FO instance segmentation（重畳ヒント源）──
  [detector]="$F/expF40_m2f_ps1sN_noGallstone_0828/fold0/best_model"
  [detector_clip]="$F/expF39_m2f_clip_hires_strongaug_0828/fold0/best_model"
)
mkdir -p "$SCRIPT_DIR/resources"
for name in "${!SRC[@]}"; do
  s="${SRC[$name]}"
  if [ ! -d "$s" ]; then echo "ERROR: $s が無い" >&2; exit 1; fi
  rm -rf "$SCRIPT_DIR/resources/$name"
  cp -r "$s" "$SCRIPT_DIR/resources/$name"
  chmod -R u+rw "$SCRIPT_DIR/resources/$name"
done
# ★★adapter を取り違えていないか **md5 で検算**する（smoke の残骸を掴む事故が実在）
declare -A WANT=(
  [m1_m03b_overlay]=c0f2795a606cd9d04e219afce0d617b2
  [m2_m03b_768]=c0f2795a606cd9d04e219afce0d617b2
  [b1_q02b_r1]=7a5b2ba9944e4c61accc87b01a6d67dd
  [b2_q00b_r4]=c069a8f12a022eeb5a524219571a2a69
  [b3_q03b_r1sa]=297a1fd92e7b570fc91630dc00e46ecc
)
for n in "${!WANT[@]}"; do
  got=$(md5sum "$SCRIPT_DIR/resources/$n/adapter_model.safetensors" | cut -d" " -f1)
  [ "$got" = "${WANT[$n]}" ] || { echo "ERROR: $n md5 が違う got=$got want=${WANT[$n]}" >&2; exit 1; }
  echo "  ✓ $n md5 一致"
done
du -sh "$SCRIPT_DIR/resources"/*
