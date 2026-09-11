#!/usr/bin/env bash
# expQ03（r1 + SurgAtlas + all-data）を 4×4090 DDP で Stage A → B。
# ★[[ddp-ranks-squat-on-gpu0]] ddp_wrap.sh は必須。torchrun には --no-python が要る。
# ★grad_accum は config 側で 4（= 1×4×4 = 有効バッチ 16、単体の accum16 と一致）。
set -uo pipefail
REPO=/workspace/Orena; cd "$REPO"
NP="${1:-4}"
export PYTHONPATH="$REPO/reference/src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export HF_HOME="$REPO/.hf" HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3
export ORIG_CVD=0,1,2,3
Q=$REPO/workspace/expQ03_alldata_sa
WRAP=$REPO/workspace/expE01_segproc_baseline/ddp_wrap.sh
chmod +x "$WRAP" 2>/dev/null
step(){ echo "########## $(date '+%F %T') $* ##########"; }

run_stage(){  # $1=ラベル $2=config $3=log
  step "$1 開始"
  grep -E "^  name:|train_part|expect_dual|variant:|grad_accum|eval_steps|^  r:|lr:" "$2"
  stdbuf -oL -eL "$REPO/.venv/bin/python" -m torch.distributed.run \
    --nproc_per_node="$NP" --master_port=29551 --no-python \
    "$WRAP" "$REPO/.venv/bin/python" \
    "$REPO/workspace/expM03_round1_external/train_lora_ext.py" --config "$2" > "$3" 2>&1
  local rc=$?; echo "=== $1 rc=$rc $(date) ===" >> "$3"
  [ $rc -ne 0 ] && { echo "★$1 失敗 rc=$rc"; return $rc; }
  # ★ログが書いた実パスから adapter を解決（expM04 の smoke 取り違え対策）
  local AD=$(grep -a "saved adapter to" "$3" | tail -1 | sed 's/.*saved adapter to //')
  echo "  adapter=$AD"
  [ -f "$AD/adapter_model.safetensors" ] || { echo "★adapter が無い"; return 1; }
  case "$AD" in *"/fold0/adapter") return 0 ;; *) echo "★fold0 以外に出た（$AD）"; return 1;; esac
}

run_stage "Q03A(all-data + SurgAtlas 混合)" "$Q/config_Q03A_alldata_sa_stageA.yaml" "$REPO/expQ03A.log" || exit 1
run_stage "Q03B(FOCUS+擬似で焼き付け)"       "$Q/config_Q03B_alldata_sa_stageB.yaml" "$REPO/expQ03B.log" || exit 1
step "Q03 CHAIN DONE"
