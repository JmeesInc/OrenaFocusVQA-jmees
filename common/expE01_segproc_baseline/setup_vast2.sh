#!/usr/bin/env bash
# 借り GPU の環境構築。**dl1 の .venv-dl1 と同じバージョンに固定**する。
#   （setup_vast.sh は torch を無指定で入れるので、実機と別バージョンになりうる。
#     学習の挙動を dl1 と揃えたいのでここでは全部ピンする）
#
#   bash setup_vast2.sh          # 構築 + 疎通確認
#   bash setup_vast2.sh --check  # 確認だけ
set -uo pipefail
REPO="${REPO:-/workspace/Orena}"
step(){ echo "########## $(date '+%F %T') $* ##########"; }

if [ "${1:-}" != "--check" ]; then
  step "システム依存"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq ffmpeg git curl build-essential || true

  step "uv"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  step "venv（python 3.12 = dl1 と同じ）"
  cd "$REPO"
  uv venv --python 3.12 .venv
  P=.venv/bin/python
  # ★dl1(.venv-dl1) と同一バージョン。sm_89 は cu126 wheel でカバーされる。
  uv pip install --python $P "torch==2.13.0" torchvision --index-url https://download.pytorch.org/whl/cu126
  uv pip install --python $P \
    "transformers==5.14.1" "accelerate==1.14.0" "peft==0.19.1" "bitsandbytes==0.50.0" \
    "qwen-vl-utils" "pandas" "pyarrow" "pyyaml" "scipy" "tabulate" "pillow" "datasets" \
    "huggingface_hub[cli]" "hf_transfer"
  # ★flash-attn は torch が入ってから + --no-build-isolation。
  #   失敗しても sdpa で動く（FRAME は1枚 ≒900 token なので実害はほぼ無い）。
  step "flash-attn（失敗しても続行）"
  uv pip install --python $P flash-attn --no-build-isolation || \
    echo "!!! flash-attn 失敗 → sdpa にフォールバック"
  uv pip install --python $P -e reference/ || true
fi

step "疎通確認"
cd "$REPO"
.venv/bin/python - <<'PY'
import torch, transformers, shutil
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    cc = torch.cuda.get_device_capability(i); p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} sm_{cc[0]}{cc[1]} {p.total_memory/2**30:.1f}GiB "
          f"FlashAttention={'可' if cc[0]>=8 else '不可(sdpa)'}")
print("  arch_list:", torch.cuda.get_arch_list())
print("transformers", transformers.__version__)
for m in ("peft","bitsandbytes","accelerate","flash_attn"):
    try:
        mod=__import__(m); print(" ", m, getattr(mod,"__version__","?"))
    except Exception as e: print(" ", m, "★無し", type(e).__name__)
import focus; print("focus(reference) OK")
print("ffmpeg", "あり" if shutil.which("ffmpeg") else "★無い")
PY
step "SETUP DONE"
