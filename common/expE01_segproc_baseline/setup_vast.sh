#!/usr/bin/env bash
# 貸しGPU（vast.ai 等）の初期セットアップ。**べき等**なので何度流しても良い。
#
# 前提: Ubuntu + NVIDIA ドライバ導入済み、CUDA 12.x、sm_80 以上（FlashAttention2 のため）。
# 使い方:
#   bash setup_vast.sh            # 環境構築 + 疎通確認
#   bash setup_vast.sh --check    # 確認だけ
set -euo pipefail
REPO="${REPO:-$HOME/Orena}"
step(){ echo "########## $(date '+%F %T') $* ##########"; }

if [ "${1:-}" != "--check" ]; then
  step "システム依存（ffmpeg は フレーム抽出に必須）"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq ffmpeg git curl build-essential || true

  step "uv 導入"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"

  step "Python 環境（.venv）"
  cd "$REPO"
  uv venv --python 3.12 .venv
  # ★torch は CUDA 版を明示。cu124/cu126 はドライバに合わせる（nvidia-smi の CUDA Version 参照）
  uv pip install --python .venv/bin/python \
    "torch" "torchvision" --index-url https://download.pytorch.org/whl/cu124
  uv pip install --python .venv/bin/python \
    "transformers==5.14.1" "accelerate" "peft" "bitsandbytes" \
    "qwen-vl-utils" "pandas" "pyarrow" "pyyaml" "scipy" "tabulate" "pillow" "datasets" \
    "opencv-python-headless"
  # ★opencv: 重畳(overlay3/overlay_render)が cv2 を import する。無いと
  #   `ModuleNotFoundError: No module named 'cv2'` で **DDP 全ランクが即死**する
  #   （2026-09-02 expN01 で実害）。headless 版でよい（GUI は使わない）。
  # ★★`flash-linear-attention` を必ず入れる（2026-08-17 に判明）。
  #   Qwen3.5 系は linear_attention 層が主体（27B は 64層中48層）で、これが無いと
  #   transformers が警告1行だけ出して**素の PyTorch 実装（fallback）**で走る。
  #   実測 Marlin-2B/4090: **51.9 → 13.8 s/it = 3.8×**。
  #   dl1 の `.venv-dl1` には入っていたので気づかず、**expV09(27B) が57時間 fallback で走った**。
  uv pip install --python .venv/bin/python "flash-linear-attention"
  # ★flash-attn は **ビルドに torch が必要**なので必ず後入れ＋ --no-build-isolation。
  #   これを忘れると "ModuleNotFoundError: No module named 'torch'" でコケる。
  # ★PyPI の `flash-attn` は torch の ABI と噛み合わないビルドを掴むことがある。
  #   2026-09-02 vast(torch 2.6.0+cu124/py312): 2.8.3.post1 が入り
  #   `undefined symbol: _ZN3c105Error...` で import 失敗 → trainer が **WARNING 1行だけ出して sdpa に落ちる**。
  #   expM03 / expN00 / expM03A の3台が気づかず sdpa で走っていた。
  #   → 公式 release の **torch/ABI/py 一致 wheel を URL 指定**する。合わなければ従来ビルドにフォールバック。
  step "flash-attn（release wheel を torch/ABI に合わせて指定）"
  FA_ABI=$(.venv/bin/python -c "import torch;print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
  FA_TORCH=$(.venv/bin/python -c "import torch;print('.'.join(torch.__version__.split('.')[:2]))")
  FA_PY=$(.venv/bin/python -c "import sys;print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch${FA_TORCH}cxx11abi${FA_ABI}-${FA_PY}-${FA_PY}-linux_x86_64.whl"
  uv pip install --python .venv/bin/python --no-deps "$FA_URL" || \
    uv pip install --python .venv/bin/python flash-attn --no-build-isolation || \
    echo "!!! flash-attn 導入失敗 → sdpa にフォールバックして続行（速度・メモリは不利）"
  # ★import まで確認する（入っただけでは駄目。ABI 不一致は import で初めて落ちる）
  .venv/bin/python -c "import flash_attn; print('flash-attn import OK', flash_attn.__version__)" || \
    echo "!!! ★flash-attn は import できない。sdpa で走る（s/it と VRAM が悪化する）"
  uv pip install --python .venv/bin/python -e reference/ || true
fi

step "疎通確認"
cd "$REPO"
.venv/bin/python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
n = torch.cuda.device_count()
for i in range(n):
    cc = torch.cuda.get_device_capability(i)
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name} sm_{cc[0]}{cc[1]} {p.total_memory/1e9:.0f}GB "
          f"FlashAttention={'可' if cc[0]>=8 else '不可(sdpa)'}")
print("transformers", transformers.__version__)
try:
    import flash_attn; print("flash-attn", flash_attn.__version__)
except Exception as e:
    print("flash-attn なし:", type(e).__name__)
import shutil; print("ffmpeg", "あり" if shutil.which("ffmpeg") else "★無い（抽出できない）")
PY
step "データ確認"
F="$REPO/workspace/expE01_segproc_baseline/frames_cache/768"
[ -d "$F" ] && echo "  frames_cache/768: $(du -sh $F|cut -f1) / $(find $F -name '*.jpg'|wc -l) 枚" \
            || echo "  ★frames_cache/768 が無い（転送されていない）"
step "SETUP DONE"
