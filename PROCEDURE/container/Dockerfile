# ★v003（FRAME）のイメージを FROM して差分だけ足す。
#   base_model は同じ Qwen3.5-9B なので **19GB のレイヤーを再利用**できる。
#   v003 が無い環境では `do_build.sh` が止まる。その場合は先に v003 を build すること。
FROM orena-focus-frame-n16000 AS procedure-algorithm-amd64

# ★PROCEDURE はクリップ mp4 のデコードが必須
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER user
WORKDIR /opt/app

# ★v010 で足りない依存を入れる（**ベースイメージには入っていない**）。
#   2026-09-01 に実測: tensorrt / timm / scikit-learn / flash-linear-attention が欠けており、
#   TRT は "No module named tensorrt" で PyTorch に、工程モデルは timm 不足で無効に落ちていた。
#   どちらも**警告だけ出して動き続ける**ので、イメージの中で import を確認すること。
#   （decord / pyyaml / pandas はベースに入っていた）
COPY --chown=user:user requirements.txt /opt/app/requirements.txt
# ★uv で入れる（pip より大幅に速く、解決も安定する）。
#   flash-attn は**ソースビルドすると 30 分以上**かかるので、
#   torch 2.5 / cu12 / cp311 の**事前ビルド wheel を URL 直指定**する。
#   ⚠️ビルド最後の import 検証を必ず通す。欠けても実行時は警告だけ出て
#     黙って劣化する（TRT→PyTorch / 工程モデル無効）ので、**ビルドで落とす**。
USER root
# ① 必須の依存（欠けると黙って劣化するので、失敗したらビルドを止める）
RUN python -m pip install --no-cache-dir uv==0.9.* \
    && uv pip install --system --no-cache --only-binary=:all: -r /opt/app/requirements.txt
# ★tensorrt は tensorrt_libs の .so を先に掴まないと import できない
#   （fo_index.py が実行時に同じことを ctypes でやる）
RUN python -c "import timm, sklearn, decord, yaml, pandas; print('deps OK')"
# ★TensorRT は NVIDIA の独自インデックスから入れる（PyPI に wheel が無い）。
#   索引が 1.67 倍速くなるが、**失敗しても PyTorch フォールバックで動く**ので続行する。
RUN uv pip install --system --no-cache --index-strategy unsafe-best-match \
      --extra-index-url https://pypi.nvidia.com "tensorrt-cu12==10.*" \
    && python -c "import ctypes,glob,sys; [ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL) for pat in ('libnvinfer.so*','libnvonnxparser.so*') for f in glob.glob(sys.prefix+'/lib/python*/site-packages/tensorrt_libs/'+pat)]; import tensorrt; print('tensorrt', tensorrt.__version__)" \
    || echo '⚠️TensorRT 無し（索引は PyTorch で動作）'
# ②-a flash-linear-attention（Qwen3.5 の高速経路）。Triton 3.1 と合う版を固定して試す。
#     入ると 3.8 倍速いが、合わなければ素の PyTorch 実装で動く（正しさは変わらない）。
RUN for v in 0.1.2 0.2.0 0.1.0; do \
      uv pip install --system --no-cache --only-binary=:all: "flash-linear-attention==$v" \
        && python -c "import fla; print('fla', '$v')" && break || true; \
    done; python -c "import fla" 2>/dev/null || echo '⚠️fla 無し（素の PyTorch 実装で動作）'
# ②-b 任意の高速化（失敗しても続行＝sdpa で動く）。
#    ⚠️ソースビルドさせない。このイメージには nvcc が無く、`causal-conv1d` は必ず失敗する。
RUN uv pip install --system --no-cache --only-binary=:all: \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
      && python -c "import flash_attn; print('flash_attn', flash_attn.__version__)" \
      || echo '⚠️flash_attn 無し（sdpa で動作する）'
USER user

# ★最大 128 フレームの vision tower forward は確保サイズが大きく振れる。断片化対策。
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── モデル資産（base_model は FROM 側のものをそのまま使う）──────────────────
# resources/adapter              : LoRA adapter(~215MB, expE03f joint 全量。PROCEDURE を含む)
#                                  ★FROM 側の FRAME adapter を上書きする
# resources/group_templates.json : 質問文テンプレ → capability group（枚数の梯子を引く）
# resources/time_count_prior.json: time の個数事前分布（採点直前処理）
COPY --chown=user:user resources/adapter /opt/app/resources/adapter
# ★2パス構成の動画側 adapter（FOCUS_TWO_PATH=1 のときだけ使う。既定は OFF なので無害）
COPY --chown=user:user resources/adapter_video /opt/app/resources/adapter_video
# ── v010 で追加: FO 索引（Mask2Former + TRT エンジン）と工程分類 ────────────
COPY --chown=user:user resources/m2f /opt/app/resources/m2f
COPY --chown=user:user resources/m2f_sm89.trt /opt/app/resources/
COPY --chown=user:user resources/m2f_classes.json /opt/app/resources/
COPY --chown=user:user resources/phase_cholec /opt/app/resources/phase_cholec
COPY --chown=user:user resources/phase_heico /opt/app/resources/phase_heico
COPY --chown=user:user resources/phase_lib /opt/app/resources/phase_lib
COPY --chown=user:user resources/group_templates.json /opt/app/resources/
COPY --chown=user:user resources/time_count_prior.json /opt/app/resources/

COPY --chown=user:user prompts_seg.py /opt/app/
COPY --chown=user:user sampling.py /opt/app/
COPY --chown=user:user video.py /opt/app/
COPY --chown=user:user router.py /opt/app/
COPY --chown=user:user answer_norm.py /opt/app/
COPY --chown=user:user time_postproc.py /opt/app/
COPY --chown=user:user fo_index.py /opt/app/
COPY --chown=user:user phase_index.py /opt/app/
COPY --chown=user:user rules_proc.py /opt/app/
COPY --chown=user:user select_frames.py /opt/app/
COPY --chown=user:user clip_reader.py /opt/app/
COPY --chown=user:user inference.py /opt/app/

# ★grand-challenge は **イメージ digest でアカウント横断の重複判定**をする。
#   同じ digest は一度しか登録できないので、上げ直すときは
#   `BUILD_ID=r2 bash do_build.sh` で末尾に極小レイヤーを1枚足す。
#   ⚠️Failed 提出でも digest は消費されるので、**再提出には必ず BUILD_ID を変える**。
ARG BUILD_ID=r1
RUN echo "${BUILD_ID}" > /opt/app/resources/.build_id

ENTRYPOINT ["python", "inference.py"]
