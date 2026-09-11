# v013 — SEGMENT + **instseg 重畳(s1)** + 質問文ルール(segrules)
#   CV（SEGMENT val 3,925問, 32f@768+anchor, judge込み, 対照は同一マシン再走）:
#     ctrl(expE06g) 0.6842 / s0 0.6381 / **s1 0.7047** / s1rules 0.7065 / s2rules 0.6828
#
# ★v003（FRAME）のイメージを FROM して差分だけ足す。
#   base_model は同じ Qwen3.5-9B なので、**19GB のレイヤーを再利用**できる
#   （自前で COPY し直すとディスクもビルド時間も 19GB 分無駄になる。v004 と同じ手口）。
#   v003 が無い環境では `do_build.sh` が止まる。その場合は先に v003 を build すること。
FROM orena-focus-frame-n16000 AS segment-algorithm-amd64

# ★SEGMENT はクリップ mp4 のデコードが必須（FRAME 版は PNG を読むだけで ffmpeg 不要だった）
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER user
WORKDIR /opt/app

# ★★scipy だけ追加で入れる（FROM 元のイメージには入っていない）。
#   Mask2Former は **推論だけでも** `Mask2FormerLoss.__init__` で scipy を要求し、
#   無いと detector が**黙ってロード失敗 → 重畳なし(s0 相当)で走る**
#   （v008 で 2026-08-22、v013 で 2026-09-03 と2度踏んだ穴。CV 換算 −0.067 相当）。
#   requirements.txt 全体を再インストールしないのは、既存の固定版を動かさないため。
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color "scipy>=1.10"

# 64フレーム分の vision tower forward で確保サイズが大きく振れる。断片化で落ちるのを防ぐ
# （本番の L40S 48GB では余裕があるが、混雑した共有GPUでの回帰テストでは効く）。
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── モデル資産（base_model は FROM 側のものをそのまま使う）──────────────────
# resources/adapter              : LoRA adapter(~215MB, expE06g joint FRAME+SEGMENT 全量)
#                                  ★FROM 側の FRAME adapter を上書きする
# resources/group_templates.json : 質問文テンプレ → capability group の表（振り分け用）
COPY --chown=user:user resources/adapter_n04 /opt/app/resources/adapter_n04
COPY --chown=user:user resources/adapter_r00c /opt/app/resources/adapter_r00c
COPY --chown=user:user resources/adapter_r00 /opt/app/resources/adapter_r00
COPY --chown=user:user resources/group_templates.json /opt/app/resources/

# ── ★★instseg 検出器（v013 の追加分。+1.6GB）─────────────────────────────
#   → in-place s1 重畳を描く。CV: s0 0.6381 → **s1 0.7047**（+0.0666）

COPY --chown=user:user prompts_seg.py /opt/app/
COPY --chown=user:user sampling.py /opt/app/
COPY --chown=user:user video.py /opt/app/
COPY --chown=user:user router.py /opt/app/
COPY --chown=user:user answer_norm.py /opt/app/
COPY --chown=user:user time_postproc.py /opt/app/
COPY --chown=user:user time_count_prior.json /opt/app/
COPY --chown=user:user postprocess.py /opt/app/
COPY --chown=user:user inference.py /opt/app/

# ★grand-challenge は **イメージ digest でアカウント横断の重複判定**をする。
#   同じ digest は一度しか登録できないので、上げ直すときは
#   `BUILD_ID=r2 bash do_build.sh` で末尾に極小レイヤーを1枚足す（重い層は再利用される）。
ARG BUILD_ID=r1
RUN echo "${BUILD_ID}" > /opt/app/resources/.build_id

ENTRYPOINT ["python", "inference.py"]
