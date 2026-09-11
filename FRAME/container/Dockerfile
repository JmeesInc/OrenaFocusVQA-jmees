# ★v003（FRAME）のイメージを FROM して差分だけ足す。
#   base_model は同じ Qwen3.5-9B なので **19GB のレイヤーを再利用**できる。
FROM orena-focus-frame-n16000 AS frame-ood-amd64

USER user
WORKDIR /opt/app

# ★scipy だけ追加で入れる（FROM 元のイメージには入っていない）。
#   Mask2Former は **推論だけでも** `Mask2FormerLoss.__init__` で scipy を要求し、
#   無いと detector が fail-soft して**重畳なしで静かに走る**（2026-08-22 に実害）。
#   requirements.txt 全体を再インストールしないのは、既存の固定版を動かさないため。
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color "scipy>=1.10"

# ── モデル資産（base_model は FROM 側のものをそのまま使う）──────────────────
# ★v017: 経路A = v011 の3本（実測構成）/ 経路B = all-data 3本（ID×aggregation 専用）
# resources/m1_m03b_overlay : expM00A（r1 描画で学習）**1024px 推論** CV 0.6581
# resources/m2_m03b_768     : expM00A（同一 adapter・768px 推論）暫定
# resources/m3_v06_r32      : expV06_frame_r32       ( 768px)
# resources/detector        : expF40 M2F 7クラス (~823MB)
# resources/detector_clip   : expF39 M2F clip specialist (~823MB, 768x1344)
COPY --chown=user:user resources/m1_m03b_overlay /opt/app/resources/m1_m03b_overlay
COPY --chown=user:user resources/m2_m03b_768 /opt/app/resources/m2_m03b_768
COPY --chown=user:user resources/m3_v06_r32 /opt/app/resources/m3_v06_r32
COPY --chown=user:user resources/b1_q02b_r1 /opt/app/resources/b1_q02b_r1
COPY --chown=user:user resources/b2_q00b_r4 /opt/app/resources/b2_q00b_r4
COPY --chown=user:user resources/b3_q03b_r1sa /opt/app/resources/b3_q03b_r1sa
COPY --chown=user:user resources/detector /opt/app/resources/detector
COPY --chown=user:user resources/detector_clip /opt/app/resources/detector_clip

COPY --chown=user:user prompts.py /opt/app/
COPY --chown=user:user answer_norm.py /opt/app/
COPY --chown=user:user ensemble.py /opt/app/
COPY --chown=user:user detector.py /opt/app/
# ★学習コードの逐語コピー（書き換え禁止）
COPY --chown=user:user overlay_render.py /opt/app/
COPY --chown=user:user postprocess.py /opt/app/
COPY --chown=user:user inference.py /opt/app/

# ★grand-challenge は **イメージ digest でアカウント横断の重複判定**をする。
#   Failed 提出でも digest は消費されるので、**上げ直すときは必ず** 
#   `BUILD_ID=r2 bash do_build.sh` で末尾に極小レイヤーを1枚足すこと。
ARG BUILD_ID=r1
RUN echo "${BUILD_ID}" > /opt/app/resources/.build_id

ENTRYPOINT ["python", "inference.py"]
