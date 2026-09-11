#!/usr/bin/env bash
# expE10（FRAME 特化 768px, fold v003）と expE06g（F+S joint 全量）の評価を順に流す。
#
# 決着させたい2問:
#   Q1「FRAME 特化 vs joint」 … expE10 vs expE03f（同一 fold v003・同一学習器）
#       ※ fold v001 の expD11(0.6464) と v003 の expE03f(0.5681) は val が 26本中21本
#         入れ替わっており比較不能だった。それを解消するのがこの実験。
#   Q2「PROCEDURE を混ぜた害の有無」 … expE06g(F+S) vs expE03f(F+S+P)
#
# 実行順は「決着の早い順 × 所要時間の短い順」。FRAME は約1.5s/問、SEGMENT は約3.6s/問。
# Usage: GPU=1 queue_e10_e06g.sh
set -uo pipefail
cd /data4/src/shunsuke/MICCAI2026/Orena
D="$PWD/workspace/expE01_segproc_baseline"
G="${GPU:-1}"
run(){ echo "########## $(date '+%F %T') $* ##########"; GPU="$G" "$@" bash "$D/run_proc_kf.sh"; }

echo "########## $(date '+%F %T') 1/4 expE10 FRAME @768（Q1 本命：特化の得意条件） ##########"
GPU="$G" RUN=expE10_frame_specialist_768_v003 SHORT=expE10_spec SIZE=768 \
  BASE=eval_expE03f_joint_frame768_n2000 bash "$D/run_proc_kf.sh" FR

echo "########## $(date '+%F %T') 2/4 expE06g SEGMENT（Q2 本命：PROC 混入の影響が出やすい） ##########"
GPU="$G" RUN=expE06g_joint_frame_segment_FULL SHORT=expE06g_joint \
  BASE=eval_expE03f_joint_seg16f448_n2000 bash "$D/run_proc_kf.sh" SG

echo "########## $(date '+%F %T') 3/4 expE06g FRAME @768（Q2 の FRAME 側） ##########"
GPU="$G" RUN=expE06g_joint_frame_segment_FULL SHORT=expE06g_joint SIZE=768 \
  BASE=eval_expE03f_joint_frame768_n2000 bash "$D/run_proc_kf.sh" FR

echo "########## $(date '+%F %T') 4/4 expE10 FRAME @448（学習解像度と推論解像度の分離） ##########"
GPU="$G" RUN=expE10_frame_specialist_768_v003 SHORT=expE10_spec SIZE=448 \
  BASE=eval_expE03f_joint_frame448_n2000 bash "$D/run_proc_kf.sh" FR

echo "########## $(date '+%F %T') QUEUE DONE ##########"
