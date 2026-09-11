#!/bin/bash
# expK00: survis-anno instseg アノテーション → FRAME 擬似 VQA 生成 + RARP cut-paste + 検証 + カード
set -eu
cd "$(dirname "$0")"
PY=../../.venv/bin/python
$PY generate_frame_pseudo_qa.py       # → out/pseudo_frame_v1.parquet
$PY validate_against_official.py      # 網羅性の検算（公式 QA と同一フレーム照合）
$PY make_cards.py                     # → out/qa_cards.png
$PY harvest_rarp_needles.py           # → out/rarp_needle_crops/ (827 crops)
$PY generate_cutpaste_qa.py           # → out/pseudo_frame_cutpaste_v1.parquet
$PY make_cutpaste_cards.py            # → out/cutpaste_cards.png
$PY pack_pseudo_frames.py             # → out/packed/ + *_packed.parquet（貸しGPU転送用 768px）
