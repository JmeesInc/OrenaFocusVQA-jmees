# OrenaFocusVQA-jmees

Team **JMEES** solution for the **[Orena Focus Challenge](https://orena-focus-challenge.org/) (MICCAI 2026)** —
surgical video question answering on the HeiCo-FOCUS / LapChole-FOCUS datasets.

The challenge is submitted as **grand-challenge.org algorithm containers**, one per track.
This repository publishes the final solution: container source, training recipes and
pointers to the trained weights.

| Track | Input | Max latency | Status |
|---|---|---|---|
| [FRAME](FRAME/) | single frame | 5 s | **final** — bucket router + 3-model vote per route |
| [SEGMENT](SEGMENT/) | ≤5 min clip | 15 s | **final** — group-routed input + two-model confidence selection |
| [PROCEDURE](PROCEDURE/) | full video | 30 s | **final** — retrieval-indexed frames + two-pass confidence selection |

## Layout

```
FRAME/ SEGMENT/ PROCEDURE/
  container/   grand-challenge algorithm container (Dockerfile + inference)
  training/    experiment configs and launch scripts for the submitted weights
common/
  expE01_segproc_baseline/   shared dataset / prompt / post-processing modules
  expM00_frame_overlay/      instance-segmentation overlay renderer (variants r0-r4)
  expM03_round1_external/    QLoRA trainer with external-VQA mixing (train_lora_ext.py)
  expK00_pseudo_vqa/         pseudo-VQA generation
  expF00_fo_instseg/         Mask2Former foreign-object instance segmentation
  expI00_phase_clf/          surgical phase index (ConvNeXtV2 + MS-TCN)
  fold/                      cross-validation fold assignments
tools/                       weight publication helpers
```

`common/` mirrors the `workspace/<experiment>/` layout of our internal repository, because the
training scripts resolve each other by relative path. See each track's README for how the paths
map when reproducing.

## Method in one paragraph

All tracks fine-tune **Qwen3.5-9B** with **QLoRA** (4-bit NF4 weights, bf16 compute, LoRA r=16 / α=32)
on the FOCUS VQA pairs, mixed with pseudo-VQA and three external surgical VQA corpora
(SurgMLLMBench, MultiBypass140, SurgAtlas) at a reduced loss weight. A Mask2Former instance
segmentation model for foreign objects and surgical tools renders a **class-coloured overlay with
instance numbers and confidences** onto the frames the VLM sees. Final containers ensemble several
adapters by majority vote under an **anytime** schedule so the latency budget is never exceeded.

## Weights

LoRA adapters (~215 MB each) and detector checkpoints (~823 MB) exceed GitHub's file size limit and
are **not** stored in git. They are published at
**[negichi/OrenaFocusVQA-jmees-weights](https://huggingface.co/negichi/OrenaFocusVQA-jmees-weights)**,
one folder per track. Each track's `container/weights_manifest.json` records the md5 of every file,
so a downloaded weight can be tied back to the training run that produced it — the same md5 values
`container/stage_resources.sh` asserts. See [tools/README.md](tools/README.md).

## Data

We do not redistribute challenge data. What is in git is our own **video-level** fold assignment
(`common/fold/<version>/folds.csv`); the question-level split is regenerated from it and the official
releases ([heico-focus-vqa](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa),
[lapchole-focus-vqa](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa)) with one
command — see [common/fold/REGENERATE.md](common/fold/REGENERATE.md). The regenerated file is
byte-identical to the one the final models were trained on.

External training corpora are obtained from their original sources under their own licenses.
