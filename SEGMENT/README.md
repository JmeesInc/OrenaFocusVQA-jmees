# SEGMENT track — group-routed input + two-model confidence selection

Final submission: **`orena-focus-segment-conf2` build `r2`** (`container/`, image ≈19 GB,
`sha256:2ab2227bd85c60c67386f631a2da7451cf8899062df31ef9de21ec6ff2162f9e`).

Two independent mechanisms stack here: **what the model is shown** is chosen per question from the
question text, and **which of two models is believed** is chosen per question from the models' own
confidence.

Leaderboard history during the validation phase:

| submission | SCORE |
|---|---:|
| v015 frame-router (`expN04`, the 1st model here) | **0.5989** |
| v012 | 0.5891 |
| v018 (`expR00` + s3 overlay + revised router) | 0.5876 |
| v005 hybrid | 0.5692 |
| v013 (s1 overlay + question-text rules) | 0.5338 |

## Input routing — frames vs. resolution

The same vision-token budget has a different optimum for different capability groups, measured on
fold0 validation (N=1958, judge included, duplicate video 0027 excluded):

| group | N | 64f @ 560 | 32f @ 768 |
|---|---:|---:|---:|
| object_recognition | 850 | **0.8247** | 0.8047 |
| temporal_grounding | 795 | **0.5962** | 0.5421 |
| aggregation | 186 | 0.5000 | **0.5376** |
| event_understanding | 74 | 0.8243 | **0.8649** |
| complex_reasoning | 53 | 0.8113 | **0.8491** |
| | | 0.7113 | 0.7197 |

Routing between them reaches **0.7345** — the same as oracle routing on the held-out table, because
the capability group is recoverable from the question template (agreement 0.978, 3.1 % of templates
unseen). The group is not in `Request`, so `router.py` infers it from the question text against
`group_templates.json`.

A second, narrower rule (`FOCUS_SEGRULES`) concentrates the frame budget inside the window named by
questions of the form `between T1 and T2` or `at T … When is it retrieved`; it fires on 367 of 4,006
questions.

## Model selection — length-normalised log-probability

Every question is answered by **two** adapters and the more confident answer is kept:

```
1st model  : expN04                       (all questions)
2nd model  : procedure_type ∈ ID set → expR00C  (FOCUS fine-tuned, all-data)
             otherwise             → expR00   (external-mixed, all-data)
pick        : higher logp_mean (mean log-probability over generated tokens)
```

`logp_mean` is a **comparison, not a threshold**, so nothing needs calibrating; in the expR00 A/B it
picked the better answer 65.9 % of the time. The clip is decoded once and both passes share the
frames. If the remaining budget falls below 90 % of pace the container drops to the 2nd model alone,
and below 50 % to a 16f@448 rush configuration — so the two-pass design degrades instead of failing.

No detector and no overlay run at inference. `expR00` was trained with s0/s3 mixed half and half
(`two_way: true`), so it reads un-overlaid frames natively, and dropping the detector removes the
7.87 s/question that v018 spent on it.

## Training

| | 1st model `expN04` | 2nd model `expR00` | 2nd model `expR00C` |
|---|---|---|---|
| init | Qwen3.5-9B (4-bit NF4 / bf16) | Qwen3.5-9B | **`expR00` adapter** |
| LoRA | r=16, α=32 | r=16, α=32 | r=16, α=32 |
| data | FOCUS FRAME 1f@448 + SEGMENT 32f@448 + 16f@560, `train_part: all` | + external VQA (SurgMLLMBench / MultiBypass140 / SurgAtlas), `loss_weight 0.5`, `procedure_dropout 0.5` | FOCUS only, `train_part: all` (40,000 = train 31,986 + val 8,014) |
| overlay | none | **s3** (bbox, class colour, confidence, no instance number), `two_way` s0:s3 = 1:1 | none |
| lr | 1e-4 cosine | 1e-4 cosine | **3e-5** (continuation, not scratch) |
| anchor | yes — frames at any `hh:mm:ss` named in the question are always included (17.1 % of SEGMENT) | yes | yes |
| adapter md5 | `71f7d92a…` | `c88fde1d…` | `09a229b1…` |

`expR00C` is the same idea as the FRAME route-B members: pre-train on the mixed corpus, then bake in
FOCUS alone. It ran 2,500 steps (1 epoch, effective batch 16) in 4 h 43 m on 4×RTX 5090.

The external lane caps FRAME at 8,000 examples. The external corpora are FRAME-heavy (40,170
combined), and letting that through skews the track ratio — adding FRAME pseudo-data alone once moved
the ratio from 1:1 to 1:1.76 and cost −0.0303. Capping keeps it at roughly 28,000 : 28,000.

All three are trained with `train_part: all`, i.e. fold0 validation is in the training set, so **none
of them is cross-validatable**; the CV numbers quoted above come from the measurable ancestors.

### Reproducing

Lay the tree out as described in [../FRAME/README.md](../FRAME/README.md#reproducing) — the same
trainer (`common/expM03_round1_external/train_lora_ext.py`) and the same fold files are used — with
`SEGMENT/training/*` placed at `<root>/workspace/`, then:

```bash
python workspace/expM03_round1_external/train_lora_ext.py \
  --config workspace/expN04_seg_framerouter/config_N04_framerouter.yaml
python workspace/expM03_round1_external/train_lora_ext.py \
  --config workspace/expR00_seg_s3/config_R00_seg_s3_r64_5090fa.yaml   # -> expR00 (r16 despite the filename)
python workspace/expM03_round1_external/train_lora_ext.py \
  --config workspace/expR00_seg_s3/config_R00C_focusft_alldata.yaml    # -> expR00C
```

The `_r64_` in the second filename is historical; the config it contains is r=16 / α=32 and its
`experiment.name` is `expR00_seg_s3_r16_alldata_5090`. `config_R00B_focusft_trainonly.yaml` is the
train-only counterpart of R00C, kept because it is the measurable version of the same step.

## Container

```bash
cd container
bash stage_resources.sh   # place the 3 adapters under resources/ and verify md5
bash do_build.sh
bash do_test_run.sh
bash do_save.sh
```

Regression test on an RTX 4090 with `--cpus 4`, 20 questions (12 ID / 8 OOD):

```
answered 20/20 | empty 0 | failed 0 | rushed 0 | unknown template 0
gate ID 12 / OOD 8 | agree 15, pick_first 9, pick_id 8, pick_ood 3
setup 134.8s | latency mean 12.77s  median 15.59s  max 17.76s
```

### The latency budget is a pool, not a per-question limit

Ten of those twenty questions exceed 15 s, which looks fatal until you compare against a container
that actually ran. v018's leaderboard record settles it:

```
v018: SCORE 0.5876 | forfeited 0 | unanswered 0 | mean_latency_per_question 14.2575 s
      batch 7248.7s of budget 7620s (95.1%),  budget = 120s + B x 15s  with B = 500
```

The judgement is made on the **batch pool**, so individual questions may exceed 15 s. Re-running v018
on the same 20 questions, same GPU and same `--cpus 4` gives the paired comparison:

| build | answered | mean | median | max | >15 s |
|---|---:|---:|---:|---:|---:|
| v018 (1 pass + detector + s3 overlay) — ran in production | 20 | 14.44 s | 15.13 s | 21.79 s | 10 |
| **v019 (2 passes, no detector)** | 20 | **12.77 s** | 15.59 s | **17.76 s** | 10 |
| ratio | | **0.88×** | | −4.0 s | |

Extrapolating: 0.88 × 14.26 = 12.55 s/question → 6,275 s of 7,620 s, **82 % of the pool** against
v018's 95.1 %. Decide latency by the ratio against a build with a production record, not by the
absolute local number — 14.44 s/question locally against 14.2575 s in production is why the ratio is
trustworthy here.

Weights are published after the challenge concludes; `container/weights_manifest.json` records the
md5 of each so a downloaded weight can be tied back to the training run.
