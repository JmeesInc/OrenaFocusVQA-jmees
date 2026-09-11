# FRAME track — bucket routing + 3-model vote per route

Final submission: **`orena-focus-frame-router` build `r2`** (`container/`, image ≈21 GB).

The FRAME score is the unweighted mean of 10 buckets (5 capability groups × in/out-of-distribution).
In practice FRAME validation collapses to four non-empty buckets, and our leaderboard submissions
decomposed as follows (question counts, N in parentheses):

| submission | obj_id (747) | agg_id (553) | obj_ood (512) | agg_ood (188) | SCORE |
|---|---:|---:|---:|---:|---:|
| **v011** (route A alone) | **538** | 324 | **262** | **129** | **0.6260** |
| v014 | 515 | 322 | 213 | 113 | 0.5722 |
| v015 (route-B model alone) | 515 | **338** | 247 | 118 | 0.6027 |
| leaderboard 1st | 537 | 342 | 274 | 125 | 0.6343 |

Two observations drive the final design:

1. **`obj_id` is decided by the overlay variant alone.** It stays at 515 for every `r4`
   (class colour + instance id + confidence) run — regardless of LoRA rank, extra external data, or
   adding the validation split to training — and only reaches 538 with `r1` (instance numbers only).
2. **The only bucket where v015 beats v011 is `agg_id`** (+14 questions).

So we keep v011 untouched where it is strong and route only the bucket where the other recipe wins.
Recombining the measured buckets gives `(0.7202 + 0.6112 + 0.5117 + 0.6862)/4 = 0.6323`.

## Routing

The route is decided **before any model runs**, from the request attributes and the question text only:

```
ID/OOD      : procedure_type ∈ {Proctocolectomy, Rectal Resection,
                                Sigmoid Resection, Laparoscopic Cholecystectomy}
capability  : detect_format → number/binary = AGGREGATION, otherwise OBJECT_RECOGNITION

route A  (all OOD + ID×object)  → the three v011 members, unchanged
route B  (ID×aggregation)       → three all-data members
```

`detect_format` agrees with the annotated capability group with purity 1.000 / 1.000 / 0.980 on FRAME
validation. Unknown or empty `procedure_type` falls to **route A**: mistaking OOD for ID would cost
`agg_ood`, while the opposite error merely reproduces v011.

Implementation: `container/inference.py` (`ROUTES`, `route_of`), voting in `container/ensemble.py`.

## Members

| route | member | width | overlay | trained by | adapter md5 |
|---|---|---|---|---|---|
| A | `m1_m03b_overlay` | 1024 | r1 | expM03B | `c0f2795a…` |
| A | `m2_m03b_768` | 768 | r1 | expM03B (same adapter, different resolution) | `c0f2795a…` |
| A | `m3_v06_r32` | 768 | none | expV06_frame_r32 | — |
| B | `b1_q02b_r1` | 768 | r1 | [expQ02B](training/expQ02_v011_alldata/) | `7a5b2ba9…` |
| B | `b2_q00b_r4` | 768 | **r4** | [expQ00C](training/expQ00_frame_alldata/) | `c069a8f1…` |
| B | `b3_q03b_r1sa` | 768 | r1 | [expQ03B](training/expQ03_alldata_sa/) | `297a1fd9…` |

Route B is three **independently trained** adapters — voting between two copies of one adapter
measured *worse* than that adapter alone (0.6528 vs 0.6581).

## Training recipe (route B)

All three route-B members share the base recipe and differ only in the factors marked below.

| | value |
|---|---|
| base | `Qwen/Qwen3.5-9B`, 4-bit NF4 weights / bf16 compute |
| LoRA | r=16, α=32 (r=64 was measured worse on the leaderboard: −0.0538) |
| resolution | 768 px, identical at train and inference |
| FOCUS data | **train + val concatenated** (`train_part: all`), 20,000 questions |
| pseudo VQA | 15,906 |
| external VQA | SurgMLLMBench 15,525 + MultiBypass140 8,000 (+ SurgAtlas 16,645 for b2/b3), `loss_weight 0.5`, procedure sentence dropped 50 % of the time |
| schedule | **Stage A** mixed corpus, lr 1e-4 cosine, 1 epoch → **Stage B** FOCUS + pseudo only, lr 5e-5 |
| batch | effective 16 (bs 1 × grad-accum 16, or ×4 under 4-GPU DDP) |
| eval | none — `train_part: all` leaves no held-out set, so the final step is taken |

| member | overlay | SurgAtlas |
|---|---|---|
| b1 (expQ02) | r1 | no |
| b2 (expQ00) | r4 | yes |
| b3 (expQ03) | r1 | yes |

The two-stage schedule matters: in a single-factor comparison where only the number of stages
differed, baking FOCUS in after the mixed stage was worth **+0.0157** (expM04A 0.6344 →
expM04B 0.6501, pooled CV over 3,928 questions, latency ignored). Build `r2` exists because b2 was
originally one-stage; `r2` replaces it with the two-stage `expQ00C` adapter so that all three
route-B members share the structure.

### Reproducing

```bash
# layout expected by the scripts (they resolve each other by relative path)
<root>/workspace/expE01_segproc_baseline/   <- common/expE01_segproc_baseline
<root>/workspace/expM00_frame_overlay/      <- common/expM00_frame_overlay
<root>/workspace/expM03_round1_external/    <- common/expM03_round1_external
<root>/workspace/expK00_pseudo_vqa/         <- common/expK00_pseudo_vqa
<root>/workspace/fold/                      <- common/fold
<root>/workspace/expQ0{0,2,3}_*/            <- FRAME/training/*
<root>/reference/                           <- the official orena-focus toolkit
<root>/data/focus/                          <- FOCUS_ROOT_DIR

# single GPU
python workspace/expM03_round1_external/train_lora_ext.py \
  --config workspace/expQ02_v011_alldata/config_Q02A_alldata_stageA.yaml
python workspace/expM03_round1_external/train_lora_ext.py \
  --config workspace/expQ02_v011_alldata/config_Q02B_alldata_stageB.yaml

# 4x4090 DDP (see training/expQ00_frame_alldata/train_ddp_q00.sh)
#   torchrun --nproc_per_node 4 --no-python \
#     workspace/expE01_segproc_baseline/ddp_wrap.sh python .../train_lora_ext.py --config ...
```

`ddp_wrap.sh` is required under DDP: without it every rank keeps a CUDA context on rank 0's GPU
(386 MiB each), which is enough to OOM a 24 GB card. `--no-python` is needed because torchrun would
otherwise try to run the wrapper as a Python file. Stage B resumes from Stage A via
`init_from_adapter` in the config — point it at the actual `fold0*/adapter` directory that Stage A
wrote (the output directory is numbered `_001`, `_002`, … when a name repeats, so a smoke run can
leave a decoy).

## Container

```bash
cd container
bash stage_resources.sh   # copy the 6 adapters + 2 detectors into resources/ and verify md5
bash do_build.sh          # docker build
bash do_test_run.sh       # local regression test, --network none
bash do_save.sh           # docker save | gzip -> submission tarball
```

Inference is **pass-based and anytime**: round 1 answers every question with the first member of its
route and writes `answer.json`; each later round adds one more member's vote. If the wall clock runs
out, whatever has been written so far stands. `answer.json` is also written — empty — before the
heavy imports, because a container that produces no output is a *failed* submission whereas empty
answers merely score zero.

Regression test on an RTX 4090, 20 questions, `--network none`:

```
full case:  route A 12 / route B 8 | passes 6 | 3-vote on 20/20 | empty 0 | 159.8s of 220s (73%)
            [m1] 2.70s/q  [b1] 2.28  [m2] 2.49  [b2] 2.28  [m3] 2.17  [b3] 2.28
```

Measure latency on an unshared GPU and on the generation the submission will run on: the same image
used 73 % of the budget alone and 84 % with another container resident on the card, and on an A4000
round 1 alone consumed 166 s of 220 s so no vote ever completed.

## Honest scope of the evidence

Leaderboard numbers exist for **the route-A trio as a set** (v011, 0.6260) and for **b2's one-stage
ancestor alone** (v015, 0.6027). b1 and b3 have neither leaderboard nor cross-validation results —
`train_part: all` consumes the validation split — and the route-B three-way vote has never been
measured as a combination. What the leaderboard supports is the *routing rationale* (the bucket
decomposition above), not each member. Since votes are decided by majority, route B's answers are
dominated by b1 and b3, i.e. by the two members without measurements, while the member with the
measured `agg_id` advantage is in the minority. SurgAtlas, present in b2 and b3, measured neutral to
slightly negative in distribution (expQ01B 0.6443 vs expM03B 0.6504) and has no leaderboard evidence
either way.
