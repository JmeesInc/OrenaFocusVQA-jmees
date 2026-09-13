# PROCEDURE track — retrieval-indexed frames + two-pass confidence selection

Final submission: **`orena-focus-procedure-p00c` build `p06scan`** (`container/`, image ≈37 GB,
tarball 25 GB, `sha256:e521051527768643b81ee96c0fb770186b39aacf29c75b120861a36fee14db50`).

The directory name `v016_procedure_p00c` upstream is a leftover from an earlier configuration — the
contents are unrelated to it. Identify a build by its `BUILD_ID` and adapter md5, not by name.

A PROCEDURE question may concern any moment of a whole operation, so the problem is **which 64
frames to look at**, not how to look at them. Two indices built offline decide that, and a second
VLM pass with a different input modality arbitrates the answer.

## Frame selection

The clip is scanned coarsely, then the budget is concentrated using two indices plus the question
text:

- **Surgical phase index** — ConvNeXtV2-tiny (SurgeNet SSL init) per frame, then a TeCNO-style MS-TCN
  over the sequence. Two models: Cholec80 for cholecystectomy, HeiCo for colorectal, each with an
  `other` class that doubles as the procedure router and the unknown-procedure fallback.
  Cholec80 official 40-video test: stage-1 acc 0.7813 → **+MS-TCN acc 0.9187 / macro-F1 0.8620 /
  edit 79.4**. HeiCo (anonymised intervals excluded): 0.5913 → **0.7329**.
  The temporal smoothing is what makes it usable as an index — edit score goes 11.2 → 98.2. Weights
  for strides 5/10/20 all ship, so the nearest one can be picked when the frame spacing changes.
- **Foreign-object index** — Mask2Former (`expF40`, 7 classes, segm AP 0.4043) over the coarse scan.
- **Question-text rules** (`rules_proc.py`) — `between T1 and T2` clips the interval; questions
  naming an `hh:mm:ss` anchor restrict to the phase holding that timestamp; `first/last visible`
  concentrates the budget at the edges of the **smoothed** mask.

The phase index is used as `phase × the anchor time in the question`, not as a prior on its own: a
question-independent phase prior moves reach from 0.2030 to 0.2356 and is not actionable, whereas
conditioning on the timestamp named in the question (19.6 % of `time` questions carry one) takes it
to 0.3805 with predicted phases — about the same as with ground-truth phases, so the index quality is
already sufficient. Dropping the phase models costs −0.0081.

Frame count is fixed at **64**. The earlier ladder that climbed to 128 for temporal questions never
actually ran (`ClipContext.times()` ignored the count), and when measured properly, 128-frame video
loses to 64-frame images on TEMPORAL (0.3173 vs 0.4148, p=0.0022) and combo2's advantage over uniform
sampling collapses from +5.3 pt to +0.3 pt. Training used 64 frames, so 64 is the matched condition.

## Two-pass confidence selection

```
pass 1  frame adapter (expP05)  — 64 images @ 448, always
pass 2  video adapter (expP06)  — 128 video frames @ 448, budget permitting
pick    the higher-confidence answer
```

Both adapters sit on the same base model and are switched with `set_adapter`. The `ClipContext` and
`ClipReader` from pass 1 are **reused** — rebuilding them would run the detector, which dominates the
budget, twice per question. Pass 2 is skipped whenever the remaining budget is below what pass 1
consumed, and always after a rush or an OOM retry, so the anytime property holds.

**Video input loses to image input on its own** (fold expP03 0.5605 < expP02 0.5934; per question,
frame wins 235 / video wins 119, p=7.0e-10 — the same direction as SEGMENT's expS00, −0.0612).
It is here as one side of a selector, not because it is strong.

## Training

| | frame adapter `expP05` | video adapter `expP06` |
|---|---|---|
| init | warm start from `expE03f` | warm start from `expE03f` |
| LoRA | r=16, α=32 | r=16, α=32 |
| data | FOCUS PROCEDURE only — train 8,000 + val 2,000 = 10,000, `train_part: all` | same |
| input | 64 images @ 448, grid 5.0 s, combo2 c9w + anchor | 128 video frames @ 448 |
| external / overlay | **none / none** | none / none |
| lr | 3e-5, effective batch 16, 2 epochs (1,250 steps) | 3e-5, 1,250 steps, 6 h 47 m on 4×RTX 5090 |
| adapter md5 | `43999b82…` | `f7faf42b…` |

The single change that earned expP05 its place: **the training-time frame selection was made
identical to inference** (combo2 c9w + anchor + 64 frames). Up to v010, training sampled 16 frames
uniformly while inference used the index — a mismatch. With it aligned, the fold version expP02
beats the control (0.5386) at **0.5934**: +0.0327 overall (207 improved / 143 worsened, p=0.0007),
driven by object_recognition **+0.0750** (560 questions, p=0.0001). expP05 changes only `train_part`.

Because `train_part: all` puts fold0 validation into training, **expP05 and expP06 cannot be
cross-validated**; every number above is from the fold versions.

The two-pass selector itself is **not shown to work**: on the fold version it moves SCORE 0.5907 →
0.5982, but McNemar gives p=0.547 (53 improved / 46 worsened), 72 % of the gain is one question in
EVENT_UNDERSTANDING (n=37), and restricting to `fo_class` — where the selector actually changes
answers, +9 of 89 — gives SCORE −0.0005. It ships as a positive-expectation judgement call.

### Reproducing

Lay the tree out as in [../FRAME/README.md](../FRAME/README.md#reproducing), with
`PROCEDURE/training/*` at `<root>/workspace/`:

```bash
python workspace/expM03_round1_external/train_lora_ext.py \
  --config workspace/expP00_proc_retrieval_train/config_P05_warm_proc64f_alldata.yaml
python workspace/expM03_round1_external/train_lora_ext.py \
  --config workspace/expP00_proc_retrieval_train/config_P06_warm_video128_alldata.yaml
```

`config_P02_*` and `config_P03_*` are the fold versions of the same two runs — the measurable ones,
and the source of every CV number quoted above.

Indices (`common/`): phase models with `expI00_phase_clf/train_phase.py` then `train_tcn.py`
(`config_cholec.yaml` / `config_heico.yaml` / `config_tcn.yaml`); detector with
`expF00_fo_instseg/train.py --config config_m2f_0828_noGallstone.yaml` (`expF40`, shipped here) and
`config_m2f_0828_clip_strongaug.yaml` (`expF39`, used by FRAME).

## Container

```bash
cd container
bash stage_resources.sh   # adapters, Mask2Former, TensorRT engine, both phase models
bash do_build.sh
bash do_test_run.sh
bash do_save.sh
```

`stage_resources.sh` also copies `phase_lib/` — `model.py`, `evaluate.py`, `dataset.py` **and
`train_tcn.py`**, because `evaluate.load_tcn` imports MSTCN from it. Omitting it does not raise: the
container silently falls back to running without the phase index.

`m2f_sm89.trt` ships but is **not used** — `uv pip install tensorrt-cu12` in the Dockerfile is
guarded by `|| echo` and fails, so the index runs on PyTorch. The measurement below is still valid
because it was taken with the same image, but do not reason from "sm_89 matches L40S so the TRT path
matches".

Regression test on an RTX 4090 (sm_89, same architecture as the L40S) with `--cpus 4`, 20 questions:

```
n=20 | empty 0 | answer.json conforms
latency mean 24.8s  median 26.2s  max 29.3s   over 30s: 0
batch 526s of 720s (73%)
two-pass: ran on 12, video answer taken on 5, skipped for budget on 8
VLM cost: frame 5.6-8.6s, video 11.2-13.6s  -> the second pass is not what makes this slow
```

### Three latency bugs this test exposed

Long clips (over 9,000 s) exceeded 30 s even on the frame pass alone. All three fixes only change
behaviour for long clips.

1. **The budget was handed out as a batch average.** `per_q = remaining / questions_left` carried
   forward time saved on earlier questions and gave a single question 39–44 s. Scoring is
   **per question** — over 30 s is wrong even with an answer — so spending 44 s on one question was a
   decision to score it zero. Fixed to `min(remaining / left, PER_QUESTION_BUDGET_S)`.
2. **The index budget only covered the detector.** The coarse decode ahead of it (540–712 frames,
   9–15.5 s) sat outside, so a 15 s allowance really cost 23–28 s. The measured decode time is now
   subtracted before the allowance is passed on (floor 3.0 s).
3. **The decode count has to be chosen before reading.** At a measured 43 frames/s, anything above
   `budget_s × 0.6 × 43` is halved (`want[::2]`), coarsening the scan 15 s → 30 s → 60 s.

| version | mean | max | over 30 s | batch |
|---|---|---|---|---|
| original (pooled budget) | — | 38.6 s | 3/8 | 89 % |
| + (1) per-question cap | 29.8 s | 37.6 s | 3/8 | 75 % |
| + (2) budget includes decode | 28.2 s | 33.9 s | 2/8 | 71 % |
| **+ (3) coarse scan derived from budget** | **25.3 s** | **28.6 s** | **0/8** | **65 %** |

Three long questions changed their answers. Whether for the better is unmeasured — but every one of
them had been over 30 s, i.e. already scored zero.

The test clips were re-encoded to the production specification (1024×576, 5 fps, I-frames at exactly
5.0 s with `-sc_threshold 0`, so seeking is at its most expensive). Against the original clips the
difference was about 15 % — not the dominant factor.

Weights are published after the challenge concludes; `container/weights_manifest.json` records the
md5 of each so a downloaded weight can be tied back to the training run.
