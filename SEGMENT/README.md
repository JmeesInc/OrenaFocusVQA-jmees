# SEGMENT track

**The final submission for this track is not fixed yet.** This directory is a placeholder; the
container source, training configs and weight pointers will be filled in once the submission is
decided, following the same layout as [../FRAME](../FRAME/).

## Where the track stands

Leaderboard results during the validation phase (SCORE):

| submission | SCORE |
|---|---:|
| v015 frame-router | **0.5989** |
| v012 | 0.5891 |
| v018 (expR00 + s3 overlay + revised router) | 0.5876 |
| v005 hybrid | 0.5692 |
| v013 (s1 overlay + question-text rules) | 0.5338 |

The shared ingredients are already published here: the joint FRAME+SEGMENT training data recipe and
the dataset/prompt/post-processing modules live in [../common](../common/), and the temporal
overlay renderer used by the SEGMENT containers is `expN00_seg_overlay3` (style `s3`).
