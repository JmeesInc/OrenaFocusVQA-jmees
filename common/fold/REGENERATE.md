# Regenerating the question-level splits

Two levels of split are used:

| file | level | in git |
|---|---|---|
| `<version>/folds.csv` | **video → fold** (GroupKFold over videos). Our own design decision. | yes |
| `qa_<version>/qa_split.csv` | **question → fold**, materialised from `folds.csv` + the official QA parquets | no — regenerate |

`qa_split.csv` carries per-question annotation columns (`primary`, `answer_format`, `ood`, …) taken
from the official releases, so it is not redistributed here. It is a deterministic function of the
video-level `folds.csv` (which *is* in git) and the official data, and regenerating it reproduces the
file used for training byte for byte.

## Command

Point `FOCUS_ROOT_DIR` at your FOCUS data root, install the official toolkit
(`pip install -e reference/[dev]`), then:

```bash
# the split every final model was trained against
python workspace/fold/generate_qa_split.py \
  --version v004 --fold-version v003 --tracks FRAME SEGMENT PROCEDURE
```

Note the version numbers do **not** line up: `qa_v004` is built on video folds `v003`, and `qa_v005`
on video folds `v001`. The configs reference the `qa_` name (`data.qa_version: v004`).

Other splits, for completeness:

```bash
python workspace/fold/generate_qa_split.py --version v001 --fold-version v001 --tracks FRAME
python workspace/fold/generate_qa_split.py --version v003 --fold-version v003 --tracks FRAME
python workspace/fold/generate_qa_split.py --version v005 --fold-version v001 --tracks FRAME SEGMENT PROCEDURE
```

## Checks the generator prints

`fold を跨ぐ動画: 0 件` must be 0 — a non-zero count means the same video appears in both train and
validation, which leaks. Question ids alone are **not** unique across tracks (49,905 distinct ids for
50,000 rows); the `uid` column is the key.
