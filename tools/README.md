# Weight publication

Trained weights are not stored in git — a LoRA adapter is ~215 MB and each Mask2Former detector
~823 MB, over GitHub's 100 MB per-file limit, and GitHub LFS free quota (1 GB) does not cover the
~3 GB a single track needs. They are published to a Hugging Face model repo instead.

## Publishing

```bash
cd FRAME/container && bash stage_resources.sh && cd ../..     # fills container/resources/
python tools/publish_weights_hf.py --track FRAME --repo <org>/OrenaFocusVQA-jmees-weights --dry-run
python tools/publish_weights_hf.py --track FRAME --repo <org>/OrenaFocusVQA-jmees-weights
```

`--dry-run` writes `FRAME/container/weights_manifest.json` (member name → Hub path + md5) without
uploading, so the manifest can be reviewed and committed first. Members that share an adapter are
uploaded once and recorded as an alias.

## Consuming

```bash
huggingface-cli download <org>/OrenaFocusVQA-jmees-weights --include 'frame/*' \
  --local-dir FRAME/container/resources
```

The directory names under `resources/` are what the container looks for; keep them. The md5 values
in the manifest are the same ones `stage_resources.sh` asserts, so a download can be verified
against the training run that produced it.

`stage_resources.sh` is the path for reproducing from our own training outputs; downloading from the
Hub is the path for reproducing from the published weights. Either leaves `resources/` in the state
`do_build.sh` expects.
