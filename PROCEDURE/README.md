# PROCEDURE track

**The final submission for this track is not fixed yet.** This directory is a placeholder; the
container source, training configs and weight pointers will be filled in once the submission is
decided, following the same layout as [../FRAME](../FRAME/).

## Where the track stands

The candidate is a two-pass, confidence-selected ensemble over a 64-frame image input, with frame
selection driven by a surgical-phase index (ConvNeXtV2-tiny + MS-TCN, trained on Cholec80 and HeiCo)
and a foreign-object index. Cross-validated SCORE for the measurable ancestor of that recipe was
0.5934 (fold v003 fold0); the submitted adapter is trained on train+val and is therefore not
measurable.

The shared training infrastructure is already published in [../common](../common/).
