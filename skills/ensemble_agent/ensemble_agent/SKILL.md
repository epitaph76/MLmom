---
name: ensemble-agent
description: Select complementary frozen single-model implementations from completed cycles.
---
# Ensemble composition

Select two or three distinct supplied experiment IDs, or explicitly skip with a reason.
Only verified single candidates from completed earlier cycles are eligible. A model
that lost a family comparison may still supply complementary predictions. Consider
different implementations, representations, weighting and target policies, measured
error slices, and total refit cost. Do not infer complementarity from different backend
names alone or invent correlations. Compare metrics only with the same split and
protocol; otherwise treat them as historical context. Never consume old predictions
as meta training data, use current holdout labels, or include a previous ensemble.

One proposal owns one independent ensemble search slot. Planner produces the resource
assignment; workers compare simple combinations before learned weights and OOF stacking.
The SDK replays each frozen component's own source, preprocessing and policies on
current fold training rows. The agent selects composition, not code or hyperparameters.
No metric gain is guaranteed. Prefer an explained skip to an invalid composition.
