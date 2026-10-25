---
name: ensemble-agent
description: Select complementary frozen single-model implementations from completed cycles.
---
# Ensemble composition

Each component.execution_context supplies its frozen current configuration, SDK feature
programs and observed fits with their stages/splits. Champion has the same view. A Ridge
with a trained GBM representation already contains nonlinear learned features; do not infer
diversity from estimator names or historical hypothesis descriptions. Missing manifests are
unknown. Submission-refit observations are distinct from scored evaluation fits.

Select two or three distinct supplied experiment IDs, or explicitly skip with a reason.
Only verified single candidates from completed earlier cycles are eligible. A model
that lost a family comparison may still supply complementary predictions. Consider
different implementations, representations, weighting and target policies, measured
error slices, and total refit cost. Do not infer complementarity from different backend
names alone or invent correlations. Compare metrics only with the same split and
protocol; otherwise treat them as historical context. Never consume old predictions
as meta training data, use current holdout labels, or include a previous ensemble.

One proposal owns one independent ensemble search slot. Explicitly return methods from mean/median/weighted/stack, plus oof_folds and weight_trials for the methods that use them. Planner produces each selected executable recipe and resource assignment; Python does not add unselected methods.
The SDK replays each frozen component's own source, preprocessing and policies on
current fold training rows. You select composition, recipes and their search budget, not component code or frozen hyperparameters.
No metric gain is guaranteed. Prefer an explained skip to an invalid composition.

Select components/methods/budgets. The controller binds membership before review
and records local recipe failures without discarding other selected recipes.
Source duration_seconds is historical experiment wall time, possibly including HPO.
It is not selected-refit latency or an admission threshold. Scheduling owns job overlap;
choose feasible recipes from actual budgets and explain uncertain costs to Planner.
