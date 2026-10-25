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
protocol; otherwise treat them as historical context. Old inner OOF may support historical
diagnostics through compare_oof_components. Never consume those old predictions as
current meta training data, use current holdout labels, or include a previous ensemble.

## Diversity Evidence

Shortlist components by their frozen representations, policies, error slices and available
measurements. read_component_history pages the pool; diversity_availability distinguishes
an OOF reference from inspected evidence. Request compare_oof_components for shortlisted
pairs or triples when artifacts are registered. Missing OOF is not a reason to invent a
measurement or automatically skip an otherwise useful experiment.

Return agent_notes.diversity_evidence with the saved measurement reference, interpretation,
unknowns, and decision_status. Explain each item:

- Scope and support: source experiments, dataset, same training/validation protocol,
  aligned source rows/folds, target scale, class order, primary metric/options and weights.
  A shared row intersection from different folds/protocols is insufficient. The tool reads
  saved historical inner-validation artifacts only; it exposes aggregates, not labels.
- Pairwise errors: weighted Pearson correlation of signed residuals per regression output
  or misclassification indicators; row-loss correlation; error intersection/union and
  weighted Jaccard. Regression overlap uses positive row MAE at or above each component's
  own unweighted 80th percentile, including ties; classification uses wrong-label events
  (any wrong label for multilabel). Report this definition and support. A constant error
  vector has undefined correlation, not zero. Brier loss and classification threshold/.5
  or argmax are diagnostics, not replacements for the primary metric.
- Incremental benefit: the tool evaluates a fixed equal-mean blend and singles on the
  exact same folds with the SDK primary metric/options/weights, on the original target
  scale. Positive improvement_over_best means better in the metric's declared direction.
  mean_fold_score and pooled_oof_score are distinct.
  For a triple, leave_one_component_out measures adding each component relative to the
  remaining pair, with the same positive-is-better convention.
  Saved label predictions can establish error overlap but cannot reconstruct the runtime
  probability blend; mark its gain unknown.
  This cheap diagnostic fits no weights or meta-model. It does not verify weighted/stack
  recipes or provide independent confirmation after configuration/composition selection.
- Full-refit cost: distinguish measured component fit durations, an estimated sum at their
  recorded row counts/CPU/RAM, unknown full ensemble latency, and additional OOF/meta fits.
  Mean/median need one full fit per component; weighted/stack also need oof_folds fits per
  component and weight/meta fitting. Historical HPO duration is not selected-refit cost.

Use decision_status=unverified_hypothesis when the proposed complementarity or recipe gain
has not been measured. measured_diversity establishes an error relationship only;
historical_gain_observed establishes only the reported fixed-mean historical diagnostic.
historical_gain_not_observed is useful negative evidence. None is a guarantee of current
ensemble improvement. State which component pairs remain unknown; an incomplete triple
must not inherit a pair's evidence. Frozen configurations and composition may already have
been chosen using these folds. Current recipe evaluation must refit fresh components within
its own folds; learned weights/meta-models need their own training-only OOF and held-out
evaluation. Correlation, different libraries and disjoint error slices alone prove no gain.

The controller persists measured Diversity Evidence, attaches it to selected recipes for
Planner and review, and marks missing/incompatible artifacts explicitly. Treat numerical
results as supplied measurements; do not replace them with an unsupported prose claim.

One proposal owns one independent ensemble search slot. Explicitly return methods from mean/median/weighted/stack, plus oof_folds and weight_trials for the methods that use them. Planner produces each selected executable recipe and resource assignment; Python does not add unselected methods.
The SDK replays each frozen component's own source, preprocessing and policies on
current fold training rows. You select composition, recipes and their search budget, not component code or frozen hyperparameters.
No metric gain is guaranteed. Prefer an explained skip to an invalid composition.

Select components/methods/budgets. The controller binds membership before review
and records local recipe failures without discarding other selected recipes.
Source duration_seconds is historical experiment wall time, possibly including HPO.
It is not selected-refit latency or an admission threshold. Scheduling owns job overlap;
choose feasible recipes from actual budgets and explain uncertain costs to Planner.
