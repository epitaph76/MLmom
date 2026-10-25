---
name: lightgbm
description: Fit CPU LightGBM 4.6.0 with native categories and internal-validation early stopping.
---
# LightGBM 4.6.0

Implement train-derived categorical levels explicitly; unseen levels become missing. Numeric NaN is supported. For numeric-coded categorical columns, explicitly cast the planned columns. OHE is a separate implementation choice. See REFERENCE.md and persist column order/vocabularies.

Single-output regression/classification use context.validation_X/y and exact primary-metric early stopping (50 rounds) only when the plan stopping_policy permits it. Export the complete configuration; for permitted early stopping freeze best_iteration_ as n_estimators; validation-free refit uses that count. Multioutput/ranking require explicit per-target/group-aware custom stopping.

Start learning_rate .03-.08, num_leaves 31-63, min_child_samples 20-50, colsample_bytree .8-1, reg_lambda 1. Row subsampling needs subsample_freq > 0. Budget tree ceilings. Optuna runs outside candidate.fit.

Ranking sorts X/y/weights together, uses contiguous queries and group sizes summing to n_rows, and retains prediction row order. Multioutput/multilabel wrappers are serial. CPU settings come from context, never model_params overrides.

export_model_params returns ALL model parameters and output/target options, not only n_estimators. Replace the count with measured best_iteration_ only for permitted early stopping. Return the selected parameters when useful for refit. No fitted-estimator audit or export/recreate check is performed.

A backend update may produce no valid split, so current_iteration() and retained tree counts
can be smaller than the requested boosting rounds after a successful fit. Keep the declared
fixed count unchanged; runtime metrics and saved artifacts reflect the executed fit; no Booster.update interception is performed. Do not raise diagnostic exceptions or edit fitted counters to force exact tree counts.
