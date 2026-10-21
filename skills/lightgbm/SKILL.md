---
name: lightgbm
description: Fit CPU LightGBM 4.6.0 with native categories and internal-validation early stopping.
---
# LightGBM 4.6.0

LibraryCandidate uses native train-derived category levels; unseen categories become missing. Numeric NaN is supported. preprocessing=['onehot'] selects the legacy dense fallback. Persist column order and category vocabularies.

Single-output regression/classification use context.validation_X/y and exact primary-metric early stopping (50 rounds). Export best_iteration_ as fixed n_estimators; validation-free refit uses that count. Multioutput/ranking require explicit per-target/group-aware custom stopping.

Start learning_rate .03-.08, num_leaves 31-63, min_child_samples 20-50, colsample_bytree .8-1, reg_lambda 1. Row subsampling needs subsample_freq > 0. Budget tree ceilings. Optuna runs outside candidate.fit.

Ranking sorts X/y/weights together, uses contiguous queries and group sizes summing to n_rows, and retains prediction row order. Multioutput/multilabel wrappers are serial. CPU settings come from context, never model_params overrides.
