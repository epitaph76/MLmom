---
name: lightgbm
description: Fit CPU tabular boosters and query rankers with LightGBM 4.6.0, optionally using native categories and early stopping.
---
# LightGBM 4.6.0

Prefer for mixed tabular data. LibraryCandidate uses LGBMClassifier/Regressor/Ranker with dense imputed OHE; no eval_set/early stopping. Native categories or early stopping require custom code.

Start learning_rate .03-.08, num_leaves 31-63, min_child_samples 20-50, colsample_bytree .8-1, reg_lambda 1. Row subsampling needs subsample_freq > 0. Keep stock n_estimators modest. Custom fit may use a high ceiling plus lightgbm.early_stopping(50-100, first_metric_only=True) on transformed context.validation_X/y. Without validation, disable early stopping; export best_iteration_ as fixed n_estimators for refit.

Native categories require stored train CategoricalDtype levels applied at inference; unknowns become missing. Numeric NaN is supported.

Ranking requires contiguous queries and group sizes summing to n_rows, plus eval_group for validation. Sort X/y/weights together; retain prediction row order. The adapter serially wraps multioutput/multilabel models; per-target early stopping needs per-target eval labels. Set n_jobs from context in code, never model_params overrides.
