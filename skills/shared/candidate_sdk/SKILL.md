---
name: candidate-sdk
description: Fold-local preprocessing, estimators and reproducible ModelSpec export.
---
# Candidate SDK

candidate.py exports create_candidate(config) and load_candidate(artifact_dir).
Objects implement fit(X,y,context), predict(X,prediction_spec), save(artifact_dir).
preprocessing.py exports create_preprocessor(config), returning fit_transform(X,y,context)
and transform(X). Keep rows/order and fresh train-fitted state for every fold/refit.
Runtime owns target/weight/prediction policies; do not apply them twice.

FitContext provides task_type, metric, direction, seed, cpus, memory_gb, time_limit_seconds,
remaining_seconds, timeouts_enabled, classes, groups, sample_weight, validation_X/y,
validation_groups/weight and work_dir. Use attributes. Threads use context.cpus.
Only check a deadline when timeouts_enabled. Workers receive no outer labels or LLM keys.

export_model_params is optional: return a finite JSON object for later refit. If absent,
the runner uses the executed trial parameters. fitted_components is not required.
There is no estimator audit or mandatory export/recreate. check_source checks syntax and
factories; run_smoke_check is optional. Real errors are returned with a traceback for repair.
load_candidate should return the restored object without nominal class identity checks.

Preserve prediction kind/class order. Save portable state; inference uses saved transforms.
Stacking uses OOF base predictions and fold-local preprocessing; fit_stack receives raw X
and create_preprocessor. Persist wrapped bases/meta learner; stack predict also takes raw X.
CatBoost after_iteration returns True to continue and False to stop.
