---
name: candidate-sdk
description: Fold-local preprocessing, estimators and reproducible ModelSpec export.
---
# Candidate SDK

Runtime v7 records FitExecutionFacts. execution_context separates config/facts/scores
from historical prose. Missing manifests are unknown; candidate APIs are unchanged.

candidate.py exports create_candidate(config) and load_candidate(artifact_dir).
Objects implement fit(X,y,context), predict(X,prediction_spec), save(artifact_dir).
preprocessing.py exports create_preprocessor(config), returning fit_transform(X,y,context)
and transform(X). Keep rows/order and fresh train-fitted state for every fold/refit.
Runtime owns target/weight/prediction policies; do not apply them twice.
SDK appends feature programs/DSL once, with fold train labels and saved transform state.

FitContext provides task_type, metric, direction, seed, cpus, memory_gb, time_limit_seconds,
remaining_seconds, timeouts_enabled, classes, groups, sample_weight, validation_X/y,
validation_groups/weight and work_dir. Use attributes. Threads use context.cpus.
Enforce deadlines only when timeouts_enabled; remaining_seconds=null means unbounded time. Native search budgets use time_limit_seconds; never pass infinity to a library. Workers receive no outer labels or LLM keys.

Optional export_model_params returns finite JSON; otherwise refit uses executed parameters.
No fitted_components, audit or export/recreate requirement. check_source checks syntax/factories;
smoke is optional. Errors return with traceback. Load objects without nominal class checks.

Preserve prediction kind/class order. Save portable state; inference uses saved transforms.
Stacking uses OOF base predictions and fold-local preprocessing; fit_stack receives raw X
and create_preprocessor. Persist wrapped bases/meta learner; stack predict also takes raw X.
CatBoost after_iteration returns True to continue and False to stop.

Search-group metadata is controller-owned/read-only; FitContext.groups means validation rows.
