---
name: candidate-sdk
description: Fold-local preprocessing, estimators and reproducible ModelSpec export.
---
# Candidate SDK

preprocessing.py: create_preprocessor(config) returns a fresh object with fit_transform(X,y,context),
transform(X). Keep rows/order; train-only statistics; target encoding needs group-safe OOF.
SDK hides validation labels and saves state. candidate.py exports create_candidate(config),
load_candidate(dir); objects implement fit(X,y,context)->self, predict(X,prediction_spec), save(dir).
load_candidate: omit nominal Candidate type checks.
X follows preprocessing_spec; never preprocess twice. Runtime owns target/weight/prediction
policies; pass context.sample_weight unchanged; preserve class order.

FitContext: task_type/metric/direction, seed/cpus/memory_gb, time_limit_seconds,
remaining_seconds/timeouts_enabled, classes/groups/sample_weight, validation_X/y/groups/weight,
work_dir. Fixed refit has no validation. Threads use context.cpus; runtime rebinds CPU hints
and audits quota separately from model settings.

export_model_params: complete finite backend JSON with nested overrides. SDK adds
policies/hashes in ModelSpec, excluding CPU thread counts. Only planned stopping may freeze
measured selected iterations. Expose model, or fitted_components() for all named base/meta
estimators; parameter_routes maps flat search keys. Smoke verifies recreate/refit,
predictions and relocation.

CatBoost after_iteration: True continues, False stops. Deadline callbacks only when enabled.
Torch: context.track_optimizer(optimizer,batch_size) establishes epoch counters.
Stacking: fit_stack(X,y,context,create_preprocessor) takes RAW X; use fresh PreparedCandidate
bases inside OOF and refit, slice groups/weights, clear validation metadata, persist bases/meta.
Stack predict takes RAW X; ordinary fit is estimator-only.
