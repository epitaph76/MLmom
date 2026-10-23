---
name: candidate-sdk
description: Compose fold-local preprocessing and estimators with reproducible ModelSpec export.
---
# Candidate SDK

preprocessing.py exports create_preprocessor(config): fresh fit_transform(X,y,context) and
transform(X). Preserve rows/order. Fit vocabularies/statistics only on current train; target
encoding needs group-safe OOF training values. SDK withholds validation labels and persists state.

candidate.py exports create_candidate(config), load_candidate(artifact_dir). Implement
fit(X,y,context)->self, predict(X,prediction_spec)->array and save(artifact_dir). X/validation_X
already follow preprocessing_spec (dataframe/dense/sparse); never repeat preprocessing.
Runtime applies target_transform, weight_policy and prediction_policy. Pass context.sample_weight
to backend fit unchanged. Own class ordering and other explicitly declared estimator behavior.

FitContext attributes: task_type/metric/direction, seed/cpus/memory_gb, time_limit_seconds,
remaining_seconds/timeouts_enabled, classes/groups/sample_weight, validation_X/y/groups/weight,
work_dir. Fixed refit has no validation.

export_model_params returns complete finite JSON backend settings including nested component
parameters and backend overrides. SDK adds policies/source hashes in ModelSpec. Only planned
stopping may freeze counts to measured selected iterations. Smoke tests recreate/refit,
configuration, predictions and relocation.
Expose model for one estimator; ensembles expose fitted_components() for every named fitted
base/meta estimator, matching plan names. parameter_routes maps flat component search keys.

CatBoost after_iteration: True continues, False stops. Only add deadline callbacks when enabled.
Torch: call context.track_optimizer(optimizer,batch_size) before training; counters establish epochs.
Stacking uses fit_stack(X,y,context,create_preprocessor) with RAW features. Create fresh
PreparedCandidate bases in OOF and final refit; slice groups/weights and clear validation metadata.
Persist bases/meta learner; predict receives RAW X. Ordinary fit is estimator-only.
