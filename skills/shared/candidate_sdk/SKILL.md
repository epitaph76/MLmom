---
name: candidate-sdk
description: Compose preprocessing and estimators with reproducible fit and export.
---
# Candidate SDK

preprocessing.py exports create_preprocessor(config): fresh fit_transform(X,y,context) and transform(X). Preserve rows/order; fit train-only vocabularies/statistics and group-safe OOF target encodings. SDK withholds validation labels and persists preprocessing with cloudpickle.

candidate.py exports create_candidate(config), load_candidate(artifact_dir). Implement fit(X,y,context)->self, predict(X,prediction_spec)->array and save(artifact_dir). X/validation_X are already transformed: follow preprocessing_spec's dataframe/dense/sparse contract; do not import or repeat preprocessing. Own target/output transforms, thresholds and class order. Save portable estimator state, never FitContext/training records. No import/load/predict training or HPO inside fit.

FitContext uses attributes: task_type/metric/direction, seed/cpus/memory_gb, time_limit_seconds/remaining_seconds/timeouts_enabled, classes/groups/sample_weight, validation_X/y/groups/weight and work_dir. Fixed refit has no validation.

export_model_params() returns COMPLETE finite JSON model_params, never a patch: retain all fixed/nested component params, weights, target/output options and flat HPO overrides. Deep-copy configuration; only planned early stopping may replace a count with a measured selected count. Incomplete/changed exports fail. Smoke recreates/refits and compares effective config and ordinary same-data predictions.

Expose self.model for one fitted estimator. Ensembles implement fitted_components(): every named fitted base/meta estimator, including PreparedCandidate bases. Names match plan name/tool_id; parameter_routes assigns flat ensemble HPO keys. Runtime verifies actual backend params/iterations/stopping after every fit, before evaluation/Champion/submission.

Only install deadline callbacks when timeouts_enabled. CatBoost after_iteration: True CONTINUES, False STOPS. Plan stopping_policy/component_stopping_policies govern early stopping. Torch calls context.track_optimizer(optimizer,batch_size) before training; runtime derives epochs from actual optimizer steps and fit_rows.

Stacking implements fit_stack(X,y,context,create_preprocessor) with RAW X. Fit fresh PreparedCandidate bases inside OOF folds and full refit; slice groups/weights and clear validation metadata. Persist bases/meta learner; predict receives RAW X. Ordinary fit remains estimator-only. Legacy files without preprocessing.py retain the combined contract.
