---
name: candidate-sdk
description: Implement MLmom candidate factories, FitContext handling and portable fit/predict/save/load state for every backend.
---
# Candidate SDK

candidate.py exports create_candidate(config) and load_candidate(artifact_dir). The object implements fit(X, y, context) returning self, predict(X, prediction_spec) returning an array, and save(artifact_dir). artifact_dir is a directory. config is the validated plan; estimator options are in model_params.

mlmom.sdk.FitContext exposes task_type, metric, direction, seed, cpus, memory_gb, time_limit_seconds, classes, groups, sample_weight, validation_X, validation_y, validation_groups, work_dir, task_context, started and remaining_seconds (property). Read attributes, not dictionary keys. task_context may be None; unsupervised y may be None.

Every candidate is generated source, including the first baseline. Implement library calls directly; do not import a stock adapter. Selected backend REFERENCE.md code is supplied to the coder only and must be adapted to the hypothesis. Persist columns, transforms, model, classes/outputs and thresholds. Bundle native files and rebase paths on load. Never persist FitContext or training/validation records.

The runner owns splits and HPO. A single baseline can declare search_space for later HPO without searching during its first fit. Consume all declared model/search parameters, and implement every planned cast, feature operation and output constraint in code. Prose does not execute a transformation.

Implement optional export_model_params() when fitting selects tree counts, epochs, thresholds or native configs: return JSON-safe fixed parameters consumed by create_candidate during validation-free refit. native_history() may return JSON-safe diagnostics. No training at import/load/predict. Verify save/load prediction equality before returning source.
