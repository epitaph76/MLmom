---
name: candidate-sdk
description: Implement the separate preprocessing and estimator contracts composed by the MLmom SDK.
---
# Separate candidate components

The planner assigns implementation. preprocessing.py exports create_preprocessor(config), returning a fresh object with fit_transform(X,y,context) and transform(X). Preserve rows/order; use training groups and weights for group-safe OOF statistics. This context never includes validation rows/labels. Store feature order and train-fitted inference maps. The SDK persists the object with cloudpickle; never persist FitContext or training records.

candidate.py exports create_candidate(config) and load_candidate(artifact_dir). The model implements fit(X,y,context) returning self, predict(X,prediction_spec) returning an array, and save(artifact_dir). The SDK already transformed X and context.validation_X. Do not fit feature transforms again or import preprocessing.py. Follow preprocessing_spec's dataframe/dense/sparse contract; adapt standalone backend reference excerpts accordingly.

FitContext is a dataclass: task_type, metric, direction, seed, cpus, memory_gb, time_limit_seconds, remaining_seconds, classes, groups, sample_weight, validation_X/y, validation_groups/weight, work_dir. Use attributes. The model coder owns target/output transforms and class order. Fixed refit can have no validation. export_model_params() returns selected tree counts/epochs/thresholds consumed without retuning. native_history() is optional.

The SDK creates fresh preprocessing and model objects for each fit, composes prediction and persists both. Save estimator files relative to artifact_dir; native paths must remain portable. The runner owns Optuna, splits and scoring; the checkpoint training agent controls search. No HPO inside fit. No training at import/load/predict. Legacy candidates without preprocessing.py retain the combined SDK.
