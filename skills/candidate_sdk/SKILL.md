---
name: candidate-sdk
description: Implement MLmom candidate factories, FitContext handling and portable fit/predict/save/load state for every backend.
---
# Candidate SDK

candidate.py exports create_candidate(config) and load_candidate(artifact_dir). The object implements fit(X, y, context) returning self, predict(X, prediction_spec) returning an array, and save(artifact_dir). artifact_dir is a directory. config is the validated plan; estimator options are in model_params.

mlmom.sdk.FitContext exposes task_type, metric, direction, seed, cpus, memory_gb, time_limit_seconds, classes, groups, sample_weight, validation_X, validation_y, validation_groups, work_dir, task_context, started and remaining_seconds (property). Read attributes, not dictionary keys. task_context may be None; unsupervised y may be None.

A stock candidate is:
```python
from mlmom.adapters import LibraryCandidate, load_adapter
def create_candidate(config):
    return LibraryCandidate(config)
def load_candidate(artifact_dir):
    return load_adapter(artifact_dir)
```
Use it only if its actual model/representation implements the hypothesis. Custom candidates persist columns, transforms, model, classes/outputs and thresholds. Bundle native model files and rebase paths on load. Do not persist FitContext training/validation records.

Implement optional export_model_params() when fitting selects tree counts, epochs, thresholds or native configs: return JSON-safe fixed parameters consumed by create_candidate during validation-free refit. native_history() may return JSON-safe diagnostics. No training at import/load/predict. Verify save/load prediction equality before returning source.
