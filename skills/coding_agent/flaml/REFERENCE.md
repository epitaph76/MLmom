> Role boundary: the SDK supplies already transformed X and validation_X. Any feature cleaning/encoding in the standalone API excerpts below belongs to the preprocessing agent; omit it from model code. Use preprocessing_spec for category indices/dtypes. Persist only estimator and output/target state; SDK persists preprocessing.

# Direct FLAML reference
Instantiate flaml.AutoML in the generated Candidate. strategy=native owns the single search.
```python
from flaml import AutoML
self.model = AutoML()
self.model.fit(X_train=X, y_train=y, X_val=context.validation_X, y_val=context.validation_y,
    task='classification' if 'classification' in context.task_type else 'regression',
    metric='macro_f1' if context.metric in {'macro_f1','f1_macro'} else context.metric,
    time_budget=max(1., context.remaining_seconds), n_jobs=context.cpus, seed=context.seed,
    estimator_list=['lgbm','rf','extra_tree'], max_iter=config.get('max_trials', 15),
    log_file_name=str(Path(context.work_dir)/'flaml-trials.jsonl'), log_type='all', verbose=0)
```
Implement predict/predict_proba ordering, inference-state persistence and native_history. export_model_params exports __native_frozen with best_estimator, best_config and best_loss. For a fixed refit write that selected configuration via flaml.automl.training_log.training_log_writer and call retrain_from_log(record_id=0, train_full=True) on allowed rows; never run another search. Generate this logic explicitly; infrastructure does not supply it. No outer Optuna.
