> Role boundary: the SDK supplies already transformed X and validation_X. Any feature cleaning/encoding in the standalone API excerpts below belongs to the preprocessing agent; omit it from model code. Use preprocessing_spec for category indices/dtypes. Persist only estimator and output/target state; SDK persists preprocessing.

# Direct AutoGluon reference
```python
from autogluon.tabular import TabularPredictor
label = '__target__'  # reject collision with a real feature
train = X.assign(**{label: np.asarray(y)})
valid = context.validation_X.assign(**{label: np.asarray(context.validation_y)}) if context.validation_X is not None else None
self.model = TabularPredictor(label=label, path=str(Path(context.work_dir)/'predictor'),
    eval_metric='root_mean_squared_error', verbosity=0)  # map the exact requested metric
self.model.fit(train, tuning_data=valid, time_limit=max(1.,context.remaining_seconds),
    hyperparameters={'GBM':{},'CAT':{},'XGB':{},'RF':{},'XT':{}}, fit_weighted_ensemble=False,
    num_cpus=context.cpus, num_gpus=0)
```
Adapt metric/class order; restrict the families above, no pretrained models. Export best family and parameters as __native_frozen. Fixed refit selects only that configuration and uses refit_full rather than searching families again. Save clones predictor files inside artifact_dir; load with TabularPredictor.load from the relocated directory, not work_dir. Do not keep a work_dir-only pickle. No outer Optuna.
