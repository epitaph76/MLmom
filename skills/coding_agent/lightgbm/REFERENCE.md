> Role boundary: the SDK supplies already transformed X and validation_X. Any feature cleaning/encoding in the standalone API excerpts below belongs to the preprocessing agent; omit it from model code. Use preprocessing_spec for category indices/dtypes. Persist only estimator and output/target state; SDK persists preprocessing.

# Direct LightGBM implementation reference

Implement these operations in your own Candidate. Numeric Year_Factor is not categorical automatically; add exact columns required by the hypothesis.
```python
X = X.copy().replace([np.inf, -np.inf], np.nan)
category_columns = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
# If planned: category_columns += ['Year_Factor']
self.categories = {c: X[c].dropna().astype(str).unique().tolist() for c in category_columns}
for c, levels in self.categories.items():
    X[c] = pd.Categorical(X[c].map(lambda v: str(v) if pd.notna(v) else np.nan), categories=levels)
cls = LGBMClassifier if 'classification' in context.task_type else LGBMRegressor
self.model = cls(**(dict(n_estimators=300, verbosity=-1, random_state=context.seed) | self.params | dict(n_jobs=context.cpus)))
kwargs = {'sample_weight': context.sample_weight} if context.sample_weight is not None else {}
# Transform internal validation using the SAME category levels.
# Exact RMSE: metric='None', eval_metric='rmse', eval_set=[(valid, context.validation_y)],
# callbacks=[early_stopping(50, first_metric_only=True, verbose=False)].
# Macro F1: custom sklearn.metrics.f1_score callback, average='macro'. Truth is encoded:
# map truth and prediction using TRAIN label ordering. Respect validation_weight.
self.model.fit(X, y, **kwargs)
# export_model_params: deep-copy ALL self.params; only for permitted early stopping
# replace frozen['n_estimators'] with measured best_iteration_ (>0). Never return a count-only patch.
```
Persist columns, category levels, model and classes. Refit without validation uses selected tree count. Align probability columns to context.classes. Row bagging requires subsample_freq>0. Multioutput wrappers are serial; ranking sorts X/y/weights/groups together and uses contiguous query sizes. Requested clipping needs explicit train-derived bounds and predict code.
