> Role boundary: the SDK supplies already transformed X and validation_X. Any feature cleaning/encoding in the standalone API excerpts below belongs to the preprocessing agent; omit it from model code. Use preprocessing_spec for category indices/dtypes. Persist only estimator and output/target state; SDK persists preprocessing.

# Direct CatBoost implementation reference
```python
self.columns = X.columns.tolist()
self.categories = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
X = X.copy().replace([np.inf, -np.inf], np.nan)
for c in self.categories:
    X[c] = X[c].fillna('__missing__').astype(str)
cls = CatBoostClassifier if 'classification' in context.task_type else CatBoostRegressor
self.model = cls(**(dict(iterations=300, verbose=False, random_seed=context.seed) | self.params |
    dict(thread_count=context.cpus, task_type='CPU', allow_writing_files=False)))
kwargs = dict(cat_features=self.categories)
if context.sample_weight is not None:
    kwargs['sample_weight'] = context.sample_weight
# Clean validation with identical columns and category policy.
# Exact supported metric: eval_set=(valid, context.validation_y), early_stopping_rounds=50, use_best_model=True.
# Macro F1: TotalF1:average=Macro;use_weights=false for unweighted validation.
# Weighted validation: Pool(valid, validation_y, cat_features=..., weight=validation_weight).
self.model.fit(X, y, **kwargs)
# export_model_params: iterations=int(self.model.tree_count_).
```
Use identical predict cleaning and class-order alignment. Persist inference state only. Set boosting_type='Ordered' explicitly if promised. Multioutput uses MultiRMSE, multilabel MultiLogloss; ranking uses sorted group_id and aligned rows/weights. Use a time callback checking context.remaining_seconds. Validation-free refit uses fixed iterations; no HPO inside fit.
