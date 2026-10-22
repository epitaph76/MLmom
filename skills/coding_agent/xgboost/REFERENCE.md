> Role boundary: the SDK supplies already transformed X and validation_X. Any feature cleaning/encoding in the standalone API excerpts below belongs to the preprocessing agent; omit it from model code. Use preprocessing_spec for category indices/dtypes. Persist only estimator and output/target state; SDK persists preprocessing.

# Direct XGBoost implementation reference

XGBoost 2.1.4: implement either native categories with enable_categorical=True and persisted train-derived levels, or fitted OHE. Numeric-coded categories need explicit column lists. Numeric NaN can remain missing; do not impute if the hypothesis preserves missing routing.
```python
cls = XGBClassifier if 'classification' in context.task_type else XGBRegressor
self.model = cls(**(dict(n_estimators=300, tree_method='hist', random_state=context.seed) | self.params |
    dict(n_jobs=context.cpus, device='cpu')))
# Classifier needs contiguous labels: fit/persist LabelEncoder on TRAIN y.
kwargs = {'sample_weight': context.sample_weight} if context.sample_weight is not None else {}
# For internal stopping set early_stopping_rounds in the CONSTRUCTOR (2.1.4),
# eval_set=[(prepared_valid, encoded_valid_y)] in fit. Choose exact metric and direction.
# Custom maximizing metrics require correct maximizing callback behavior.
self.model.fit(prepared_train, encoded_train_y, **kwargs)
# export_model_params: n_estimators=int(self.model.best_iteration)+1 after early stopping.
# Refit without validation removes early_stopping_rounds/callbacks and trains fixed count.
```
Inverse-map labels; align predict_proba with context.classes, not encoded classes. Multioutput wrappers are serial. Ranking needs sorted qid and correctly defined query weights, not arbitrary row weights. Persist transforms, encoder, model and tree count. Requested clipping is a separate predict operation using bounds learned from train.
