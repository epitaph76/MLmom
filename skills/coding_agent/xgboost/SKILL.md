---
name: xgboost
description: Train regularized CPU histogram boosters and query rankers with the pinned XGBoost 2.1.4 API.
---
# XGBoost 2.1.4

Generate CPU hist code with an explicit representation: fitted OHE or native categorical dtype. Persist the target LabelEncoder and use serial multioutput wrappers. Implement internal-validation early stopping and frozen iteration count when planned. REFERENCE.md gives pinned-API examples; do not assume category recoding from newer versions.

Start learning_rate .03-.1, max_depth 3-8, min_child_weight 1-10, subsample/colsample_bytree .7-1, reg_lambda 1-10. Bound stock n_estimators. Custom code sets early_stopping_rounds/eval_metric in the constructor, passes internal eval_set to fit, and disables stopping on validation-free refit. Export best_iteration+1 as n_estimators. sklearn predict uses best_iteration; native Booster needs iteration_range.

Native categories use pandas category, enable_categorical=True and identical train-derived vocabularies/codes. Native serialization needs JSON/UBJSON plus transforms/class metadata. Encode single-label y contiguously; inverse-map labels and align probabilities.

Ranking needs sorted qid and joint permutation of X/y/metadata. Ranking weights are per-query: the stock adapter forwards sample_weight unchanged, so weighted ranking needs a compatible custom path or rejection. Preserve prediction row order.

export_model_params returns ALL model parameters and output/target options. For permitted early stopping freeze the measured selected round count, preserving every other setting. Return the selected parameters when useful for refit. No fitted-estimator audit or export/recreate check is performed.
