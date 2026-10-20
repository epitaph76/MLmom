---
name: xgboost
description: Train regularized CPU histogram boosters and query rankers with the pinned XGBoost 2.1.4 API.
---
# XGBoost 2.1.4

LibraryCandidate uses CPU hist, dense OHE, target LabelEncoder and serial multioutput wrappers; no eval_set. Native categories/early stopping need custom code. Do not assume category recoding from newer versions.

Start learning_rate .03-.1, max_depth 3-8, min_child_weight 1-10, subsample/colsample_bytree .7-1, reg_lambda 1-10. Bound stock n_estimators. Custom code sets early_stopping_rounds/eval_metric in the constructor, passes internal eval_set to fit, and disables stopping on validation-free refit. Export best_iteration+1 as n_estimators. sklearn predict uses best_iteration; native Booster needs iteration_range.

Native categories use pandas category, enable_categorical=True and identical train-derived vocabularies/codes. Native serialization needs JSON/UBJSON plus transforms/class metadata. Encode single-label y contiguously; inverse-map labels and align probabilities.

Ranking needs sorted qid and joint permutation of X/y/metadata. Ranking weights are per-query: the stock adapter forwards sample_weight unchanged, so weighted ranking needs a compatible custom path or rejection. Preserve prediction row order.
