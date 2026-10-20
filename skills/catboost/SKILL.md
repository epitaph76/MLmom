---
name: catboost
description: Train CPU CatBoost 1.2.8 for categorical-heavy tabular prediction, multioutput targets or ranking.
---
# CatBoost 1.2.8

Prefer for many/high-cardinality categories. LibraryCandidate keeps native categories, fills them with a string sentinel and uses a time callback. It supplies no eval_set; validation early stopping needs custom code.

Use identical category string conversion/sentinel at fit/predict: categorical float/NaN values are invalid, numeric NaN is allowed. Persist category columns/order. Start depth 4-8, learning_rate .03-.1, l2_leaf_reg 3-10 and bounded iterations. Keep allow_writing_files=False or confine native files to work_dir.

Custom early stopping uses context.validation_X/y, cat_features, early_stopping_rounds and use_best_model=True. Without validation disable use_best_model and train fixed iterations. Export tree_count_ as iterations for refit.

The adapter uses MultiRMSE for multioutput regression and MultiLogloss for multilabel. Ranker fit/Pool group_id must be contiguous; permute X/y/weights together. Do not confuse sample_weight with group_weight. Use one deliberate class-weighting mechanism. CPU/thread_count belong in code, not plan overrides.
