---
name: catboost
description: Fit CPU CatBoost 1.2.8 with native categories and internal-validation early stopping.
---
# CatBoost 1.2.8

Implement native categories, stable missing-category strings and, only when timeouts_enabled, a time callback in the generated candidate; REFERENCE.md gives direct-library examples. Numeric NaN is allowed. Persist columns, categories and identical fit/predict cleaning.

Single-output regression/classification use context.validation_X/y and 50-round early stopping only when the plan stopping_policy permits it and use_best_model=True for supported exact metrics: RMSE, MAE, Accuracy, Logloss/MultiClass, macro F1. Other metrics disable stopping rather than substitute a proxy. Export the complete configuration; for permitted early stopping, freeze tree_count_ as iterations; refit without validation uses fixed iterations.

Start depth 4-8, learning_rate .03-.1, l2_leaf_reg 3-10 and bounded iterations. Keep allow_writing_files=False. Never wrap candidate.fit in another HPO loop.

Multioutput uses MultiRMSE; multilabel uses MultiLogloss. Ranker query groups must be contiguous; sort X/y/weights together. sample_weight is not group_weight. Use one deliberate class-weighting mechanism. CPU/thread_count come from context.

## Verified callback and full export
Only install a callback when context.timeouts_enabled. A minimal callback is:
```python
class TimeLimit:
    def after_iteration(self, info):
        return context.remaining_seconds > 0  # True CONTINUES; False STOPS
if context.timeouts_enabled:
    kwargs['callbacks'] = [TimeLimit()]
```
export_model_params returns a deep copy of ALL configured params, replacing iterations with tree_count_ only for explicitly allowed early stopping. For fixed fitting retain the requested limit; save/load must retain the learned estimator. No fitted-component audit is required.
