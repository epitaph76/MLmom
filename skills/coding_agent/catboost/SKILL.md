---
name: catboost
description: Fit CPU CatBoost 1.2.8 with native categories and exact or fixed stopping.
---
# CatBoost 1.2.8

Preprocessing supplies stable string/sentinel categories and numeric NaN. Consume its categorical
names/indices without fitting another cleaner. Set task_type=CPU, thread_count=context.cpus and
allow_writing_files=False. Typical starting ranges are depth 4-8 and learning_rate .03-.1.

REFERENCE.md is a complete single-target candidate. Exact native stopping is used only for its
supported metric with identity target/output policies and default options. Otherwise fit fixed
iterations and record the reason; a custom exact callback can extend the recipe. Only install
deadline callbacks when timeouts_enabled. after_iteration returns True to continue, False to stop.
An optional export retains all parameters and selected iterations when stopping actually ran.

Multioutput uses MultiRMSE; multilabel uses MultiLogloss. Ranking needs contiguous query groups
and joint X/y/weight sorting. sample_weight is not group_weight. Choose one class-weight mechanism.
Persist the native model and target mapping, without validation callback closures.