---
name: lightautoml
description: Use LightAutoML 0.4.2 through the restricted untuned tabular lgb/cb preset and supported objective mappings.
---
# LightAutoML 0.4.2

Only regression/binary/multiclass; strategy='native', patience_trials=null. Supported metrics are rmse/roc_auc/log_loss but uses Task defaults: regression RMSE, binary AUC, multiclass cross-entropy. Require an exact task/metric match; an accepted name alone does not imply correct optimization.

The generated candidate implements timeout, cpu_limit, memory_limit, gpu_ids='', general_params={'use_algos':[['lgb','cb']]} and reader_params cv=3/seed. Never override them, add *_tuned algorithms or nest Optuna. Leave memory headroom: preset memory management is limited. No text/pretrained pipelines.

fit_predict uses target roles and internal valid_data. Keep service columns/reserved target out of X. predict(X).data gives scores; binary one-column probability expands to [1-p,p]. Inverse-map encoded labels.

Keep exported __native_frozen for fixed-preset refit. Its training-as-validation path fits fixed learners on allowed rows; this in-sample report is never an evaluation score. Persist all fitted inference state and check reload. Reject unsupported objectives or excessive startup cost.
