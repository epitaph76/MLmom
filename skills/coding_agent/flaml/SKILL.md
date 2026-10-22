---
name: flaml
description: Run bounded native FLAML 2.3.6 search and refit the frozen selected configuration in generated candidate code.
---
# FLAML 2.3.6

Use strategy='native', patience_trials=null for regression/binary/multiclass. Mappings: rmse, mae, f1_macro -> macro_f1, roc_auc, accuracy, log_loss. Verify task/metric compatibility, especially multiclass AUC. Upstream ranking/forecasting is outside the supported task contract.

The generated candidate must implement AutoML.fit, internal X_val/y_val, time_budget, n_jobs, seed, logs and estimator_list=['lgbm','rf','extra_tree']. max_trials maps to max_iter. Do not duplicate reserved options in model_params, add families or wrap FLAML in outer search. Inspect the pinned API before extra options.

Only explicit holdout is verified in the generated path. Reuse supplied validation with eval_method=holdout, ensemble=False and retrain_full=False. Do not select native CV or grouped native validation; upstream capabilities do not establish support here. Native history exposes trial log/best config. Native loss may be 1-score; the runner reports the original metric.

Keep exported __native_frozen estimator/config and historical loss. Fixed refit uses retrain_from_log on permitted train rows, not new AutoML.fit search. Persist complete inference state. Prefer a direct booster if native startup cannot fit the budget.
