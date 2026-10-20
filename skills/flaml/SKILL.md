---
name: flaml
description: Run bounded native FLAML 2.3.6 search and refit the frozen selected configuration through the project adapter.
---
# FLAML 2.3.6

Use strategy='native', patience_trials=null for regression/binary/multiclass. Mappings: rmse, mae, f1_macro -> macro_f1, roc_auc, accuracy, log_loss. Verify task/metric compatibility, especially multiclass AUC. Upstream ranking/forecasting is outside this adapter.

The adapter owns AutoML.fit, internal X_val/y_val, time_budget, n_jobs, seed, logs and estimator_list=['lgbm','rf','extra_tree']. max_trials maps to max_iter. Do not duplicate reserved options in model_params, add families or wrap FLAML in outer search. Inspect the pinned API before extra options.

Reuse supplied validation. Custom CV must respect groups/classes; upstream group support does not make the stock path group-aware. Native history exposes trial log/best config. Native loss may be 1-score; the runner reports the original metric.

Keep exported __native_frozen estimator/config and historical loss. Fixed refit uses retrain_from_log on permitted train rows, not new AutoML.fit search. Persist complete inference state. Prefer a direct booster if native startup cannot fit the budget.
