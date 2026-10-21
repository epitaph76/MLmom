---
name: autogluon
description: Use restricted CPU AutoGluon Tabular 1.3.1 when multiple model families fit the search and refit budget.
---
# AutoGluon 1.3.1

Only regression/binary/multiclass; strategy='native', patience_trials=null. Mapped metrics: rmse, mae, f1_macro, accuracy, roc_auc, log_loss. Prefer direct boosting on short budgets. Allowed model families: GBM/CAT/XGB/RF/XT; weighted ensemble is disabled.

The generated candidate implements hyperparameters, time_limit, num_cpus, num_gpus=0, label/path and internal tuning_data. Do not duplicate reserved settings in model_params. Verified fit_strategy='sequential' and memory_limit below worker allocation may be fit options; memory_limit is soft. Avoid bagging/stacking under tight limits, parallel Ray, quality presets and pretrained/foundation families.

Align predict_proba DataFrame columns to context.classes. Keep the reserved target name out of X. Native leaderboard is diagnostic; the runner owns comparison.

Keep exported __native_frozen family/params. Fixed refit fits that family then refit_full, without family search. The predictor references native files: save must clone them into artifact_dir and load rebases paths. A pickle pointing at work_dir is insufficient; check moved artifact predictions.
