---
name: autogluon
description: Use explicit non-pretrained AutoGluon Tabular 1.3.1 families and portable frozen refit.
---
# AutoGluon 1.3.1

Regression/binary/multiclass use strategy=native and patience_trials=null. Restrict the reference
to GBM/CAT/XGB/RF/XT, CPU resources, fit_strategy=sequential and no native weighted ensemble.
Mapped objectives are RMSE, MAE, macro-F1, accuracy, AUC and log-loss. The complete reference is
unweighted, with identity policies/default metric options. Other objectives/policies need an
exact custom recipe or a direct fixed estimator; installed upstream support alone is not proof.
validation_strategies lists measured protocols; an empty list is not an automatic refusal.

Export selected family, hyperparameters and hyperparameters_fit, including trained counts.
Frozen fitting selects that configuration, disables new early-stop selection and calls refit_full.
Avoid bagging/stacking under small budgets, Ray and quality/foundation presets. A supported
memory_limit is soft; keep headroom.

Use transformed DataFrames, a collision-free label column and supplied tuning_data. Align
predict_proba columns to context.classes. Before save remove cached fit data, clone predictor
files into artifact_dir, then rebase with TabularPredictor.load after relocation.