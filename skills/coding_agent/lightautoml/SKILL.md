---
name: lightautoml
description: Use LightAutoML 0.4.2 untuned tabular learners with explicit objectives and fixed refit.
---
# LightAutoML 0.4.2

Use strategy=native, patience_trials=null for regression/binary/multiclass. Objectives must match
the task: RMSE/AUC/log-loss. Task('reg') defaults to MSE upstream; REFERENCE.md supplies an explicit
RMSE callable. A matching name alone does not prove the native objective is correct.
validation_strategies describes measured recipes; unverified is not a Python admission veto.

The complete unweighted reference selects one explicit untuned lgb or cb learner with fixed
num_trees/defaults, preventing blend-weight reselection on refit. Broader multi-learner presets
need their own frozen component/weight implementation. No *_tuned algorithms or outer Optuna.
The reference requires identity policies/default options during native selection; use an exact
custom recipe or a direct fixed estimator for other combinations.

Pass supplied valid_data and target roles; keep service columns out of X. predict(X).data gives
scores; expand binary one-column probabilities and inverse-map labels. Set cpu_limit/memory_limit
and gpu_ids=''. Frozen train-as-validation only satisfies the preset API; its score is not evidence.
Persist inference state; no text/pretrained pipeline is included.