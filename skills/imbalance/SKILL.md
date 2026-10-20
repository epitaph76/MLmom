---
name: imbalance
description: Address rare classes with deliberate weighting, fold-local sampling and internally selected thresholds.
---
# Imbalance

Inspect class counts and metric; keep an unweighted baseline. Try one mechanism: class_weight, CatBoost auto_class_weights or positive-class weighting. Do not automatically combine weights and oversampling. Honor supplied sample weights and check estimator support.

Resample only training rows after splitting, never before CV or on validation. SMOTE needs adequate neighbors and suitable numeric representation; do not assume categorical/sparse support. Use already installed samplers and a persisted custom pipeline when needed.

Threshold tuning is for label metrics, not AUC/AP/log_loss output. Fit on X/y and select using internal validation or valid OOF predictions; never in-sample or outer holdout scores. TunedThresholdClassifierCV must respect this separation and group folds.

Map binary decisions to context.classes[0/1], not hardcoded 0/1 for string targets. Multilabel may need per-target thresholds and constant-label fallbacks. Custom candidates must export thresholds and consume them during fixed refit to avoid resetting the policy. Weighting may harm probability calibration; assess the requested metric.
