---
name: imbalance
description: Compare task-compatible weights, fold-local sampling and classification thresholds.
---
# Weighting and imbalance

Regression: no class balancing, SMOTE or classification thresholds. For single-target regression,
declare runtime weight_policy=tail_aware when justified: fold-local target quantile, combined
with supplied sample weights. Compare global and both tail metrics on identical splits.
Multioutput regression may consume supplied weights but has no default tail-aware policy.

Classification: inspect class counts, minority support, fold coverage, overlap/conflicting labels
and the actual metric. Compare with an unweighted control on identical folds using per-class
precision/recall/confusion and requested metric. Accuracy alone does not establish rare-class quality.
Declare balanced/sqrt_balanced in runtime weight_policy, separately from native class_weight.
Pass context.sample_weight unchanged; never recompute runtime weights or alter attributes after fit.

Prefer weights before synthesis; avoid automatically combining aggressive weights and sampling.
Over/undersampling belongs to the estimator's fold-local imbalanced-learn Pipeline, not
preprocessing.py. Align X/y/groups/weights; reject samplers unable to preserve required metadata.
Validation/inference keep natural prevalence. Declare resampling as a separate intervention.
SMOTE needs a meaningful scaled numeric space and enough neighbours. Use SMOTENC for mixed
features or SMOTEN for categorical features only when the pinned API supports the task. Never
interpolate one-hot coordinates blindly. Conservative ratios need ablations against simple sampling.
Borderline/ADASYN/cleaning may amplify noise. Balanced forests/bagging also require a paired test.

Thresholds use internal validation or valid group-safe OOF, never outer holdout/test. Preserve
original labels and context class order; map binary decisions through classes, not hardcoded 0/1.
Threshold tuning is for label metrics, not AUC/AP/log_loss. If probabilities matter, check calibration
on untouched natural prevalence. Persist/export explicit custom thresholds for fixed refit.
Synthetic samplers are not assumed multilabel-compatible; use supported per-label policies and
constant-label fallbacks. Verify sampler/threshold APIs in the pinned environment.
