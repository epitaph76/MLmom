---
name: imbalance
description: Compare weighting, fold-local sampling, balanced ensembles and thresholds for rare classes.
---
# Imbalanced classification

Inspect per-class counts, proportions, absolute minority size, fold coverage, overlap, conflicting labels and the requested metric. Accuracy alone is not evidence of rare-class quality. Keep an unweighted baseline and compare alternatives on identical folds using the task metric plus per-class precision/recall and confusion diagnostics when available.

Try class or sample weights before complex synthesis. Do not automatically combine aggressive weighting and resampling. Random over- and undersampling are useful controls. Sampling belongs to coding_agent inside an imbalanced-learn Pipeline or equivalent persisted estimator wrapper, after the split and fitted only on the training fold. preprocessing.py preserves rows; validation and inference retain natural prevalence. Any sampled training X/y/groups/weights must stay aligned; reject a sampler that cannot honor required weights or groups. Honor supplied sample weights and check the selected estimator API.

Use vanilla SMOTE only for a meaningful scaled numeric neighbour space with enough minority observations. Use SMOTENC for mixed numeric/categorical data and SMOTEN for categorical data when the pinned imbalanced-learn API supports the required case. Never interpolate one-hot dummy coordinates blindly. Conservative ratios are candidates; do not force 1:1. Borderline, ADASYN and cleaning methods can amplify or remove noisy boundary cases, so require an ablation against weights and simple sampling.

Balanced forests or bagging are candidates when the majority is large and redundant. Threshold tuning applies to label decisions and uses only internal validation or valid OOF scores, never the outer holdout or inference set. Preserve original class labels and context class order. If probabilities matter, assess calibration on untouched natural-prevalence data because weighting and sampling can distort posterior probabilities.

Synthetic samplers require a supported single-output target; do not assume SMOTE/SMOTENC/SMOTEN support multilabel targets. For multilabel use supported per-label weighting/thresholds and constant-label fallbacks. Map binary decisions to context.classes[0/1], never hardcoded 0/1 for string labels. Threshold tuning is for label metrics, not AUC/AP/log_loss output. TunedThresholdClassifierCV must respect internal/OOF and group separation. Custom candidates must export selected thresholds and consume them during fixed refit rather than resetting the policy. Do not assume newer sampler or threshold APIs exist in the pinned environment.
