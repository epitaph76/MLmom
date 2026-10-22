---
name: imbalance
description: Compare weighting, fold-local sampling, balanced ensembles and thresholds for rare classes.
---
# Imbalanced classification

Inspect per-class counts, proportions, absolute minority size, fold coverage, overlap, conflicting labels and the requested metric. Accuracy alone is not evidence of rare-class quality. Keep an unweighted baseline and compare alternatives on identical folds using the task metric plus per-class precision/recall and confusion diagnostics when available.

Try class or sample weights before complex synthesis. Do not automatically combine aggressive weighting and resampling. Random over- and undersampling are useful controls. Every sampler belongs inside an imbalanced-learn Pipeline after the split and is fitted only on the training fold; validation and inference retain natural prevalence.

Use vanilla SMOTE only for a meaningful scaled numeric neighbour space with enough minority observations. Use SMOTENC for mixed numeric/categorical data and SMOTEN for categorical data when the pinned imbalanced-learn API supports the required case. Never interpolate one-hot dummy coordinates blindly. Conservative ratios are candidates; do not force 1:1. Borderline, ADASYN and cleaning methods can amplify or remove noisy boundary cases, so require an ablation against weights and simple sampling.

Balanced forests or bagging are candidates when the majority is large and redundant. Threshold tuning applies to label decisions and uses only internal validation or valid OOF scores, never the outer holdout or inference set. Preserve original class labels and context class order. If probabilities matter, assess calibration on untouched natural-prevalence data because weighting and sampling can distort posterior probabilities.
