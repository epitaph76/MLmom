---
name: preprocessing-core
description: Enforce fold-local, persisted and resource-aware preprocessing for every candidate.
---
# Preprocessing core

The orchestrator owns splits. Fit imputers, encoders, scalers, selectors and target statistics only on current training rows; validation and inference are transform-only. Preserve rows and order. Training resampling belongs to coding_agent and must align y, groups and weights.

Choose a representation compatible with the estimator; trees rarely need scaling. Implement planned feature transforms in preprocessing.py. Runtime owns target_transform, weight_policy and prediction_policy; never duplicate them. The SDK fits and persists preprocessing separately.

Exclude target, ID, group, fold and weight columns unless explicitly assigned as features. Handle infinities deliberately. Persist fitted vocabularies, bounds, dates, selected columns and thresholds; reuse them after load. Every fold and final refit learns fresh state from its allowed training rows, retaining the selected recipe and settings. Model parameter export belongs to the estimator.

Design for missing values, unseen categories, shuffled columns, non-default indices and single-row prediction. Optional check_preprocessing(source) exercises synthetic fit/transform/serialization without an estimator. inspect_api checks the installed signature. These are tools, not required smoke or admission checks.

Use list_skills/read_skills for a relevant preprocessing/shared card when needed; foreign role cards remain unavailable. Dynamically read card versions are retained with the candidate.

PreprocessingSpec.categorical_features declares output names for DataFrames or indices
for matrices. Use backend-specific native formats: CatBoost strings, LightGBM train-derived
category dtype/codes, XGBoost stable category dtype. FitContext has no validation scorer here.
